"""Weather Underground scraper using Playwright for JavaScript-rendered pages."""

from __future__ import annotations

import re
from datetime import datetime

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from src.utils.logger import get_logger
from src.weather.cities import CityConfig
from src.weather.models import DailyScrapeResult, HourlyObservation

log = get_logger(__name__)

BASE_URL = "https://www.wunderground.com/history/daily"
FORECAST_URL = "https://www.wunderground.com/hourly"


def _parse_time(time_str: str) -> tuple[int, str]:
    """Convert a time string to (hour_int, 'HH:MM') 24h format.

    Handles:
      - 12-hour AM/PM: '1:30 PM' → (13, '13:30'), '11 AM' → (11, '11:00')
      - 24-hour: '15:00' → (15, '15:00'), '0:20' → (0, '00:20')
    """
    # Normalize WU forecast page quirk: '10 :00 am' → '10:00 am'
    time_str = time_str.strip().replace(" :", ":")

    # Try 12-hour AM/PM formats first (WU default for US locale)
    for fmt in ("%I:%M %p", "%I %p"):
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.hour, dt.strftime("%H:%M")
        except ValueError:
            continue

    # Try 24-hour formats (some WU locales or non-US pages)
    for fmt in ("%H:%M", "%H"):
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.hour, dt.strftime("%H:%M")
        except ValueError:
            continue

    # Last resort: extract digits that look like H:MM or HH:MM
    match = re.match(r"(\d{1,2}):(\d{2})", time_str)
    if match:
        hour = int(match.group(1))
        minute = match.group(2)
        if 0 <= hour <= 23:
            return hour, f"{hour:02d}:{minute}"

    log.warning("unparseable_time", time_str=time_str)
    return 0, "00:00"


def _strip_numeric(value: str) -> float | None:
    """Extract the first numeric value from a string, stripping units.

    Handles values like '14 °C', '57 °F', '15 km/h', '997.61 hPa', '90 %', etc.
    Returns None if no numeric value can be found.
    """
    if not value or value.strip() in ("", "-", "--", "N/A"):
        return None
    # Match optional negative sign, digits, optional decimal
    match = re.search(r"-?\d+\.?\d*", value)
    if match:
        return float(match.group())
    return None


def _fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius, rounded to nearest integer.

    WU stores data in Fahrenheit internally and displays Celsius as whole
    numbers (e.g. 91°F → 33°C, not 32.8°C). We must match this rounding
    so our scraped values agree with what WU shows and what Polymarket
    uses to resolve markets.
    """
    return round((f - 32) * 5 / 9, 0)


def _extract_wind_direction(wind_cell: str) -> str | None:
    """Extract wind direction from a wind cell like 'NNW' or 'North'.

    WU sometimes puts direction and speed in separate columns,
    sometimes combined. This handles the direction part.
    """
    if not wind_cell or wind_cell.strip() in ("", "-", "--", "N/A", "CALM"):
        return None
    # Wind direction is typically compass abbreviation
    cleaned = wind_cell.strip()
    # If it's just compass directions (letters only), return it
    if re.match(r"^[NSEW]{1,3}$", cleaned, re.IGNORECASE):
        return cleaned.upper()
    # Try to extract direction letters at the start
    match = re.match(r"^([NSEW]{1,3})\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return cleaned


class WUScraper:
    """Playwright-based Weather Underground scraper.

    Usage:
        scraper = WUScraper()
        await scraper.start()
        observations = await scraper.scrape_daily(city_config)
        await scraper.stop()
    """

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def start(self) -> None:
        """Launch the Playwright browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        # Default context (no timezone override) — used as fallback.
        # Per-city scrapes create their own context with the correct timezone.
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        log.info("scraper_started")

    async def _create_city_context(self, city: CityConfig) -> BrowserContext:
        """Create a browser context with the city's timezone.

        WU may use browser-side JavaScript to format observation timestamps.
        Setting the correct timezone ensures times render in the station's
        local time, not in the server's system timezone (often UTC).
        """
        return await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            timezone_id=city.timezone,
            locale="en-US",
        )

    async def stop(self) -> None:
        """Close the browser and clean up."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._context = None
        self._browser = None
        self._playwright = None
        log.info("scraper_stopped")

    def _build_url(self, city: CityConfig, date: str | None = None) -> str:
        """Build the WU history URL for a city and optional date."""
        url = f"{BASE_URL}/{city.wu_url_path}"
        if date:
            # Convert YYYY-MM-DD to YYYY-M-D (WU format, no zero padding)
            parts = date.split("-")
            year = parts[0]
            month = str(int(parts[1]))
            day = str(int(parts[2]))
            url += f"/date/{year}-{month}-{day}"
        return url

    async def scrape_daily(
        self, city: CityConfig, date: str | None = None
    ) -> DailyScrapeResult:
        """Scrape hourly observation data + summary high temp for a city on a given date.

        Creates a temporary browser context with the city's timezone so
        that WU's JavaScript renders observation times in local time.

        Args:
            city: CityConfig for the target city.
            date: Date string in YYYY-MM-DD format, or None for today.

        Returns:
            DailyScrapeResult with observations list and summary_high_temp.
            summary_high_temp comes from the WU Summary Table "High Temp (Actual)"
            and may be higher than max(observations) due to ASOS readings not
            shown in the hourly table.
        """
        if not self._browser:
            raise RuntimeError("Scraper not started. Call .start() first.")

        url = self._build_url(city, date)
        log.info("scraping_page", city=city.key, date=date or "today", url=url, timezone=city.timezone)

        # Use a per-city context with the correct timezone
        city_ctx = await self._create_city_context(city)
        page: Page = await city_ctx.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)

            # Wait for the history table to render (Angular app)
            # The table is typically inside a section with observation data
            try:
                await page.wait_for_selector(
                    "table.observation-table, table[class*='observation'], "
                    "table[aria-label*='observation'], "
                    "lib-city-history-observation table, "
                    "div.observation-table table, "
                    "table tbody tr td",
                    timeout=30000,
                )
            except Exception:
                log.warning("table_selector_timeout", city=city.key, date=date)
                # Try a broader wait - just wait for any table with rows
                try:
                    await page.wait_for_selector("table tbody tr", timeout=15000)
                except Exception:
                    log.error("no_table_found", city=city.key, date=date)
                    return DailyScrapeResult(observations=[], summary_high_temp=None)

            # Extract hourly observations and summary high from the same page load
            observations = await self._extract_observations(page, city)
            summary_high_temp = await self._extract_summary_high(page, city)

            # WU returns Fahrenheit by default. Convert to Celsius for
            # cities that resolve in Celsius on Polymarket.
            if city.unit == "celsius" and observations:
                observations = self._convert_to_celsius(observations)

            log.info(
                "scrape_complete",
                city=city.key,
                date=date or "today",
                observations=len(observations),
                unit=city.unit,
                summary_high=summary_high_temp,
            )
            return DailyScrapeResult(observations=observations, summary_high_temp=summary_high_temp)
        except Exception as exc:
            log.error("scrape_failed", city=city.key, date=date, error=str(exc))
            return DailyScrapeResult(observations=[], summary_high_temp=None)
        finally:
            await page.close()
            await city_ctx.close()

    async def _extract_observations(
        self, page: Page, city: CityConfig
    ) -> list[HourlyObservation]:
        """Extract hourly observations from all tables on the page."""

        # Use JavaScript to find and parse the observation table
        # WU renders the hourly observation table via Angular;
        # we grab the rendered DOM
        table_data = await page.evaluate("""
            () => {
                // Find all tables on the page
                const tables = document.querySelectorAll('table');
                let targetTable = null;

                // Look for the table with hourly observations
                // It typically has columns: Time, Temperature, Dew Point, etc.
                for (const table of tables) {
                    const headers = table.querySelectorAll('thead th, thead td');
                    const headerTexts = Array.from(headers).map(h => h.textContent.trim().toLowerCase());
                    if (headerTexts.some(h => h.includes('time')) &&
                        headerTexts.some(h => h.includes('temp'))) {
                        targetTable = table;
                        break;
                    }
                }

                if (!targetTable) {
                    // Fallback: look for table inside observation component
                    const obsSection = document.querySelector(
                        'lib-city-history-observation, ' +
                        '[class*="observation"], ' +
                        '.daily-history-observation'
                    );
                    if (obsSection) {
                        targetTable = obsSection.querySelector('table');
                    }
                }

                if (!targetTable) {
                    // Last resort: find the largest table
                    let maxRows = 0;
                    for (const table of tables) {
                        const rows = table.querySelectorAll('tbody tr');
                        if (rows.length > maxRows) {
                            maxRows = rows.length;
                            targetTable = table;
                        }
                    }
                }

                if (!targetTable) return { headers: [], rows: [] };

                const headers = Array.from(
                    targetTable.querySelectorAll('thead th, thead td')
                ).map(h => h.textContent.trim());

                const bodyRows = targetTable.querySelectorAll('tbody tr');
                const rows = [];
                for (const row of bodyRows) {
                    const cells = row.querySelectorAll('td');
                    const cellTexts = Array.from(cells).map(c => c.textContent.trim());
                    if (cellTexts.length > 0 && cellTexts.some(t => t !== '')) {
                        rows.push(cellTexts);
                    }
                }

                return { headers, rows };
            }
        """)

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not rows:
            log.warning("no_observation_rows", city=city.key)
            return []

        log.debug("raw_table", headers=headers, row_count=len(rows))

        # Map column indices
        col_map = self._map_columns(headers)
        observations = []

        for row_cells in rows:
            obs = self._parse_row(row_cells, col_map)
            if obs:
                observations.append(obs)

        return observations

    async def _extract_summary_high(
        self, page: Page, city: CityConfig
    ) -> float | None:
        """Extract the WU Summary Table 'High Temp (Actual)' value.

        This value aggregates ALL station readings including ASOS reports not
        shown in the hourly observations table. It's what Polymarket uses for
        resolution and may be higher than max(observations.temperature).

        Returns temperature in city's unit, or None if extraction fails.
        """
        try:
            raw_value = await page.evaluate("""
                () => {
                    // Strategy 1: find a table row whose first cell says "High Temp"
                    const tables = document.querySelectorAll('table');
                    for (const table of tables) {
                        const rows = table.querySelectorAll('tr');
                        for (const row of rows) {
                            const cells = row.querySelectorAll('td, th');
                            if (cells.length >= 2) {
                                const label = cells[0].textContent.trim().toLowerCase();
                                if (label.includes('high temp')) {
                                    return cells[1].textContent.trim();
                                }
                            }
                        }
                    }

                    // Strategy 2: walk all td/th for a "High Temp" label, get next sibling
                    const allCells = document.querySelectorAll('td, th');
                    for (const cell of allCells) {
                        const label = cell.textContent.trim().toLowerCase();
                        if (label.includes('high temp')) {
                            const next = cell.nextElementSibling;
                            if (next && next.textContent.trim()) {
                                return next.textContent.trim();
                            }
                            const row = cell.closest('tr');
                            if (row) {
                                const rowCells = row.querySelectorAll('td, th');
                                if (rowCells.length >= 2) {
                                    return rowCells[1].textContent.trim();
                                }
                            }
                        }
                    }
                    return null;
                }
            """)

            if raw_value is None:
                log.debug("summary_high_not_found", city=city.key)
                return None

            value = _strip_numeric(raw_value)
            if value is None:
                log.debug("summary_high_no_numeric", city=city.key, raw=raw_value)
                return None

            # WU page renders in Fahrenheit (locale=en-US). Convert for Celsius cities.
            if city.unit == "celsius":
                value = _fahrenheit_to_celsius(value)

            log.debug("summary_high_extracted", city=city.key, raw=raw_value, value=value)
            return value

        except Exception as exc:
            log.warning("summary_high_extraction_failed", city=city.key, error=str(exc))
            return None

    @staticmethod
    def _convert_to_celsius(observations: list[HourlyObservation]) -> list[HourlyObservation]:
        """Convert temperature and dew point from Fahrenheit to Celsius."""
        converted = []
        for obs in observations:
            converted.append(HourlyObservation(
                time_local=obs.time_local,
                hour=obs.hour,
                temperature=_fahrenheit_to_celsius(obs.temperature),
                dew_point=_fahrenheit_to_celsius(obs.dew_point) if obs.dew_point is not None else None,
                humidity=obs.humidity,
                wind_speed=obs.wind_speed,
                wind_gust=obs.wind_gust,
                wind_direction=obs.wind_direction,
                pressure=obs.pressure,
                precip_rate=obs.precip_rate,
                condition=obs.condition,
            ))
        return converted

    async def scrape_forecast_high(self, city: CityConfig) -> float | None:
        """Scrape today's forecast high from WU hourly forecast page.

        Args:
            city: CityConfig for the target city.

        Returns:
            Forecast high temperature (in city's unit), or None if scraping fails.
        """
        if not self._browser:
            raise RuntimeError("Scraper not started. Call .start() first.")

        url = f"{FORECAST_URL}/{city.wu_url_path}"
        log.info("scraping_forecast", city=city.key, url=url)

        city_ctx = await self._create_city_context(city)
        page = await city_ctx.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await page.wait_for_selector("table", timeout=15000)

            # Extract all temperature values from the hourly forecast table
            temps = await page.evaluate("""
                () => {
                    const tables = document.querySelectorAll('table');
                    const temps = [];
                    for (const table of tables) {
                        const headers = Array.from(table.querySelectorAll('thead th, thead td'))
                            .map(h => h.textContent.trim().toLowerCase());
                        const tempIdx = headers.findIndex(h => h.includes('temp'));
                        if (tempIdx < 0) continue;
                        const rows = table.querySelectorAll('tbody tr');
                        for (const row of rows) {
                            const cells = row.querySelectorAll('td');
                            if (cells.length > tempIdx) {
                                const text = cells[tempIdx].textContent.trim();
                                const match = text.match(/-?\\d+/);
                                if (match) temps.push(parseInt(match[0]));
                            }
                        }
                    }
                    return temps;
                }
            """)

            if not temps:
                log.warning("no_forecast_temps", city=city.key)
                return None

            forecast_high = max(temps)

            # WU forecast page shows Celsius for non-US cities already
            # but if city is celsius and temps seem to be in F, convert
            if city.unit == "celsius" and forecast_high > 50:
                forecast_high = round((forecast_high - 32) * 5 / 9)

            log.info("forecast_scraped", city=city.key, forecast_high=forecast_high, readings=len(temps))
            return float(forecast_high)

        except Exception as exc:
            log.warning("forecast_scrape_failed", city=city.key, error=str(exc))
            return None
        finally:
            await page.close()
            await city_ctx.close()

    async def scrape_forecast_hourly(self, city: CityConfig) -> list[dict]:
        """Scrape the WU hourly forecast page and return per-hour forecast data.

        Args:
            city: CityConfig for the target city.

        Returns:
            List of dicts with keys: hour, time_local, temperature, feels_like,
            humidity, condition, wind_speed. Temperature in city's unit.
            Returns [] on failure.
        """
        if not self._browser:
            raise RuntimeError("Scraper not started. Call .start() first.")

        url = f"{FORECAST_URL}/{city.wu_url_path}"
        log.info("scraping_forecast_hourly", city=city.key, url=url)

        city_ctx = await self._create_city_context(city)
        page = await city_ctx.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            try:
                await page.wait_for_selector("table", timeout=15000)
            except Exception:
                log.warning("forecast_hourly_no_table", city=city.key)
                return []

            table_data = await page.evaluate("""
                () => {
                    const tables = document.querySelectorAll('table');
                    let targetTable = null;
                    for (const table of tables) {
                        const headers = Array.from(table.querySelectorAll('thead th, thead td'))
                            .map(h => h.textContent.trim().toLowerCase());
                        if (headers.some(h => h.includes('time')) &&
                            headers.some(h => h.includes('temp'))) {
                            targetTable = table;
                            break;
                        }
                    }
                    if (!targetTable) {
                        // Fallback: largest table
                        let maxRows = 0;
                        for (const table of tables) {
                            const rows = table.querySelectorAll('tbody tr');
                            if (rows.length > maxRows) {
                                maxRows = rows.length;
                                targetTable = table;
                            }
                        }
                    }
                    if (!targetTable) return { headers: [], rows: [] };

                    const headers = Array.from(
                        targetTable.querySelectorAll('thead th, thead td')
                    ).map(h => h.textContent.trim());

                    const bodyRows = targetTable.querySelectorAll('tbody tr');
                    const rows = [];
                    for (const row of bodyRows) {
                        const cells = row.querySelectorAll('td');
                        const cellTexts = Array.from(cells).map(c => c.textContent.trim());
                        if (cellTexts.length > 0 && cellTexts.some(t => t !== '')) {
                            rows.push(cellTexts);
                        }
                    }
                    return { headers, rows };
                }
            """)

            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])

            if not rows:
                log.warning("forecast_hourly_no_rows", city=city.key)
                return []

            col_map = self._map_columns(headers)
            # Check for a "feels like" column not covered by _map_columns
            feels_col: int | None = None
            for i, h in enumerate(headers):
                if "feel" in h.lower():
                    feels_col = i
                    break

            result = []
            for row_cells in rows:
                def get(key: str) -> str:
                    idx = col_map.get(key)
                    if idx is not None and idx < len(row_cells):
                        return row_cells[idx]
                    return ""

                time_str = get("time")
                if not time_str:
                    continue
                temp_str = get("temperature")
                temp = _strip_numeric(temp_str)
                if temp is None:
                    continue

                hour, time_24h = _parse_time(time_str)
                feels_raw = row_cells[feels_col] if feels_col is not None and feels_col < len(row_cells) else ""
                feels = _strip_numeric(feels_raw)
                humidity = _strip_numeric(get("humidity"))
                condition = get("condition") or None
                if condition and condition.strip() in ("", "-", "--"):
                    condition = None
                wind_speed = _strip_numeric(get("wind_speed"))

                if city.unit == "celsius":
                    temp = _fahrenheit_to_celsius(temp)
                    if feels is not None:
                        feels = _fahrenheit_to_celsius(feels)

                result.append({
                    "hour": hour,
                    "time_local": time_24h,
                    "temperature": temp,
                    "feels_like": feels,
                    "humidity": humidity,
                    "condition": condition,
                    "wind_speed": wind_speed,
                })

            log.info(
                "forecast_hourly_complete",
                city=city.key,
                rows=len(result),
                unit=city.unit,
            )
            return result

        except Exception as exc:
            log.warning("forecast_hourly_failed", city=city.key, error=str(exc))
            return []
        finally:
            await page.close()
            await city_ctx.close()

    def _map_columns(self, headers: list[str]) -> dict[str, int]:
        """Map column names to indices based on header text."""
        col_map: dict[str, int] = {}
        for i, header in enumerate(headers):
            h = header.lower().strip()
            if "time" in h:
                col_map["time"] = i
            elif "dew" in h:
                col_map["dew_point"] = i
            elif "temp" in h:
                col_map["temperature"] = i
            elif "humid" in h:
                col_map["humidity"] = i
            elif "gust" in h:
                col_map["wind_gust"] = i
            elif "speed" in h:
                col_map["wind_speed"] = i
            elif "wind" in h and "wind_speed" not in col_map:
                # Generic "Wind" column - could be direction or speed
                if "wind" not in col_map:
                    col_map["wind"] = i
            elif "press" in h:
                col_map["pressure"] = i
            elif "precip" in h:
                col_map["precip_rate"] = i
            elif "condition" in h:
                col_map["condition"] = i

        return col_map

    def _parse_row(
        self, cells: list[str], col_map: dict[str, int]
    ) -> HourlyObservation | None:
        """Parse a single table row into an HourlyObservation."""

        def get_cell(key: str) -> str:
            idx = col_map.get(key)
            if idx is not None and idx < len(cells):
                return cells[idx]
            return ""

        time_str = get_cell("time")
        if not time_str:
            return None

        temp_str = get_cell("temperature")
        temp = _strip_numeric(temp_str)
        if temp is None:
            return None

        hour, time_24h = _parse_time(time_str)

        # Parse optional fields
        dew_point = _strip_numeric(get_cell("dew_point"))
        humidity = _strip_numeric(get_cell("humidity"))

        # Wind: WU sometimes has separate Wind, Wind Speed, Wind Gust columns
        # or combines them
        wind_speed = _strip_numeric(get_cell("wind_speed"))
        wind_gust = _strip_numeric(get_cell("wind_gust"))

        # If there's a generic "wind" column and no wind_speed mapped
        wind_cell = get_cell("wind")
        wind_direction = None
        if wind_cell:
            wind_direction = _extract_wind_direction(wind_cell)
            # If we didn't find a specific wind_speed column, try to extract
            # speed from the generic wind column
            if wind_speed is None:
                wind_speed = _strip_numeric(wind_cell)

        pressure = _strip_numeric(get_cell("pressure"))
        precip_rate = _strip_numeric(get_cell("precip_rate"))
        condition = get_cell("condition") or None
        if condition and condition.strip() in ("", "-", "--"):
            condition = None

        return HourlyObservation(
            time_local=time_24h,
            hour=hour,
            temperature=temp,
            dew_point=dew_point,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_gust=wind_gust,
            wind_direction=wind_direction,
            pressure=pressure,
            precip_rate=precip_rate,
            condition=condition,
        )
