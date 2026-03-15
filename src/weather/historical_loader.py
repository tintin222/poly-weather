"""Historical weather data loader using WU scraper and database."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.db.repositories import WeatherRepository
from src.utils.logger import get_logger
from src.weather.cities import CityConfig
from src.weather.models import DailyPeak, HourlyObservation
from src.weather.scraper import WUScraper

log = get_logger(__name__)

# Rate limit: seconds between page loads (WU handles 3s fine)
RATE_LIMIT_SECONDS = 3

# Log progress every N days
PROGRESS_LOG_INTERVAL = 10


def _compute_daily_peak(
    city_key: str, date: str, observations: list[HourlyObservation]
) -> DailyPeak | None:
    """Compute the daily peak temperature from a list of observations."""
    if not observations:
        return None

    peak_obs = max(observations, key=lambda o: o.temperature)
    return DailyPeak(
        city_key=city_key,
        date=date,
        peak_temp=peak_obs.temperature,
        peak_hour=peak_obs.hour,
        observation_count=len(observations),
    )


class HistoricalLoader:
    """Loads historical weather data from Weather Underground into the database.

    Usage:
        loader = HistoricalLoader(scraper, repo)
        await loader.backfill(city, days=365)
    """

    def __init__(self, scraper: WUScraper, repo: WeatherRepository) -> None:
        self._scraper = scraper
        self._repo = repo

    async def backfill(self, city: CityConfig, days: int = 365) -> None:
        """Scrape N days of historical data for a city.

        Goes backwards from yesterday (since today may be incomplete).

        Args:
            city: CityConfig for the target city.
            days: Number of days to go back.
        """
        tz = ZoneInfo(city.timezone)
        today = datetime.now(tz).date()

        total_stored = 0
        total_failed = 0

        log.info(
            "backfill_start",
            city=city.key,
            days=days,
            start_date=str(today - timedelta(days=1)),
            end_date=str(today - timedelta(days=days)),
        )

        for i in range(1, days + 1):
            target_date = today - timedelta(days=i)
            date_str = target_date.strftime("%Y-%m-%d")

            # Skip if we already have data for this day
            existing = self._repo.get_daily_peak(city.key, date_str)
            if existing:
                log.debug("backfill_skip_existing", city=city.key, date=date_str)
                continue

            try:
                observations = await self._scraper.scrape_daily(city, date_str)

                if observations:
                    # Store observations in batch
                    obs_dicts = [obs.to_dict() for obs in observations]
                    count = self._repo.upsert_hourly_observations_batch(
                        city.key, date_str, obs_dicts
                    )
                    total_stored += count

                    # Compute and store daily peak
                    peak = _compute_daily_peak(city.key, date_str, observations)
                    if peak:
                        self._repo.upsert_daily_peak(
                            city_key=peak.city_key,
                            date=peak.date,
                            peak_temp=peak.peak_temp,
                            peak_hour=peak.peak_hour,
                            observation_count=peak.observation_count,
                        )
                else:
                    total_failed += 1
                    log.warning("no_data_for_date", city=city.key, date=date_str)

            except Exception as exc:
                total_failed += 1
                log.error(
                    "backfill_day_failed",
                    city=city.key,
                    date=date_str,
                    error=str(exc),
                )

            # Rate limiting after each actual scrape
            await asyncio.sleep(RATE_LIMIT_SECONDS)

            # Progress logging
            if i % PROGRESS_LOG_INTERVAL == 0:
                log.info(
                    "backfill_progress",
                    city=city.key,
                    days_processed=i,
                    days_total=days,
                    observations_stored=total_stored,
                    failures=total_failed,
                )

        log.info(
            "backfill_complete",
            city=city.key,
            days_processed=days,
            observations_stored=total_stored,
            failures=total_failed,
        )
