"""OpenWeather API client for current conditions and 3-hour forecasts.

Requires an API key (free tier: 60 calls/min).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from src.utils.logger import get_logger
from src.weather.cities import CityConfig

log = get_logger(__name__)

CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


async def fetch(city: CityConfig, api_key: str) -> dict:
    """Fetch current weather and 3-hour forecast for a city.

    Returns a dict with:
      - current: dict of current conditions
      - hourly_forecast: list of dicts (today only, sorted by hour)
      - unit: 'fahrenheit' or 'celsius'
      - fetched_at: ISO timestamp
    Returns empty sections on error (never raises).
    """
    empty = {
        "current": {},
        "hourly_forecast": [],
        "unit": city.unit,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if not api_key:
        return empty

    units = "imperial" if city.unit == "fahrenheit" else "metric"
    unit_label = "°F" if city.unit == "fahrenheit" else "°C"
    base_params = {
        "lat": city.lat,
        "lon": city.lon,
        "appid": api_key,
        "units": units,
    }

    try:
        log.info("openweather_fetch", city=city.key)
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Current conditions
            resp = await client.get(CURRENT_URL, params=base_params)
            resp.raise_for_status()
            cur = resp.json()

            # 3-hour forecast (5 days, we filter to today)
            resp2 = await client.get(FORECAST_URL, params=base_params)
            resp2.raise_for_status()
            fc = resp2.json()

        current = {
            "temperature": cur["main"]["temp"],
            "feels_like": cur["main"]["feels_like"],
            "humidity": cur["main"]["humidity"],
            "pressure": cur["main"]["pressure"],
            "wind_speed": cur["wind"].get("speed", 0),
            "condition": cur["weather"][0]["description"] if cur.get("weather") else "",
        }

        # Filter forecast to today (in city local time)
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(city.timezone)
        today_str = datetime.now(tz).strftime("%Y-%m-%d")

        hourly_forecast = []
        for item in fc.get("list", []):
            dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            dt_local = dt_utc.astimezone(tz)
            if dt_local.strftime("%Y-%m-%d") != today_str:
                continue
            hourly_forecast.append({
                "hour": dt_local.hour,
                "temperature": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "condition": item["weather"][0]["description"] if item.get("weather") else "",
            })

        hourly_forecast.sort(key=lambda x: x["hour"])
        log.info("openweather_done", city=city.key, current_temp=current["temperature"],
                 forecast_points=len(hourly_forecast))

        return {
            "current": current,
            "hourly_forecast": hourly_forecast,
            "unit": city.unit,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        log.warning("openweather_error", city=city.key, error=str(exc))
        return empty
