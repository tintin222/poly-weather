"""WeatherAPI client for current conditions and hourly forecasts.

Requires an API key (free tier: ~1M calls/month).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from src.utils.logger import get_logger
from src.weather.cities import CityConfig

log = get_logger(__name__)

BASE_URL = "https://api.weatherapi.com/v1/forecast.json"


async def fetch(city: CityConfig, api_key: str) -> dict:
    """Fetch current weather and hourly forecast for a city.

    Returns a dict with:
      - current: dict of current conditions
      - hourly_forecast: list of dicts (today, sorted by hour)
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

    params = {
        "key": api_key,
        "q": f"{city.lat},{city.lon}",
        "days": 1,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        log.info("weatherapi_fetch", city=city.key)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        is_f = city.unit == "fahrenheit"
        cur = data.get("current", {})
        current = {
            "temperature": cur.get("temp_f") if is_f else cur.get("temp_c"),
            "feels_like": cur.get("feelslike_f") if is_f else cur.get("feelslike_c"),
            "humidity": cur.get("humidity"),
            "wind_speed": cur.get("wind_mph") if is_f else cur.get("wind_kph"),
            "condition": cur.get("condition", {}).get("text", ""),
        }

        hourly_forecast = []
        forecast_day = data.get("forecast", {}).get("forecastday", [])
        if forecast_day:
            for h in forecast_day[0].get("hour", []):
                dt = datetime.fromisoformat(h["time"])
                hourly_forecast.append({
                    "hour": dt.hour,
                    "temperature": h.get("temp_f") if is_f else h.get("temp_c"),
                    "feels_like": h.get("feelslike_f") if is_f else h.get("feelslike_c"),
                    "humidity": h.get("humidity"),
                    "condition": h.get("condition", {}).get("text", ""),
                })

        hourly_forecast.sort(key=lambda x: x["hour"])
        log.info("weatherapi_done", city=city.key, current_temp=current["temperature"],
                 forecast_points=len(hourly_forecast))

        return {
            "current": current,
            "hourly_forecast": hourly_forecast,
            "unit": city.unit,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        log.warning("weatherapi_error", city=city.key, error=str(exc))
        return empty
