"""Open-Meteo API client for live observations and hourly forecasts.

Free API, no key required. Returns actual observed data for past hours
and forecast data for future hours.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from src.utils.logger import get_logger
from src.weather.cities import CityConfig

log = get_logger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "weather_code",
]

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "weather_code",
]

# WMO weather codes → human-readable labels
WMO_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Severe thunderstorm",
}


async def fetch(city: CityConfig) -> dict:
    """Fetch current conditions and today's hourly data for a city.

    Returns a dict with:
      - current: dict of current conditions
      - hourly: list of dicts, each with an `is_forecast` bool
      - unit: 'fahrenheit' or 'celsius'
      - fetched_at: ISO timestamp
    """
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "current": ",".join(CURRENT_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": city.timezone,
        "forecast_days": 2,
        "past_days": 0,
    }
    if city.unit == "fahrenheit":
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
    else:
        params["wind_speed_unit"] = "kmh"

    log.info("open_meteo_fetch", city=city.key)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    now_iso = data["current"]["time"]  # e.g. "2026-03-20T15:00"
    now_dt = datetime.fromisoformat(now_iso)

    current = {
        "time": now_iso,
        "temperature": data["current"]["temperature_2m"],
        "feels_like": data["current"]["apparent_temperature"],
        "humidity": data["current"]["relative_humidity_2m"],
        "precipitation": data["current"]["precipitation"],
        "wind_speed": data["current"]["wind_speed_10m"],
        "wind_direction": data["current"]["wind_direction_10m"],
        "pressure": data["current"]["surface_pressure"],
        "condition": WMO_CODES.get(data["current"].get("weather_code", 0), "Unknown"),
    }

    hourly_times = data["hourly"]["time"]
    hourly = []
    for i, t in enumerate(hourly_times):
        dt = datetime.fromisoformat(t)
        hourly.append({
            "time": t,
            "hour": dt.hour,
            "date": dt.strftime("%Y-%m-%d"),
            "temperature": data["hourly"]["temperature_2m"][i],
            "feels_like": data["hourly"]["apparent_temperature"][i],
            "humidity": data["hourly"]["relative_humidity_2m"][i],
            "precipitation": data["hourly"]["precipitation"][i],
            "wind_speed": data["hourly"]["wind_speed_10m"][i],
            "wind_direction": data["hourly"]["wind_direction_10m"][i],
            "pressure": data["hourly"]["surface_pressure"][i],
            "condition": WMO_CODES.get(data["hourly"]["weather_code"][i], "Unknown"),
            "is_forecast": dt > now_dt,
        })

    log.info("open_meteo_done", city=city.key, hours=len(hourly))
    return {
        "current": current,
        "hourly": hourly,
        "unit": city.unit,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def fetch_historical(
    city: CityConfig, start_date: str, end_date: str
) -> dict[tuple[str, int], float]:
    """Fetch historical hourly temperature from Open-Meteo archive API.

    Returns a dict keyed by (date_str, hour_int) → temperature.
    Only temperatures are fetched (lightweight single-variable request).
    """
    params: dict = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": city.timezone,
    }
    if city.unit == "fahrenheit":
        params["temperature_unit"] = "fahrenheit"

    log.info("open_meteo_archive_fetch", city=city.key, start=start_date, end=end_date)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(ARCHIVE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    result: dict[tuple[str, int], float] = {}
    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])
    for t, temp in zip(times, temps):
        if temp is None:
            continue
        dt = datetime.fromisoformat(t)
        result[(dt.strftime("%Y-%m-%d"), dt.hour)] = round(temp, 1)

    log.info("open_meteo_archive_done", city=city.key, points=len(result))
    return result
