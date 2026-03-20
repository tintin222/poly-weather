"""NOAA Aviation Weather METAR client.

Fetches real surface observations directly from ASOS/AWOS stations —
the same stations Weather Underground re-displays. Free, no auth required.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import httpx

from src.utils.logger import get_logger
from src.weather.cities import CityConfig

log = get_logger(__name__)

BASE_URL = "https://aviationweather.gov/api/data/metar"


def _dewpoint_to_humidity(temp_c: float, dewp_c: float) -> float:
    """Approximate relative humidity from temperature and dewpoint (Magnus formula)."""
    a, b = 17.625, 243.04
    gamma = (a * dewp_c) / (b + dewp_c)
    gamma_t = (a * temp_c) / (b + temp_c)
    return round(100 * math.exp(gamma - gamma_t), 1)


def _celsius_to_fahrenheit(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _knots_to_mph(knots: float) -> float:
    return round(knots * 1.15078, 1)


def _knots_to_kmh(knots: float) -> float:
    return round(knots * 1.852, 1)


async def fetch(city: CityConfig, hours: int = 24) -> dict:
    """Fetch the last ``hours`` METAR observations for a city.

    Returns a dict with:
      - observations: list of dicts (sorted oldest → newest), each with an ``hour`` int
      - unit: 'fahrenheit' or 'celsius'
      - station_id: ICAO station used
      - fetched_at: ISO timestamp
    """
    params = {
        "ids": city.wu_station_id,
        "format": "json",
        "hours": hours,
    }

    log.info("metar_fetch", city=city.key, station=city.wu_station_id)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        raw = resp.json()

    if not raw:
        log.warning("metar_empty", city=city.key)
        return {"observations": [], "unit": city.unit,
                "station_id": city.wu_station_id,
                "fetched_at": datetime.now(timezone.utc).isoformat()}

    observations = []
    for entry in raw:
        obs_ts = entry.get("obsTime")
        if not obs_ts:
            continue
        dt = datetime.fromtimestamp(obs_ts, tz=timezone.utc)
        # Convert to city local time for hour alignment
        from zoneinfo import ZoneInfo
        dt_local = dt.astimezone(ZoneInfo(city.timezone))

        temp_c = entry.get("temp")
        dewp_c = entry.get("dewp")
        if temp_c is None:
            continue

        wspd_knots = entry.get("wspd")
        if city.unit == "fahrenheit":
            temperature = _celsius_to_fahrenheit(temp_c)
            feels_like = _celsius_to_fahrenheit(dewp_c) if dewp_c is not None else None
            wind_speed = _knots_to_mph(wspd_knots) if wspd_knots is not None else None
        else:
            temperature = round(float(temp_c), 1)
            feels_like = round(float(dewp_c), 1) if dewp_c is not None else None
            wind_speed = _knots_to_kmh(wspd_knots) if wspd_knots is not None else None

        humidity = _dewpoint_to_humidity(temp_c, dewp_c) if dewp_c is not None else None

        observations.append({
            "time_utc": dt.isoformat(),
            "time_local": dt_local.strftime("%H:%M"),
            "hour": dt_local.hour,
            "date": dt_local.strftime("%Y-%m-%d"),
            "temperature": temperature,
            "dewpoint": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": entry.get("wdir"),
            "pressure_slp": entry.get("slp"),
            "visibility": entry.get("visib"),
            "raw": entry.get("rawOb", ""),
        })

    # Sort oldest → newest
    observations.sort(key=lambda o: o["time_utc"])

    log.info("metar_done", city=city.key, observations=len(observations))
    return {
        "observations": observations,
        "unit": city.unit,
        "station_id": city.wu_station_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
