#!/usr/bin/env python3
"""Scrape today's Weather Underground data for a city and store in database."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.db.database import Database
from src.db.repositories import WeatherRepository
from src.utils.logger import get_logger, setup_logging
from src.utils.time_utils import today_in_tz
from src.weather.cities import get_city
from src.weather.models import HourlyObservation
from src.weather.scraper import WUScraper


log = get_logger(__name__)


def print_observations_table(observations: list[HourlyObservation]) -> None:
    """Print a formatted table of hourly observations."""
    if not observations:
        print("No observations found.")
        return

    header = (
        f"{'Time':<12} {'Hour':>4} {'Temp':>7} {'Dew Pt':>7} {'Humid':>6} "
        f"{'Wind':>6} {'Gust':>6} {'Dir':<5} {'Press':>9} {'Precip':>7} {'Condition'}"
    )
    print(header)
    print("-" * len(header))

    for obs in sorted(observations, key=lambda o: o.hour):
        temp = f"{obs.temperature:.1f}" if obs.temperature is not None else "-"
        dew = f"{obs.dew_point:.1f}" if obs.dew_point is not None else "-"
        hum = f"{obs.humidity:.0f}%" if obs.humidity is not None else "-"
        ws = f"{obs.wind_speed:.0f}" if obs.wind_speed is not None else "-"
        wg = f"{obs.wind_gust:.0f}" if obs.wind_gust is not None else "-"
        wd = obs.wind_direction or "-"
        pr = f"{obs.pressure:.1f}" if obs.pressure is not None else "-"
        pc = f"{obs.precip_rate:.2f}" if obs.precip_rate is not None else "-"
        cond = obs.condition or "-"

        print(
            f"{obs.time_local:<12} {obs.hour:>4} {temp:>7} {dew:>7} {hum:>6} "
            f"{ws:>6} {wg:>6} {wd:<5} {pr:>9} {pc:>7} {cond}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape today's WU weather data for a city."
    )
    parser.add_argument(
        "--city",
        required=True,
        help="City name (must match a key in CITIES, e.g. 'Ankara', 'NYC').",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date in YYYY-MM-DD format. Defaults to today.",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level)

    # Look up the city
    city = get_city(args.city)
    if not city:
        print(f"Error: Unknown city '{args.city}'. Check src/weather/cities.py.")
        sys.exit(1)

    # Determine date
    date_str = args.date or today_in_tz(city.timezone)
    print(f"Scraping {city.display_name} for {date_str}...")

    # Initialize database
    db = Database(settings.db_path)
    db.initialize()
    repo = WeatherRepository(db)

    # Initialize scraper and run
    scraper = WUScraper(headless=True)
    try:
        await scraper.start()
        observations = await scraper.scrape_daily(city, date_str)

        if observations:
            # Print summary
            print(f"\nFound {len(observations)} observations:\n")
            print_observations_table(observations)

            # Store in database
            obs_dicts = [obs.to_dict() for obs in observations]
            count = repo.upsert_hourly_observations_batch(city.key, date_str, obs_dicts)
            print(f"\nStored {count} observations in database.")

            # Compute and store daily peak
            peak_obs = max(observations, key=lambda o: o.temperature)
            repo.upsert_daily_peak(
                city_key=city.key,
                date=date_str,
                peak_temp=peak_obs.temperature,
                peak_hour=peak_obs.hour,
                observation_count=len(observations),
            )
            print(
                f"Daily peak: {peak_obs.temperature} at hour {peak_obs.hour} "
                f"({peak_obs.time_local})"
            )
        else:
            print("No observations found. The page may not have loaded properly.")

    finally:
        await scraper.stop()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
