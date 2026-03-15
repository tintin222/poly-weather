#!/usr/bin/env python3
"""Backfill historical Weather Underground data for a city."""

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
from src.utils.logger import setup_logging
from src.weather.cities import get_city
from src.weather.historical_loader import HistoricalLoader
from src.weather.scraper import WUScraper


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical WU weather data for a city."
    )
    parser.add_argument(
        "--city",
        required=True,
        help="City name (must match a key in CITIES, e.g. 'Ankara', 'NYC').",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backfill (default: 30).",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level)

    # Look up the city
    city = get_city(args.city)
    if not city:
        print(f"Error: Unknown city '{args.city}'. Check src/weather/cities.py.")
        sys.exit(1)

    print(f"Backfilling {args.days} days of data for {city.display_name}...")

    # Initialize database
    db = Database(settings.db_path)
    db.initialize()
    repo = WeatherRepository(db)

    # Initialize scraper and run
    scraper = WUScraper(headless=True)
    try:
        await scraper.start()
        loader = HistoricalLoader(scraper, repo)
        await loader.backfill(city, days=args.days)
        print("Backfill complete.")
    finally:
        await scraper.stop()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
