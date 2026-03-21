#!/usr/bin/env python3
"""Backfill historical weather data for a city or all cities.

Sources:
  --source wu   (default) Weather Underground via Playwright scraping
  --source om   Open-Meteo archive API (use for cities where WU scraping fails, e.g. Hong Kong)

Examples:
  # Single city, N days back (WU)
  python scripts/backfill_historical.py --city NYC --days 30

  # Single city, specific date range (WU)
  python scripts/backfill_historical.py --city NYC --start-date 2023-03-01 --end-date 2023-04-30

  # All cities, specific date range (WU)
  python scripts/backfill_historical.py --all-cities --start-date 2023-03-01 --end-date 2023-04-30

  # Hong Kong via Open-Meteo (WU scraping broken for VHHH)
  python scripts/backfill_historical.py --city "Hong Kong" --start-date 2024-03-01 --end-date 2024-04-30 --source om
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from datetime import date as _date, timedelta
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.db.database import Database
from src.db.repositories import WeatherRepository
from src.polymarket.resolution_url_checker import get_effective_wu_url_path
from src.utils.logger import setup_logging
from src.weather.cities import CITIES, get_city
from src.weather.historical_loader import HistoricalLoader
from src.weather.models import HourlyObservation
from src.weather.scraper import WUScraper


async def backfill_city_wu(
    loader: HistoricalLoader,
    db: Database,
    city_key: str,
    days: int | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    city = get_city(city_key)
    if not city:
        print(f"  [SKIP] Unknown city key: {city_key}")
        return

    effective_path = get_effective_wu_url_path(city, db)
    if effective_path != city.wu_url_path:
        city = dataclasses.replace(city, wu_url_path=effective_path)

    if start_date and end_date:
        print(f"  Backfilling {city.display_name} (WU): {start_date} → {end_date}")
        await loader.backfill_date_range(city, start_date, end_date)
    else:
        print(f"  Backfilling {city.display_name} (WU): last {days} days")
        await loader.backfill(city, days=days)


async def backfill_city_om(
    repo: WeatherRepository,
    city_key: str,
    start_date: str,
    end_date: str,
) -> None:
    """Backfill using Open-Meteo archive API instead of WU scraping."""
    from src.weather import open_meteo

    city = get_city(city_key)
    if not city:
        print(f"  [SKIP] Unknown city key: {city_key}")
        return

    print(f"  Backfilling {city.display_name} (Open-Meteo): {start_date} → {end_date}")
    om_data = await open_meteo.fetch_historical(city, start_date, end_date)

    # Group by date
    by_date: dict[str, list[tuple[int, float]]] = {}
    for (date_str, hour), temp in om_data.items():
        by_date.setdefault(date_str, []).append((hour, temp))

    total_stored = 0
    total_skipped = 0

    cur = _date.fromisoformat(start_date)
    end = _date.fromisoformat(end_date)
    while cur <= end:
        date_str = cur.isoformat()
        cur += timedelta(days=1)

        if repo.get_daily_peak(city_key, date_str):
            total_skipped += 1
            continue

        hours = sorted(by_date.get(date_str, []))
        if not hours:
            print(f"    [WARN] No OM data for {date_str}")
            continue

        observations = []
        for hour, temp in hours:
            ampm = "AM" if hour < 12 else "PM"
            h12 = hour % 12 or 12
            observations.append(HourlyObservation(
                time_local=f"{h12}:00 {ampm}",
                hour=hour,
                temperature=temp,
            ))

        obs_dicts = [o.to_dict() for o in observations]
        count = repo.upsert_hourly_observations_batch(city_key, date_str, obs_dicts)
        total_stored += count

        peak_obs = max(observations, key=lambda o: o.temperature)
        repo.upsert_daily_peak(
            city_key=city_key,
            date=date_str,
            peak_temp=peak_obs.temperature,
            peak_hour=peak_obs.hour,
            observation_count=len(observations),
            summary_high_temp=None,
        )

    print(f"  Done. stored={total_stored} skipped={total_skipped}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical weather data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city",
        help="City name (e.g. 'Ankara', 'NYC', 'Hong Kong').",
    )
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Run for all configured cities.",
    )
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days back from yesterday (default: 30). Cannot be used with --start-date.",
    )
    range_group.add_argument(
        "--start-date",
        metavar="YYYY-MM-DD",
        help="Start of date range (inclusive). Requires --end-date.",
    )
    parser.add_argument(
        "--end-date",
        metavar="YYYY-MM-DD",
        help="End of date range (inclusive). Required when --start-date is set.",
    )
    parser.add_argument(
        "--source",
        choices=["wu", "om"],
        default="wu",
        help="Data source: 'wu' = Weather Underground (default), 'om' = Open-Meteo archive.",
    )
    args = parser.parse_args()

    if args.start_date and not args.end_date:
        parser.error("--end-date is required when --start-date is set.")
    if args.end_date and not args.start_date:
        parser.error("--start-date is required when --end-date is set.")
    if args.source == "om" and not (args.start_date and args.end_date):
        parser.error("--source om requires --start-date and --end-date.")

    setup_logging(settings.log_level)

    city_keys = list(CITIES.keys()) if args.all_cities else [args.city]

    db = Database(settings.db_path)
    db.initialize()
    repo = WeatherRepository(db)

    if args.source == "om":
        for i, city_key in enumerate(city_keys, 1):
            if args.all_cities:
                print(f"\n[{i}/{len(city_keys)}] {city_key}")
            await backfill_city_om(repo, city_key, args.start_date, args.end_date)
        print("\nAll done.")
        db.close()
        return

    scraper = WUScraper(headless=True)
    try:
        await scraper.start()
        loader = HistoricalLoader(scraper, repo)

        for i, city_key in enumerate(city_keys, 1):
            if args.all_cities:
                print(f"\n[{i}/{len(city_keys)}] {city_key}")
            await backfill_city_wu(
                loader, db, city_key,
                days=args.days,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        print("\nAll done.")
    finally:
        await scraper.stop()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
