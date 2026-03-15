"""CLI script to train peak temperature models.

Usage:
    .venv/bin/python -m scripts.train_models --city Ankara --date 2026-03-15
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timedelta

from config import settings
from src.db.database import Database
from src.db.repositories import WeatherRepository
from src.model.peak_predictor import PeakPredictor
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def generate_synthetic_data(
    repo: WeatherRepository,
    city_key: str,
    target_date: str,
    num_days: int = 50,
    years_back: int = 3,
) -> None:
    """Generate synthetic hourly observations for testing.

    Creates realistic temperature patterns: low at night, peak around 1-3 PM,
    decline in evening. Inserts into the database.
    """
    target = datetime.strptime(target_date, "%Y-%m-%d")

    days_generated = 0
    for year_offset in range(1, years_back + 1):
        days_per_year = num_days // years_back
        for day_offset in range(-days_per_year // 2, days_per_year // 2 + 1):
            d = target.replace(year=target.year - year_offset) + timedelta(days=day_offset)
            date_str = d.strftime("%Y-%m-%d")

            # Random base temperature and peak hour
            base_temp = random.uniform(5.0, 15.0)
            daily_range = random.uniform(8.0, 15.0)
            peak_hour = random.choice([12, 13, 13, 14, 14, 14, 15, 15])

            observations = []
            for hour in range(0, 24):
                # Sinusoidal temperature pattern peaking at peak_hour
                phase = math.pi * (hour - peak_hour) / 12.0
                temp = base_temp + daily_range * max(0, math.cos(phase))
                # Add noise
                temp += random.gauss(0, 0.5)

                observations.append({
                    "time_local": f"{hour:02d}:00",
                    "hour": hour,
                    "temperature": round(temp, 1),
                    "humidity": random.uniform(30, 80),
                    "wind_speed": random.uniform(0, 20),
                    "condition": "Partly Cloudy",
                })

            repo.upsert_hourly_observations_batch(city_key, date_str, observations)
            days_generated += 1

    log.info("synthetic_data_generated", city=city_key, days=days_generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train peak temperature models")
    parser.add_argument("--city", required=True, help="City key (e.g. Ankara)")
    parser.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic data if no historical data exists",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level)

    db = Database(settings.db_path)
    with db:
        repo = WeatherRepository(db)
        predictor = PeakPredictor(db)

        # Check if we have data
        from src.utils.time_utils import date_range_for_window, parse_date

        d = parse_date(args.date)
        day_start, day_end = date_range_for_window(args.date, window_days=10)
        existing = repo.get_historical_hourly(args.city, d.month, day_start, day_end, years_back=3)

        if not existing:
            log.info("no_existing_data", city=args.city, date=args.date)
            if args.synthetic:
                log.info("generating_synthetic_data")
                generate_synthetic_data(repo, args.city, args.date)
            else:
                print(f"No historical data for {args.city}. Use --synthetic to generate test data.")
                return

        # Train
        dist = predictor.train(args.city, args.date)

        # Print analysis
        print(f"\n{'='*60}")
        print(f"Peak Hour Analysis for {args.city} around {args.date}")
        print(f"{'='*60}")
        print(f"  Mean peak hour:   {dist.mean_peak_hour:.1f}")
        print(f"  Std peak hour:    {dist.std_peak_hour:.1f}")
        low, high = dist.percentile_range()
        print(f"  10-90% range:     {low}:00 - {high}:00")
        print()
        print(f"  CDF Table (P(peak <= hour)):")
        print(f"  {'Hour':>6}  {'P(peak occurred)':>18}")
        print(f"  {'-'*6}  {'-'*18}")
        for hour in range(6, 22):
            prob = dist.cdf(hour)
            bar = "#" * int(prob * 30)
            print(f"  {hour:>4}:00  {prob:>18.3f}  {bar}")
        print()


if __name__ == "__main__":
    main()
