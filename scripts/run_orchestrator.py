"""Main entry point for the poly-weather betting bot."""

import argparse
import asyncio
import sys

sys.path.insert(0, ".")

from config import settings
from src.strategy.orchestrator import Orchestrator
from src.utils.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="poly-weather betting orchestrator")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument(
        "--weather-interval", type=int, default=settings.weather_poll_interval_minutes,
        help="Weather poll interval in minutes",
    )
    parser.add_argument(
        "--market-interval", type=int, default=settings.market_poll_interval_minutes,
        help="Market poll interval in minutes",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level)

    orchestrator = Orchestrator(settings)

    try:
        if args.once:
            asyncio.run(orchestrator.run_once())
        else:
            asyncio.run(orchestrator.run_loop(
                weather_interval_minutes=args.weather_interval,
                market_interval_minutes=args.market_interval,
            ))
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
