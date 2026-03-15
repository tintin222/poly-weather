"""Main orchestrator that ties all modules together.

Runs the betting loop:
1. Scrape weather data from Weather Underground
2. Poll market data from Polymarket
3. Evaluate betting opportunities
4. Place paper/live bets when conditions are met
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.db.database import Database
from src.db.repositories import MarketRepository, WeatherRepository
from src.model.peak_predictor import PeakPredictor
from src.polymarket.gamma_api import parse_city_and_date
from src.polymarket.market_monitor import MarketMonitor
from src.polymarket.trader import PolymarketTrader
from src.strategy.evaluator import EVEvaluator
from src.strategy.risk_manager import RiskManager
from src.utils.logger import get_logger
from src.utils.time_utils import current_hour_in_tz, today_in_tz
from src.weather.cities import get_city
from src.weather.scraper import WUScraper

if TYPE_CHECKING:
    from config import Settings
    from src.weather.cities import CityConfig

log = get_logger(__name__)


class Orchestrator:
    """Main betting orchestrator with integrated WU scraper."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = Database(settings.db_path)
        self._db.initialize()

        self._weather_repo = WeatherRepository(self._db)
        self._market_repo = MarketRepository(self._db)

        self._market_monitor = MarketMonitor(self._market_repo)
        self._trader = PolymarketTrader(settings)
        self._evaluator = EVEvaluator(
            min_confidence=settings.min_confidence,
            min_expected_value=settings.min_expected_value,
        )
        self._risk_manager = RiskManager(
            market_repo=self._market_repo,
            max_bet_size=settings.max_bet_size,
            max_daily_exposure=settings.max_daily_exposure,
        )
        self._peak_predictor = PeakPredictor(self._db)

    async def run_once(self) -> None:
        """Run a single evaluation cycle: scrape weather, update markets, evaluate, bet."""
        log.info("orchestrator_cycle_start")

        # Step 1: Update market data
        await self._market_monitor.poll_all()

        # Step 2: Get all today's active cities from the DB
        cities_today = self._get_todays_cities()
        log.info("cities_to_evaluate", count=len(cities_today))

        # Step 3: Scrape weather + evaluate each city (one at a time)
        scraper = WUScraper(headless=True)
        await scraper.start()
        try:
            for city in cities_today:
                try:
                    await self._scrape_and_evaluate(city, scraper)
                except Exception as exc:
                    log.error("city_evaluation_error", city=city.key, error=str(exc))
        finally:
            await scraper.stop()

        log.info("orchestrator_cycle_complete")

    def _get_todays_cities(self) -> list[CityConfig]:
        """Get all cities that have active Polymarket events today."""
        active_events = self._market_repo.get_active_events()
        city_keys = set()
        for event in active_events:
            city_key = event["city_key"]
            city = get_city(city_key)
            if city:
                today = today_in_tz(city.timezone)
                if event["date"] == today:
                    city_keys.add(city_key)

        return [get_city(k) for k in sorted(city_keys) if get_city(k)]

    async def _scrape_and_evaluate(self, city: CityConfig, scraper: WUScraper) -> None:
        """Scrape today's WU data for a city, then evaluate for betting."""
        today = today_in_tz(city.timezone)
        current_hour = current_hour_in_tz(city.timezone)

        # Skip if too early or too late
        if current_hour < 6 or current_hour > 22:
            log.debug("outside_trading_hours", city=city.key, hour=current_hour)
            return

        # Scrape today's weather
        observations_raw = await scraper.scrape_daily(city, today)
        if observations_raw:
            obs_dicts = [obs.to_dict() for obs in observations_raw]
            self._weather_repo.upsert_hourly_observations_batch(city.key, today, obs_dicts)
            peak_obs = max(observations_raw, key=lambda o: o.temperature)
            self._weather_repo.upsert_daily_peak(
                city_key=city.key, date=today,
                peak_temp=peak_obs.temperature, peak_hour=peak_obs.hour,
                observation_count=len(observations_raw),
            )
            log.info("scraped", city=city.key, obs=len(observations_raw),
                     peak=peak_obs.temperature, peak_hour=peak_obs.hour)

        # Now evaluate with DB data
        await self._evaluate_city(city)

    async def _evaluate_city(self, city: CityConfig) -> None:
        """Evaluate betting opportunities for a single city."""
        today = today_in_tz(city.timezone)
        current_hour = current_hour_in_tz(city.timezone)

        observations = self._weather_repo.get_hourly_observations(city.key, today)
        if not observations:
            return

        max_temp = max(obs["temperature"] for obs in observations)
        peak_hour = next(
            obs["hour"] for obs in observations if obs["temperature"] == max_temp
        )

        # Train predictor
        try:
            self._peak_predictor.train(city.key, today)
        except Exception as exc:
            log.warning("predictor_train_error", city=city.key, error=str(exc))
            return

        prediction = self._peak_predictor.predict_confidence(
            city.key, current_hour, max_temp, observations
        )
        log.info("prediction", city=city.key, hour=current_hour,
                 max_temp=max_temp, confidence=round(prediction.confidence, 4))

        # Get today's active events for this city
        active_events = self._market_repo.get_active_events(today)
        city_events = [e for e in active_events if e["city_key"] == city.key]

        for event in city_events:
            brackets = self._market_repo.get_brackets_for_event(event["event_id"])
            if not brackets:
                continue

            matching = self._evaluator.find_matching_bracket(
                max_temp, [dict(b) for b in brackets]
            )
            if not matching:
                continue

            yes_price = matching.get("current_yes_price", 0)
            if not yes_price or yes_price <= 0:
                continue

            decision = self._evaluator.evaluate(
                confidence=prediction.confidence,
                market_price=yes_price,
                current_max_temp=max_temp,
                bracket_label=matching["bracket_label"],
            )

            if decision.action == "SKIP":
                log.info("bet_skipped", city=city.key,
                         bracket=matching["bracket_label"], reason=decision.reason)
                continue

            # Risk check
            position = self._risk_manager.compute_position(
                date=today, price=yes_price, kelly_fraction=decision.kelly_fraction,
            )
            if not position.approved:
                log.info("position_rejected", city=city.key, reason=position.reason)
                continue

            token_id = matching.get("clob_token_id_yes", "")
            if not token_id:
                continue

            # Place the bet
            order_result = self._trader.place_limit_buy(
                token_id=token_id, price=yes_price, size=position.size,
                tick_size=matching.get("tick_size", "0.01"),
                neg_risk=bool(matching.get("neg_risk", False)),
            )

            self._market_repo.record_bet(
                event_id=event["event_id"],
                bracket_id=matching["id"],
                city_key=city.key,
                date=today,
                bracket_label=matching["bracket_label"],
                side="BUY",
                token_id=token_id,
                price=yes_price,
                size=position.size,
                cost=position.cost,
                current_max_temp=max_temp,
                peak_hour=peak_hour,
                model_confidence=prediction.confidence,
                expected_value=decision.expected_value,
                order_id=order_result.order_id,
                order_status=order_result.status,
                is_paper=order_result.is_paper,
            )

            log.info("bet_placed", city=city.key, bracket=matching["bracket_label"],
                     price=yes_price, size=position.size, cost=position.cost,
                     order_id=order_result.order_id, paper=order_result.is_paper)

    async def run_loop(
        self,
        weather_interval_minutes: int = 30,
        market_interval_minutes: int = 5,
    ) -> None:
        """Run the orchestrator in a continuous loop."""
        log.info("orchestrator_started",
                 weather_interval=weather_interval_minutes,
                 market_interval=market_interval_minutes,
                 paper_mode=self._settings.paper_mode)

        last_weather_scrape = 0
        last_market_poll = 0

        while True:
            try:
                now = time.time()

                # Market poll (more frequent)
                if now - last_market_poll >= market_interval_minutes * 60:
                    await self._market_monitor.poll_all()
                    last_market_poll = now

                # Weather scrape + evaluation (less frequent)
                if now - last_weather_scrape >= weather_interval_minutes * 60:
                    cities_today = self._get_todays_cities()
                    if cities_today:
                        scraper = WUScraper(headless=True)
                        await scraper.start()
                        try:
                            for city in cities_today:
                                try:
                                    await self._scrape_and_evaluate(city, scraper)
                                except Exception as exc:
                                    log.error("evaluation_error", city=city.key, error=str(exc))
                        finally:
                            await scraper.stop()
                    last_weather_scrape = now

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                log.info("orchestrator_stopped")
                break
            except Exception as exc:
                log.error("orchestrator_error", error=str(exc))
                await asyncio.sleep(60)

    def shutdown(self) -> None:
        """Clean up resources."""
        self._db.close()
        log.info("orchestrator_shutdown")
