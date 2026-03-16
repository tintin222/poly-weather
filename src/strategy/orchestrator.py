"""Main orchestrator: detects peak-passed conditions and places bets.

Strategy: when the daily peak temperature has passed with high confidence,
buy the matching bracket at a discount.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.db.database import Database
from src.db.repositories import MarketRepository, WeatherRepository
from src.model.bracket_analyzer import find_bracket_for_temp
from src.model.features import compute_current_max
from src.model.peak_predictor import PeakPredictor
from src.polymarket.market_monitor import MarketMonitor
from src.polymarket.trader import PolymarketTrader
from src.utils.logger import get_logger
from src.utils.time_utils import current_hour_in_tz, today_in_tz
from src.weather.cities import get_city
from src.weather.scraper import WUScraper

if TYPE_CHECKING:
    from config import Settings
    from src.weather.cities import CityConfig

log = get_logger(__name__)


class Orchestrator:
    """Main betting orchestrator with bracket edge analysis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = Database(settings.db_path)
        self._db.initialize()

        self._weather_repo = WeatherRepository(self._db)
        self._market_repo = MarketRepository(self._db)

        self._market_monitor = MarketMonitor(self._market_repo)
        self._trader = PolymarketTrader(settings)
        self._peak_predictor = PeakPredictor(self._db)

        # Load paper account settings
        self._bet_size = settings.bet_size
        self._max_daily_exposure = settings.max_daily_exposure
        self._min_edge = settings.min_edge

        self._init_paper_account()

    def _init_paper_account(self) -> None:
        """Ensure paper_account row exists in DB."""
        existing = self._db.conn.execute("SELECT id FROM paper_account WHERE id = 1").fetchone()
        if not existing:
            self._db.conn.execute(
                """INSERT INTO paper_account (id, initial_balance, bet_size, max_daily_exposure, min_edge, updated_at)
                   VALUES (1, ?, ?, ?, ?, ?)""",
                (self._settings.initial_balance, self._bet_size,
                 self._max_daily_exposure, self._min_edge, int(time.time())),
            )
            self._db.conn.commit()

    def _get_paper_config(self) -> dict:
        """Load paper account config from DB."""
        row = self._db.conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        if row:
            return dict(row)
        return {
            "initial_balance": self._settings.initial_balance,
            "bet_size": self._bet_size,
            "max_daily_exposure": self._max_daily_exposure,
            "min_edge": self._min_edge,
        }

    async def run_once(self) -> None:
        """Run a single evaluation cycle."""
        log.info("orchestrator_cycle_start")
        await self._market_monitor.poll_all()

        cities_today = self._get_todays_cities()
        log.info("cities_to_evaluate", count=len(cities_today))

        scraper = WUScraper(headless=True)
        await scraper.start()
        try:
            for city in cities_today:
                try:
                    await self._scrape_and_evaluate(city, scraper)
                except Exception as exc:
                    log.error("city_error", city=city.key, error=str(exc))
        finally:
            await scraper.stop()

        log.info("orchestrator_cycle_complete")

    def _get_todays_cities(self) -> list[CityConfig]:
        """Get all cities with active Polymarket events today."""
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
        """Scrape today's WU data, then evaluate for betting."""
        today = today_in_tz(city.timezone)
        current_hour = current_hour_in_tz(city.timezone)

        if current_hour < 6 or current_hour > 22:
            return

        # Scrape
        obs_raw = await scraper.scrape_daily(city, today)
        if obs_raw:
            obs_dicts = [obs.to_dict() for obs in obs_raw]
            self._weather_repo.upsert_hourly_observations_batch(city.key, today, obs_dicts)
            peak_obs = max(obs_raw, key=lambda o: o.temperature)
            self._weather_repo.upsert_daily_peak(
                city_key=city.key, date=today,
                peak_temp=peak_obs.temperature, peak_hour=peak_obs.hour,
                observation_count=len(obs_raw),
            )

        # Evaluate
        await self._evaluate_city(city)

    async def _evaluate_city(self, city: CityConfig) -> None:
        """Evaluate using peak-confidence: bet when peak has passed with high confidence."""
        today = today_in_tz(city.timezone)
        current_hour = current_hour_in_tz(city.timezone)
        config = self._get_paper_config()
        bet_size = config.get("bet_size", self._bet_size)
        max_daily = config.get("max_daily_exposure", self._max_daily_exposure)
        min_conf = config.get("min_edge", 0.85)  # reuse min_edge field as confidence threshold

        # Check daily exposure
        daily_spent = self._market_repo.get_daily_exposure(today)
        if daily_spent >= max_daily:
            return

        # Need observations
        observations = self._weather_repo.get_hourly_observations(city.key, today)
        if not observations or len(observations) < 5:
            return

        # Get current peak
        max_temp, peak_hour = compute_current_max(observations)

        # Need historical data for confidence model
        hist_count = self._db.conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
            (city.key,),
        ).fetchone()["cnt"]
        if hist_count < 5:
            return

        # Train and predict peak confidence
        try:
            self._peak_predictor.train(city.key, today)
            prediction = self._peak_predictor.predict_confidence(
                city.key, current_hour, max_temp, observations
            )
        except Exception as exc:
            log.warning("predictor_error", city=city.key, error=str(exc))
            return

        confidence = prediction.confidence
        log.info("peak_eval", city=city.key, hour=current_hour,
                 peak=max_temp, conf=f"{confidence*100:.0f}%")

        # Only bet when confidence is HIGH
        if confidence < min_conf:
            return

        # Get today's events
        active_events = self._market_repo.get_active_events(today)
        city_events = [e for e in active_events if e["city_key"] == city.key]

        for event in city_events:
            brackets = self._market_repo.get_brackets_for_event(event["event_id"])
            if not brackets:
                continue

            bracket_list = [
                {
                    "label": b["bracket_label"],
                    "yes_price": b["current_yes_price"] or 0,
                    "id": b["id"],
                    "clob_token_id_yes": b["clob_token_id_yes"],
                }
                for b in brackets
            ]

            # Find bracket matching current peak
            matching = find_bracket_for_temp(max_temp, bracket_list)
            if not matching:
                continue

            price = matching.get("yes_price", 0) or 0
            if price >= 0.97 or price <= 0:
                continue  # already settled or no liquidity

            # Edge = confidence - market price
            edge = confidence - price
            if edge < 0.03:
                continue  # no meaningful edge

            # Check no duplicate bet
            existing = self._db.conn.execute(
                "SELECT id FROM bets WHERE event_id = ? AND date = ?",
                (event["event_id"], today),
            ).fetchone()
            if existing:
                continue

            # Position sizing
            size = round(bet_size / price, 2) if price > 0 else 0
            cost = round(price * size, 2)
            remaining = max_daily - daily_spent
            if cost > remaining:
                cost = remaining
                size = round(cost / price, 2)
            if cost < 1.0:
                continue

            # Place bet
            token_id = matching.get("clob_token_id_yes", "")
            if not token_id:
                continue

            order_result = self._trader.place_limit_buy(
                token_id=token_id, price=price, size=size,
            )

            self._market_repo.record_bet(
                event_id=event["event_id"],
                bracket_id=matching["id"],
                city_key=city.key,
                date=today,
                bracket_label=matching["label"],
                side="BUY",
                token_id=token_id,
                price=price,
                size=size,
                cost=cost,
                current_max_temp=max_temp,
                peak_hour=peak_hour,
                model_confidence=confidence,
                expected_value=edge,
                order_id=order_result.order_id,
                order_status=order_result.status,
                is_paper=order_result.is_paper,
            )

            daily_spent += cost
            log.info("AUTO_BET", city=city.key, bracket=matching["label"],
                     conf=f"{confidence*100:.0f}%", price=f"{price*100:.0f}¢",
                     edge=f"+{edge*100:.0f}%", cost=f"${cost:.2f}",
                     paper=order_result.is_paper)

    @staticmethod
    def _is_wu_scrape_time() -> bool:
        """Check if current minute is :05 or :35."""
        import datetime
        minute = datetime.datetime.now().minute
        return minute in (5, 35)

    async def run_loop(self, market_interval_minutes: int = 5) -> None:
        """Run the orchestrator in a continuous loop."""
        log.info("orchestrator_started",
                 market_interval=market_interval_minutes,
                 weather_schedule="every :05 and :35",
                 paper_mode=self._settings.paper_mode)

        last_market_poll = 0
        last_weather_minute = -1

        while True:
            try:
                now = time.time()
                import datetime
                current_minute = datetime.datetime.now().minute

                if now - last_market_poll >= market_interval_minutes * 60:
                    await self._market_monitor.poll_all()
                    last_market_poll = now

                if self._is_wu_scrape_time() and current_minute != last_weather_minute:
                    last_weather_minute = current_minute
                    log.info("wu_scrape_triggered", minute=current_minute)
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

                await asyncio.sleep(30)

            except KeyboardInterrupt:
                log.info("orchestrator_stopped")
                break
            except Exception as exc:
                log.error("orchestrator_error", error=str(exc))
                await asyncio.sleep(60)

    def shutdown(self) -> None:
        self._db.close()
        log.info("orchestrator_shutdown")
