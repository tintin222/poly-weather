"""Main orchestrator: detects peak-passed conditions and places bets.

Strategy: when the daily peak temperature has passed with high confidence,
buy the matching bracket at a discount.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.db.database import Database
from src.db.repositories import JobRunRepository, MarketRepository, WeatherRepository
from src.model.bracket_analyzer import find_bracket_for_temp
from src.model.features import compute_current_max
from src.model.ml_trainer import MLTrainer
from src.model.peak_predictor import PeakPredictor
from src.polymarket.market_monitor import MarketMonitor
from src.polymarket.trader import PolymarketTrader
from src.strategy.bot_status import bot_status
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
        self._job_repo = JobRunRepository(self._db)

        self._market_monitor = MarketMonitor(self._market_repo)
        self._trader = PolymarketTrader(settings)
        self._peak_predictor = PeakPredictor(self._db)
        self._ml_trainer = MLTrainer(self._db)

        # Load paper account settings
        self._bet_size = settings.bet_size
        self._max_daily_exposure = settings.max_daily_exposure
        self._min_edge = settings.min_edge

        # Forecast high cache: {city_key: (date_str, forecast_high)}
        self._forecast_cache: dict[str, tuple[str, float]] = {}

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
        cycle_run_id = self._job_repo.create_run("orchestrator_cycle")
        try:
            poll_run_id = self._job_repo.create_run("market_poll")
            try:
                count = await self._market_monitor.poll_all()
                self._job_repo.finish_run(poll_run_id, "success", records_processed=count)
            except Exception as exc:
                self._job_repo.finish_run(poll_run_id, "error", error_message=str(exc))
                raise

            cities_today = self._get_todays_cities()
            log.info("cities_to_evaluate", count=len(cities_today))

            scraper = WUScraper(headless=True)
            await scraper.start()
            try:
                for city in cities_today:
                    scrape_run_id = self._job_repo.create_run("scrape", city.key)
                    try:
                        await self._scrape_and_evaluate(city, scraper)
                        self._job_repo.finish_run(scrape_run_id, "success")
                    except Exception as exc:
                        self._job_repo.finish_run(scrape_run_id, "error", error_message=str(exc))
                        log.error("city_error", city=city.key, error=str(exc))
            finally:
                await scraper.stop()

            self._job_repo.finish_run(cycle_run_id, "success", records_processed=len(cities_today))
        except Exception as exc:
            self._job_repo.finish_run(cycle_run_id, "error", error_message=str(exc))
            raise

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

    async def _evaluate_cities_only(self) -> None:
        """Evaluate all today's cities using existing weather data (no scrape).

        Called on every market poll cycle so we react to price changes
        without waiting for the next scrape window.
        """
        cities_today = self._get_todays_cities()
        if not cities_today:
            return

        bot_status.last_action = f"Evaluating {len(cities_today)} cities"
        bot_status.log_event("evaluate", f"Evaluating {len(cities_today)} cities (prices updated)")

        for city in cities_today:
            try:
                await self._evaluate_city(city)
                bot_status.cities_evaluated += 1
            except Exception as exc:
                log.error("evaluate_only_error", city=city.key, error=str(exc))
                bot_status.errors_recent += 1

    async def _get_forecast_high(self, city: CityConfig, today: str) -> float | None:
        """Return today's forecast high from Open-Meteo, cached per city per day."""
        cached = self._forecast_cache.get(city.key)
        if cached and cached[0] == today:
            return cached[1]
        try:
            from src.weather import open_meteo
            data = await open_meteo.fetch(city)
            today_temps = [h["temperature"] for h in data["hourly"] if h["date"] == today]
            if today_temps:
                high = max(today_temps)
                self._forecast_cache[city.key] = (today, high)
                log.info("forecast_high_cached", city=city.key, forecast_high=high)
                return high
        except Exception as exc:
            log.warning("forecast_high_error", city=city.key, error=str(exc))
        return None

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

        # FIX 1: Time gate — never bet before 11am local time.
        # Peak temps occur between 12-16h for almost all cities; early morning
        # observations are nighttime lows and will produce the wrong bracket.
        if current_hour < 11:
            log.debug("time_gate_skip", city=city.key, hour=current_hour)
            return

        # Need observations
        observations = self._weather_repo.get_hourly_observations(city.key, today)
        if not observations or len(observations) < 8:
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

        # --- ML classifier signal: P(current_hour is the peak hour) ---
        import datetime as _dt
        _today_dt = _dt.date.today()
        _doy = _today_dt.timetuple().tm_yday
        # 7-day rolling peak hour from DB
        _past = self._db.conn.execute(
            """SELECT peak_hour FROM daily_peaks WHERE city_key = ? AND peak_hour > 0
               ORDER BY date DESC LIMIT 7""",
            (city.key,),
        ).fetchall()
        _rolling = float(sum(r["peak_hour"] for r in _past) / len(_past)) if _past else 14.0
        _obs_dicts = [dict(o) if not isinstance(o, dict) else o for o in observations]
        ml_prob = self._ml_trainer.predict_is_peak(
            city_key=city.key,
            current_hour=current_hour,
            today_obs=_obs_dicts,
            rolling_peak_hour=_rolling,
            day_of_year=_doy,
        )

        # Combined confidence: blend CDF (60%) and ML peak probability (40%)
        # ML probability baseline is 1/15 ≈ 6.7%; rescale to 0-1 relative to max ~0.95
        if ml_prob is not None:
            # Rescale ML probability: 0.067 (random) → 0, 0.95 (strong) → 1
            ml_scaled = min(max((ml_prob - 0.067) / (0.95 - 0.067), 0.0), 1.0)
            combined_conf = 0.6 * confidence + 0.4 * ml_scaled
            log.info("peak_eval", city=city.key, hour=current_hour,
                     peak=max_temp, cdf_conf=f"{confidence*100:.0f}%",
                     ml_prob=f"{ml_prob*100:.1f}%", combined=f"{combined_conf*100:.0f}%")
            confidence = combined_conf
        else:
            log.info("peak_eval", city=city.key, hour=current_hour,
                     peak=max_temp, conf=f"{confidence*100:.0f}%", ml="no_model")

        # Only bet when confidence is HIGH
        if confidence < min_conf:
            return

        # FIX 2: Forecast gate — if Open-Meteo expects the temp to climb
        # significantly more today, the peak hasn't arrived yet.
        forecast_high = await self._get_forecast_high(city, today)
        if forecast_high is not None:
            gap = forecast_high - max_temp
            if gap > 3:
                log.info("forecast_gate_skip", city=city.key,
                         current_max=max_temp, forecast_high=forecast_high, gap=round(gap, 1))
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
            # FIX 3: Price floor — skip brackets priced below $0.01.
            # Near-zero prices mean the market considers the outcome essentially
            # impossible; our model isn't reliable enough to fight that signal,
            # and the huge share sizes (bet_size / ~0) create phantom positions.
            if price >= 0.97 or price < 0.01:
                continue

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

            bot_status.bets_placed += 1
            bot_status.last_bet_at = time.time()
            bot_status.log_event(
                "AUTO_BET",
                f"{city.key}: {matching['label']} @ {price*100:.0f}¢, "
                f"conf={confidence*100:.0f}%, edge=+{edge*100:.0f}%, "
                f"cost=${cost:.2f}, paper={order_result.is_paper}",
            )

            log.info("AUTO_BET", city=city.key, bracket=matching["label"],
                     conf=f"{confidence*100:.0f}%", price=f"{price*100:.0f}¢",
                     edge=f"+{edge*100:.0f}%", cost=f"${cost:.2f}",
                     paper=order_result.is_paper)

    async def run_loop(self, market_interval_minutes: int = 5, weather_interval_minutes: int = 5) -> None:
        """Run the orchestrator in a continuous loop.

        - Market prices polled every ``market_interval_minutes``
        - After each market poll, re-evaluate all cities with existing weather data
        - Weather scrape every ``weather_interval_minutes``
        """
        bot_status.running = True
        bot_status.started_at = time.time()
        bot_status.last_action = "Starting up"
        bot_status.log_event("start", f"Bot started, paper={self._settings.paper_mode}")

        log.info("orchestrator_started",
                 market_interval=market_interval_minutes,
                 weather_interval=weather_interval_minutes,
                 paper_mode=self._settings.paper_mode)

        last_market_poll = 0
        last_weather_scrape = 0

        while True:
            try:
                now = time.time()

                # Poll market prices and evaluate on every interval
                if now - last_market_poll >= market_interval_minutes * 60:
                    poll_run_id = self._job_repo.create_run("market_poll")
                    try:
                        count = await self._market_monitor.poll_all()
                        self._job_repo.finish_run(poll_run_id, "success", records_processed=count)
                        bot_status.last_action = f"Polled {count} markets"
                        bot_status.log_event("market_poll", f"Polled {count} markets")
                    except Exception as exc:
                        self._job_repo.finish_run(poll_run_id, "error", error_message=str(exc))
                        bot_status.errors_recent += 1
                        bot_status.log_event("error", f"Market poll failed: {exc}")
                        raise
                    last_market_poll = now

                    # Re-evaluate cities with existing data after each price update
                    await self._evaluate_cities_only()
                    bot_status.cycle_count += 1
                    bot_status.last_cycle_at = time.time()

                # Full scrape + evaluate every weather_interval_minutes
                if now - last_weather_scrape >= weather_interval_minutes * 60:
                    last_weather_scrape = now
                    log.info("wu_scrape_triggered", interval_minutes=weather_interval_minutes)
                    bot_status.last_action = "Scraping weather data"
                    bot_status.log_event("scrape", "Weather scrape triggered")

                    cities_today = self._get_todays_cities()
                    if cities_today:
                        scraper = WUScraper(headless=True)
                        await scraper.start()
                        try:
                            for city in cities_today:
                                scrape_run_id = self._job_repo.create_run("scrape", city.key)
                                try:
                                    await self._scrape_and_evaluate(city, scraper)
                                    self._job_repo.finish_run(scrape_run_id, "success")
                                except Exception as exc:
                                    self._job_repo.finish_run(scrape_run_id, "error", error_message=str(exc))
                                    log.error("evaluation_error", city=city.key, error=str(exc))
                                    bot_status.errors_recent += 1
                                    bot_status.log_event("error", f"Scrape failed for {city.key}: {exc}")
                        finally:
                            await scraper.stop()

                await asyncio.sleep(30)

            except KeyboardInterrupt:
                log.info("orchestrator_stopped")
                bot_status.running = False
                bot_status.log_event("stop", "Stopped by user")
                break
            except Exception as exc:
                log.error("orchestrator_error", error=str(exc))
                bot_status.errors_recent += 1
                bot_status.log_event("error", f"Loop error: {exc}")
                await asyncio.sleep(60)

    def shutdown(self) -> None:
        self._db.close()
        bot_status.running = False
        bot_status.log_event("shutdown", "Orchestrator shutdown")
        log.info("orchestrator_shutdown")
