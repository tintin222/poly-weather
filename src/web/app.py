"""FastAPI dashboard for poly-weather paper trading results."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.db.database import Database
from src.db.repositories import JobRunRepository, MarketRepository, ModelResultRepository, WeatherRepository
from src.model.ml_trainer import MLTrainer
from src.utils.logger import get_logger

log = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="poly-weather", docs_url="/api/docs")

# Database singleton - initialized on startup
_db: Database | None = None

# In-memory job tracking for async scrape/backfill tasks
_jobs: dict[str, dict] = {}
# Track which cities already have a backfill running/completed this session
_backfill_started: set[str] = set()
# Separate semaphores so scrapes aren't blocked by backfills
_scrape_semaphore = asyncio.Semaphore(3)   # for today's scrapes (fast, ~5s each)
_backfill_semaphore = asyncio.Semaphore(1)  # for 30-day backfills (slow, ~2min each)

# ML trainer singleton (caches trained models in-memory)
_ml_trainer: MLTrainer | None = None


def get_ml_trainer() -> MLTrainer:
    global _ml_trainer
    if _ml_trainer is None:
        _ml_trainer = MLTrainer(get_db())
    return _ml_trainer


def get_job_repo() -> JobRunRepository:
    return JobRunRepository(get_db())


def get_db() -> Database:
    global _db
    if _db is None:
        from config import settings
        _db = Database(settings.db_path)
        _db.initialize()
    return _db


def _auto_backfill_if_needed(city_key: str) -> str | None:
    """Start a 30-day backfill for a city if it has <5 days of data.

    Returns the job_id if started, None if not needed.
    """
    if city_key in _backfill_started:
        # Check if the job is still running
        running = any(
            j.get("city") == city_key and j.get("type") == "backfill" and j.get("status") == "running"
            for j in _jobs.values()
        )
        if running:
            return None
        # Job finished — check if we need more data
        db = get_db()
        hist = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
            (city_key,),
        ).fetchone()["cnt"]
        if hist >= 5:
            return None
        # Need more data — allow retry
        _backfill_started.discard(city_key)

    db = get_db()
    hist = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
        (city_key,),
    ).fetchone()["cnt"]

    if hist >= 5:
        _backfill_started.add(city_key)
        return None

    from src.weather.cities import get_city
    city = get_city(city_key)
    if not city:
        return None

    _backfill_started.add(city_key)
    job_id = f"auto-{city_key[:8].lower()}-{uuid.uuid4().hex[:4]}"
    _jobs[job_id] = {
        "status": "running", "type": "backfill", "city": city_key,
        "days": 30, "started": time.time(), "auto": True,
    }
    log.info("auto_backfill_started", city=city_key, job_id=job_id)
    asyncio.create_task(_run_backfill_job(job_id, city, 30))
    return job_id


# Track which cities have been auto-scraped today
_scrape_started: set[str] = set()
# Cache forecast highs per city (scraped once per session)
_forecasts: dict[str, float] = {}
_forecast_started: set[str] = set()


def _auto_scrape_if_needed(city_key: str, city, event_date: str) -> str | None:
    """Auto-scrape today's WU data for a city if not already done."""
    from src.utils.time_utils import today_in_tz

    today = today_in_tz(city.timezone)
    if event_date != today:
        return None

    scrape_key = f"{city_key}:{today}"
    if scrape_key in _scrape_started:
        return None

    db = get_db()
    existing = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM hourly_observations WHERE city_key = ? AND date = ?",
        (city_key, today),
    ).fetchone()["cnt"]

    if existing > 0:
        _scrape_started.add(scrape_key)
        return None

    _scrape_started.add(scrape_key)
    job_id = f"auto-scrape-{city_key[:8].lower()}-{uuid.uuid4().hex[:4]}"
    _jobs[job_id] = {
        "status": "running", "type": "scrape", "city": city_key,
        "started": time.time(), "auto": True,
    }
    log.info("auto_scrape_started", city=city_key, date=today, job_id=job_id)
    asyncio.create_task(_run_scrape_job(job_id, city, today))
    return job_id


def _auto_forecast_if_needed(city_key: str, city) -> None:
    """Auto-scrape today's WU forecast high for a city if not done yet."""
    if city_key in _forecast_started:
        return
    _forecast_started.add(city_key)
    asyncio.create_task(_run_forecast_job(city_key, city))


async def _run_forecast_job(city_key: str, city) -> None:
    """Background task: scrape WU forecast high."""
    from src.weather.scraper import WUScraper

    async with _scrape_semaphore:
        try:
            scraper = WUScraper(headless=True)
            await scraper.start()
            try:
                forecast_high = await scraper.scrape_forecast_high(city)
            finally:
                await scraper.stop()

            if forecast_high is not None:
                _forecasts[city_key] = forecast_high
                log.info("forecast_cached", city=city_key, forecast_high=forecast_high)
        except Exception as exc:
            log.warning("forecast_job_failed", city=city_key, error=str(exc))


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    db = get_db()
    conn = db.conn

    # Summary stats
    total_bets = conn.execute("SELECT COUNT(*) as cnt FROM bets").fetchone()["cnt"]
    paper_bets = conn.execute(
        "SELECT COUNT(*) as cnt FROM bets WHERE is_paper = 1"
    ).fetchone()["cnt"]
    won_bets = conn.execute(
        "SELECT COUNT(*) as cnt FROM bets WHERE won = 1"
    ).fetchone()["cnt"]
    lost_bets = conn.execute(
        "SELECT COUNT(*) as cnt FROM bets WHERE won = 0"
    ).fetchone()["cnt"]
    pending_bets = conn.execute(
        "SELECT COUNT(*) as cnt FROM bets WHERE won IS NULL"
    ).fetchone()["cnt"]
    total_pnl = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as total FROM bets WHERE pnl IS NOT NULL"
    ).fetchone()["total"]
    total_cost = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) as total FROM bets"
    ).fetchone()["total"]

    # Recent bets
    recent_bets = conn.execute("""
        SELECT b.*, e.title as event_title
        FROM bets b
        LEFT JOIN polymarket_events e ON b.event_id = e.event_id
        ORDER BY b.created_at DESC
        LIMIT 50
    """).fetchall()

    # Active events today
    from src.utils.time_utils import today_in_tz
    today = today_in_tz("UTC")
    active_events = conn.execute("""
        SELECT e.*, COUNT(br.id) as bracket_count
        FROM polymarket_events e
        LEFT JOIN polymarket_brackets br ON e.event_id = br.event_id
        WHERE e.status = 'active' AND e.date >= ?
        GROUP BY e.event_id
        ORDER BY e.date, e.city_key
        LIMIT 50
    """, (today,)).fetchall()

    # Daily P&L summary
    daily_pnl = conn.execute("""
        SELECT date,
               COUNT(*) as bet_count,
               SUM(cost) as total_cost,
               SUM(CASE WHEN won = 1 THEN (size - cost) ELSE 0 END) as profit,
               SUM(CASE WHEN won = 0 THEN -cost ELSE 0 END) as loss,
               COALESCE(SUM(pnl), 0) as net_pnl
        FROM bets
        GROUP BY date
        ORDER BY date DESC
        LIMIT 30
    """).fetchall()

    # Weather data stats
    observation_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM hourly_observations"
    ).fetchone()["cnt"]
    peak_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM daily_peaks"
    ).fetchone()["cnt"]
    cities_with_data = conn.execute(
        "SELECT COUNT(DISTINCT city_key) as cnt FROM hourly_observations"
    ).fetchone()["cnt"]

    # Latest observations per city
    latest_peaks = conn.execute("""
        SELECT dp.city_key, dp.date, dp.peak_temp, dp.peak_hour, dp.observation_count
        FROM daily_peaks dp
        INNER JOIN (
            SELECT city_key, MAX(date) as max_date
            FROM daily_peaks
            GROUP BY city_key
        ) latest ON dp.city_key = latest.city_key AND dp.date = latest.max_date
        ORDER BY dp.city_key
    """).fetchall()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_bets": total_bets,
        "paper_bets": paper_bets,
        "won_bets": won_bets,
        "lost_bets": lost_bets,
        "pending_bets": pending_bets,
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "recent_bets": recent_bets,
        "active_events": active_events,
        "daily_pnl": daily_pnl,
        "observation_count": observation_count,
        "peak_count": peak_count,
        "cities_with_data": cities_with_data,
        "latest_peaks": latest_peaks,
        "now": int(time.time()),
    })


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status():
    db = get_db()
    conn = db.conn
    total = conn.execute("SELECT COUNT(*) as cnt FROM bets").fetchone()["cnt"]
    pending = conn.execute(
        "SELECT COUNT(*) as cnt FROM bets WHERE won IS NULL"
    ).fetchone()["cnt"]
    obs = conn.execute(
        "SELECT COUNT(*) as cnt FROM hourly_observations"
    ).fetchone()["cnt"]
    return {"total_bets": total, "pending_bets": pending, "observations": obs, "status": "running"}


@app.get("/api/bot-status")
async def api_bot_status():
    """Return the current auto-betting bot status."""
    from src.strategy.bot_status import bot_status
    return bot_status.to_dict()


@app.get("/api/bets")
async def api_bets(limit: int = 50):
    db = get_db()
    rows = db.conn.execute(
        "SELECT * FROM bets ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/events")
async def api_events(status: str = "active", limit: int = 50):
    db = get_db()
    rows = db.conn.execute(
        "SELECT * FROM polymarket_events WHERE status = ? ORDER BY date DESC LIMIT ?",
        (status, limit),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/brackets/{event_id}")
async def api_brackets(event_id: str):
    db = get_db()
    rows = db.conn.execute(
        "SELECT * FROM polymarket_brackets WHERE event_id = ? ORDER BY bracket_label",
        (event_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/peaks/{city_key}")
async def api_peaks(city_key: str, limit: int = 30):
    db = get_db()
    rows = db.conn.execute(
        "SELECT * FROM daily_peaks WHERE city_key = ? ORDER BY date DESC LIMIT ?",
        (city_key, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Opportunities page
# ---------------------------------------------------------------------------

@app.get("/opportunities", response_class=HTMLResponse)
async def opportunities_page(request: Request):
    return templates.TemplateResponse("opportunities.html", {"request": request})


@app.get("/api/opportunities")
async def api_opportunities():
    """Build peak-confidence based guidance for each active market."""
    import time as _time
    from config import settings
    from src.model.bracket_analyzer import generate_peak_guidance
    from src.model.peak_predictor import PeakPredictor
    from src.utils.time_utils import current_hour_in_tz, today_in_tz
    from src.weather.cities import get_city

    db = get_db()
    weather_repo = WeatherRepository(db)
    market_repo = MarketRepository(db)
    predictor = PeakPredictor(db)

    active_events = market_repo.get_active_events()
    opportunities = []

    for event in active_events:
        city_key = event["city_key"]
        event_date = event["date"]
        city = get_city(city_key)
        if not city:
            continue

        from src.utils.time_utils import now_in_tz
        city_now = now_in_tz(city.timezone)
        today = city_now.strftime("%Y-%m-%d")
        current_hour = city_now.hour
        current_minute = city_now.minute
        is_today = event_date == today

        # Auto-actions
        _auto_backfill_if_needed(city_key)
        _auto_scrape_if_needed(city_key, city, event_date)

        # Get brackets
        brackets = market_repo.get_brackets_for_event(event["event_id"])
        bracket_list = [
            {
                "id": b["id"],
                "label": b["bracket_label"],
                "yes_price": b["current_yes_price"] or 0,
                "token_id_yes": b["clob_token_id_yes"],
            }
            for b in brackets
        ]

        # Get observations
        observations = weather_repo.get_hourly_observations(city_key, event_date)

        # WU scrape timestamp
        wu_scraped_at = None
        if observations:
            row = db.conn.execute(
                "SELECT MAX(scraped_at) as ts FROM hourly_observations WHERE city_key = ? AND date = ?",
                (city_key, event_date),
            ).fetchone()
            wu_scraped_at = row["ts"] if row else None

        # Peak confidence from PeakPredictor
        peak_conf = None
        peak_detail = None
        mean_peak_hour = None
        hist_count = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
            (city_key,),
        ).fetchone()["cnt"]

        if observations and len(observations) >= 2 and hist_count >= 5:
            try:
                from src.model.features import compute_current_max
                max_temp, _ = compute_current_max(observations)
                dist = predictor.train(city_key, event_date, window_days=15, years_back=1)
                mean_peak_hour = dist.mean_peak_hour
                prediction = predictor.predict_confidence(
                    city_key, current_hour, max_temp, observations
                )
                peak_conf = prediction.confidence
                peak_detail = {
                    "base_confidence": prediction.base_confidence,
                    "mean_peak_hour": prediction.mean_peak_hour,
                    "hours_past_mean": prediction.hours_past_mean,
                }
            except Exception as exc:
                log.warning("confidence_error", city=city_key, error=str(exc))
        elif hist_count >= 5:
            # Even without today's observations, get mean peak hour for sorting
            try:
                dist = predictor.train(city_key, event_date, window_days=15, years_back=1)
                mean_peak_hour = dist.mean_peak_hour
            except Exception:
                pass

        # Generate peak-confidence based guidance
        guidance = generate_peak_guidance(
            city_key=city_key,
            date=event_date,
            brackets=bracket_list,
            observations=observations if observations else None,
            peak_confidence=peak_conf,
            peak_confidence_detail=peak_detail,
            current_hour=current_hour if is_today else None,
            unit=city.unit if city else "celsius",
            wu_scraped_at=wu_scraped_at,
            market_polled_at=event["updated_at"],
            min_confidence=settings.min_confidence,
        )

        slug = event["slug"]
        event_status = event["status"]

        # Peak proximity: how far (in hours) is the current time from the
        # historical mean peak hour.  Negative = before peak, positive = after.
        hours_to_peak = None  # signed distance
        peak_proximity = None  # absolute distance
        peak_phase = "UNKNOWN"

        if is_today and mean_peak_hour is not None:
            hours_to_peak = round(current_hour - mean_peak_hour, 1)
            peak_proximity = round(abs(hours_to_peak), 1)
            if peak_proximity <= 1:
                peak_phase = "AT_PEAK"
            elif peak_proximity <= 3:
                peak_phase = "NEAR_PEAK"
            elif hours_to_peak < 0:
                peak_phase = "APPROACHING"
            else:
                peak_phase = "COOLING"
        elif not is_today:
            peak_phase = "FUTURE"

        opportunities.append({
            "event_id": event["event_id"],
            "city_key": city_key,
            "city_name": city.display_name if city else city_key,
            "date": event_date,
            "slug": slug,
            "polymarket_url": f"https://polymarket.com/event/{slug}" if slug else None,
            "status": event_status,
            "is_today": is_today,
            "is_closed": event_status == "closed",
            "current_hour": current_hour if is_today else None,
            "current_time": f"{current_hour}:{current_minute:02d}" if is_today else None,
            "unit": city.unit if city else "celsius",
            "hist_days": hist_count,
            "mean_peak_hour": round(mean_peak_hour, 1) if mean_peak_hour is not None else None,
            "hours_to_peak": hours_to_peak,
            "peak_proximity": peak_proximity,
            "peak_phase": peak_phase,
            "g": {
                "action": guidance.action,
                "reason": guidance.reason,
                "peak": guidance.current_peak,
                "peak_hour": guidance.peak_hour,
                "now_temp": guidance.current_temp,
                "now_hour": guidance.latest_hour,
                "trend": guidance.trend,
                "hours_since": guidance.hours_since_peak,
                "obs_count": guidance.observation_count,
                "conf": round(guidance.confidence, 4),
                "conf_reason": guidance.confidence_reason,
                "match_bracket": guidance.matching_bracket,
                "match_price": guidance.matching_price,
                "match_id": guidance.matching_bracket_id,
                "match_token": guidance.matching_token_id,
                "edge": round(guidance.edge, 4),
                "gain_pct": guidance.potential_gain_pct,
                "brackets": guidance.brackets,
                "wu_at": guidance.wu_scraped_at,
                "mkt_at": guidance.market_polled_at,
            },
        })

    # Sort: today's opportunities by peak proximity (closest to peak first),
    # then future events by date.  Within equal proximity, sort by city name.
    def _sort_key(o):
        if o["is_today"]:
            # Primary: today first (0), Secondary: peak proximity ascending
            prox = o["peak_proximity"] if o["peak_proximity"] is not None else 99
            return (0, prox, o["city_key"])
        else:
            # Future events after today's, sorted by date
            return (1, 0, o["date"], o["city_key"])

    opportunities.sort(key=_sort_key)
    return opportunities


# ---------------------------------------------------------------------------
# Scrape / backfill / refresh actions
# ---------------------------------------------------------------------------

@app.post("/api/resolve-bets")
async def api_resolve_bets():
    """Manually trigger bet resolution for all closed markets."""
    db = get_db()
    market_repo = MarketRepository(db)
    from src.polymarket.market_monitor import MarketMonitor
    monitor = MarketMonitor(market_repo)
    resolved = monitor._resolve_settled_bets()
    return {"status": "ok", "resolved": resolved}


@app.post("/api/refresh-markets")
async def api_refresh_markets():
    """Re-poll Polymarket for latest events and prices."""
    db = get_db()
    market_repo = MarketRepository(db)
    job_repo = JobRunRepository(db)
    run_id = job_repo.create_run("market_poll")

    from src.polymarket.market_monitor import MarketMonitor
    monitor = MarketMonitor(market_repo)
    try:
        count = await monitor.poll_all()
        job_repo.finish_run(run_id, "success", records_processed=count)
    except Exception as exc:
        job_repo.finish_run(run_id, "error", error_message=str(exc))
        raise
    return {"status": "ok", "events_processed": count}


@app.post("/api/scrape/{city_key}")
async def api_scrape(city_key: str, date: str | None = None):
    """Trigger a WU scrape for a city. Returns a job_id to poll."""
    from src.weather.cities import get_city
    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "type": "scrape", "city": city_key, "started": time.time()}

    asyncio.create_task(_run_scrape_job(job_id, city, date))
    return {"job_id": job_id, "status": "running"}


async def _run_scrape_job(job_id: str, city, date: str | None):
    """Background task: scrape WU and store results. Uses semaphore for 1-at-a-time."""
    from src.utils.time_utils import today_in_tz
    from src.weather.scraper import WUScraper

    async with _scrape_semaphore:
        log.info("scrape_job_acquired_lock", job_id=job_id, city=city.key)
        db = get_db()
        repo = WeatherRepository(db)
        job_repo = JobRunRepository(db)
        scrape_date = date or today_in_tz(city.timezone)
        run_id = job_repo.create_run("scrape", city.key)

        try:
            scraper = WUScraper(headless=True)
            await scraper.start()
            try:
                observations = await scraper.scrape_daily(city, scrape_date)
            finally:
                await scraper.stop()

            if observations:
                obs_dicts = [obs.to_dict() for obs in observations]
                count = repo.upsert_hourly_observations_batch(city.key, scrape_date, obs_dicts)

                peak_obs = max(observations, key=lambda o: o.temperature)
                repo.upsert_daily_peak(
                    city_key=city.key, date=scrape_date,
                    peak_temp=peak_obs.temperature, peak_hour=peak_obs.hour,
                    observation_count=len(observations),
                )
                job_repo.finish_run(run_id, "success", records_processed=count)
                _jobs[job_id] = {
                    "status": "complete", "type": "scrape", "city": city.key,
                    "date": scrape_date, "observations": count,
                    "peak_temp": peak_obs.temperature, "peak_hour": peak_obs.hour,
                }
            else:
                job_repo.finish_run(run_id, "success", records_processed=0)
                _jobs[job_id] = {"status": "complete", "type": "scrape", "city": city.key, "observations": 0}

        except Exception as exc:
            log.error("scrape_job_failed", job_id=job_id, error=str(exc))
            job_repo.finish_run(run_id, "error", error_message=str(exc))
            _jobs[job_id] = {"status": "error", "type": "scrape", "error": str(exc)}


@app.post("/api/purge-rescrape/{city_key}")
async def api_purge_rescrape(city_key: str, days: int = 30):
    """Purge ALL observations + daily peaks for a city, then re-backfill.

    Use this after deploying the Playwright timezone fix to replace
    data that was scraped with wrong timezone (UTC instead of local).
    """
    from src.weather.cities import get_city
    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}

    db = get_db()

    # Count existing data
    obs_count = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM hourly_observations WHERE city_key = ?",
        (city_key,),
    ).fetchone()["cnt"]
    peak_count = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
        (city_key,),
    ).fetchone()["cnt"]

    # Purge
    db.conn.execute("DELETE FROM hourly_observations WHERE city_key = ?", (city_key,))
    db.conn.execute("DELETE FROM daily_peaks WHERE city_key = ?", (city_key,))
    db.conn.commit()
    log.info("purge_complete", city=city_key, obs_deleted=obs_count, peaks_deleted=peak_count)

    # Reset backfill tracking so auto-backfill can retrigger
    _backfill_started.discard(city_key)
    _scrape_started.discard(f"{city_key}:{__import__('src.utils.time_utils', fromlist=['today_in_tz']).today_in_tz(city.timezone)}")

    # Start a fresh backfill
    job_id = f"purge-{city_key[:8].lower()}-{uuid.uuid4().hex[:4]}"
    _jobs[job_id] = {
        "status": "running", "type": "purge_rescrape", "city": city_key,
        "days": days, "started": time.time(),
        "purged_obs": obs_count, "purged_peaks": peak_count,
    }
    asyncio.create_task(_run_backfill_job(job_id, city, days))
    return {
        "job_id": job_id, "status": "running",
        "purged_obs": obs_count, "purged_peaks": peak_count,
        "backfill_days": days,
    }


@app.post("/api/purge-rescrape-all")
async def api_purge_rescrape_all(days: int = 365):
    """Purge and re-scrape ALL cities. Use after timezone fix deployment."""
    from src.weather.cities import CITIES

    db = get_db()
    results = {}
    for city_key in CITIES:
        obs = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM hourly_observations WHERE city_key = ?",
            (city_key,),
        ).fetchone()["cnt"]
        peaks = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
            (city_key,),
        ).fetchone()["cnt"]
        db.conn.execute("DELETE FROM hourly_observations WHERE city_key = ?", (city_key,))
        db.conn.execute("DELETE FROM daily_peaks WHERE city_key = ?", (city_key,))
        _backfill_started.discard(city_key)
        results[city_key] = {"purged_obs": obs, "purged_peaks": peaks}
    db.conn.commit()

    # Start backfill-all job
    job_id = f"purge-all-{uuid.uuid4().hex[:4]}"
    city_keys = list(CITIES.keys())
    _jobs[job_id] = {
        "status": "running", "type": "purge_rescrape_all",
        "total_cities": len(city_keys), "done_cities": 0,
        "current_city": None, "days": days,
        "started": time.time(),
    }
    asyncio.create_task(_run_backfill_all_job(job_id, city_keys, days))
    return {"job_id": job_id, "status": "running", "cities": results, "backfill_days": days}


@app.post("/api/backfill/{city_key}")
async def api_backfill(city_key: str, days: int = 30):
    """Trigger a historical backfill for a city. Returns a job_id to poll."""
    from src.weather.cities import get_city
    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "running", "type": "backfill", "city": city_key,
        "days": days, "started": time.time(),
    }

    asyncio.create_task(_run_backfill_job(job_id, city, days))
    return {"job_id": job_id, "status": "running", "days": days}


async def _run_backfill_job(job_id: str, city, days: int):
    """Background task: backfill historical data. Uses its own semaphore."""
    from src.weather.historical_loader import HistoricalLoader
    from src.weather.scraper import WUScraper

    async with _backfill_semaphore:
        log.info("backfill_job_acquired_lock", job_id=job_id, city=city.key)
        db = get_db()
        repo = WeatherRepository(db)
        job_repo = JobRunRepository(db)
        run_id = job_repo.create_run("backfill", city.key, metadata={"days": days})

        try:
            scraper = WUScraper(headless=True)
            await scraper.start()
            try:
                loader = HistoricalLoader(scraper, repo)
                await loader.backfill(city, days=days)
            finally:
                await scraper.stop()

            total = db.conn.execute(
                "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
                (city.key,),
            ).fetchone()["cnt"]

            job_repo.finish_run(run_id, "success", records_processed=total)
            _jobs[job_id] = {
                "status": "complete", "type": "backfill", "city": city.key,
                "days": days, "total_peaks": total,
            }
        except Exception as exc:
            log.error("backfill_job_failed", job_id=job_id, error=str(exc))
            job_repo.finish_run(run_id, "error", error_message=str(exc))
            _jobs[job_id] = {"status": "error", "type": "backfill", "error": str(exc)}


@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str):
    """Check status of a scrape/backfill job."""
    if job_id not in _jobs:
        return {"error": "Unknown job", "job_id": job_id}
    return {"job_id": job_id, **_jobs[job_id]}


@app.get("/api/city/{city_key}/analysis")
async def api_city_analysis(city_key: str):
    """Return peak hour distribution and CDF for a city."""
    from src.model.peak_predictor import PeakPredictor
    from src.utils.time_utils import today_in_tz
    from src.weather.cities import get_city

    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}

    db = get_db()
    today = today_in_tz(city.timezone)

    predictor = PeakPredictor(db)
    try:
        dist = predictor.train(city_key, today, window_days=15, years_back=1)
    except Exception:
        return {"city_key": city_key, "has_model": False, "cdf": []}

    cdf_table = [{"hour": h, "probability": round(dist.cdf(h), 4)} for h in range(24)]

    # Recent peaks
    peaks = db.conn.execute(
        "SELECT date, peak_temp, peak_hour FROM daily_peaks WHERE city_key = ? ORDER BY date DESC LIMIT 30",
        (city_key,),
    ).fetchall()

    return {
        "city_key": city_key,
        "has_model": True,
        "mean_peak_hour": round(dist.mean_peak_hour, 1),
        "std_peak_hour": round(dist.std_peak_hour, 1),
        "percentile_range": dist.percentile_range(10, 90),
        "cdf": cdf_table,
        "recent_peaks": [dict(p) for p in peaks],
    }


@app.post("/api/place-bet")
async def api_place_bet(
    event_id: str,
    bracket_id: int,
    token_id: str,
    price: float,
    size: float,
    city_key: str,
    date: str,
    bracket_label: str,
    current_max_temp: float,
    peak_hour: int,
    model_confidence: float,
    expected_value: float,
):
    """Manually place a paper bet from the opportunities page."""
    from config import settings
    from src.polymarket.trader import PolymarketTrader

    db = get_db()
    market_repo = MarketRepository(db)
    trader = PolymarketTrader(settings)

    result = trader.place_limit_buy(
        token_id=token_id, price=price, size=size,
    )

    bet_id = market_repo.record_bet(
        event_id=event_id, bracket_id=bracket_id, city_key=city_key,
        date=date, bracket_label=bracket_label, side="BUY",
        token_id=token_id, price=price, size=size,
        cost=round(price * size, 2),
        current_max_temp=current_max_temp, peak_hour=peak_hour,
        model_confidence=model_confidence, expected_value=expected_value,
        order_id=result.order_id, order_status=result.status,
        is_paper=result.is_paper,
    )

    return {
        "bet_id": bet_id,
        "order_id": result.order_id,
        "status": result.status,
        "success": result.success,
        "is_paper": result.is_paper,
    }


# ---------------------------------------------------------------------------
# Paper trading account & config
# ---------------------------------------------------------------------------

@app.get("/api/paper-account")
async def api_paper_account():
    """Get paper trading account summary: balance, P&L, bet history."""
    db = get_db()
    conn = db.conn

    # Get or create paper account config
    account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    if not account:
        from config import settings
        conn.execute(
            """INSERT INTO paper_account (id, initial_balance, bet_size, max_daily_exposure, min_edge, updated_at)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (settings.initial_balance, settings.bet_size, settings.max_daily_exposure, settings.min_edge, int(time.time())),
        )
        conn.commit()
        account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()

    initial_balance = account["initial_balance"]

    # Calculate current balance from bets
    total_cost = conn.execute("SELECT COALESCE(SUM(cost), 0) as v FROM bets").fetchone()["v"]
    total_winnings = conn.execute(
        "SELECT COALESCE(SUM(size), 0) as v FROM bets WHERE won = 1"
    ).fetchone()["v"]  # each winning share pays $1, so size = winnings
    total_lost = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) as v FROM bets WHERE won = 0"
    ).fetchone()["v"]
    pending_cost = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) as v FROM bets WHERE won IS NULL"
    ).fetchone()["v"]

    realized_pnl = total_winnings - (total_cost - pending_cost)
    current_balance = initial_balance - total_cost + total_winnings

    # Stats
    total_bets = conn.execute("SELECT COUNT(*) as v FROM bets").fetchone()["v"]
    won = conn.execute("SELECT COUNT(*) as v FROM bets WHERE won = 1").fetchone()["v"]
    lost = conn.execute("SELECT COUNT(*) as v FROM bets WHERE won = 0").fetchone()["v"]
    pending = conn.execute("SELECT COUNT(*) as v FROM bets WHERE won IS NULL").fetchone()["v"]

    # Recent bets
    recent = conn.execute("""
        SELECT b.*, e.title as event_title, e.status as event_status
        FROM bets b
        LEFT JOIN polymarket_events e ON b.event_id = e.event_id
        ORDER BY b.created_at DESC LIMIT 50
    """).fetchall()

    return {
        "config": {
            "initial_balance": initial_balance,
            "bet_size": account["bet_size"],
            "max_daily_exposure": account["max_daily_exposure"],
            "min_edge": account["min_edge"],
        },
        "balance": {
            "initial": initial_balance,
            "current": round(current_balance, 2),
            "total_invested": round(total_cost, 2),
            "total_winnings": round(total_winnings, 2),
            "realized_pnl": round(realized_pnl, 2),
            "pending_cost": round(pending_cost, 2),
            "pnl_pct": round((realized_pnl / initial_balance) * 100, 2) if initial_balance > 0 else 0,
        },
        "stats": {
            "total_bets": total_bets,
            "won": won,
            "lost": lost,
            "pending": pending,
            "win_rate": round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0,
        },
        "bets": [dict(r) for r in recent],
    }


@app.post("/api/paper-account/config")
async def api_update_paper_config(
    initial_balance: float | None = None,
    bet_size: float | None = None,
    max_daily_exposure: float | None = None,
    min_edge: float | None = None,
):
    """Update paper trading configuration."""
    db = get_db()
    conn = db.conn

    # Ensure row exists
    existing = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    if not existing:
        from config import settings
        conn.execute(
            """INSERT INTO paper_account (id, initial_balance, bet_size, max_daily_exposure, min_edge, updated_at)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (settings.initial_balance, settings.bet_size, settings.max_daily_exposure, settings.min_edge, int(time.time())),
        )
        conn.commit()
        existing = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()

    updates = []
    params = []
    if initial_balance is not None:
        updates.append("initial_balance = ?")
        params.append(initial_balance)
    if bet_size is not None:
        updates.append("bet_size = ?")
        params.append(bet_size)
    if max_daily_exposure is not None:
        updates.append("max_daily_exposure = ?")
        params.append(max_daily_exposure)
    if min_edge is not None:
        updates.append("min_edge = ?")
        params.append(min_edge)

    if updates:
        updates.append("updated_at = ?")
        params.append(int(time.time()))
        params.append(1)
        conn.execute(f"UPDATE paper_account SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    return {"status": "ok", "config": dict(conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone())}


@app.post("/api/paper-account/reset")
async def api_reset_paper_account():
    """Reset paper trading: delete all bets, keep config."""
    db = get_db()
    db.conn.execute("DELETE FROM bets")
    db.conn.commit()
    return {"status": "ok", "message": "All bets deleted. Balance reset."}


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Config page for paper trading settings."""
    return templates.TemplateResponse("config.html", {"request": request})


# ---------------------------------------------------------------------------
# City detail page
# ---------------------------------------------------------------------------

@app.get("/city/{city_key}", response_class=HTMLResponse)
async def city_detail_page(city_key: str, request: Request):
    from src.weather.cities import get_city
    city = get_city(city_key)
    return templates.TemplateResponse("city.html", {
        "request": request,
        "city_key": city_key,
        "city": city,
    })


@app.get("/api/city/{city_key}/hourly")
async def api_city_hourly(
    city_key: str,
    page: int = 1,
    page_size: int = 50,
    date_from: str | None = None,
    date_to: str | None = None,
):
    db = get_db()
    repo = WeatherRepository(db)
    rows, total = repo.get_hourly_paginated(city_key, page, page_size, date_from, date_to)
    total_pages = max(1, (total + page_size - 1) // page_size)
    return {
        "rows": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/api/city/{city_key}/daily-peaks")
async def api_city_daily_peaks(city_key: str, limit: int = 30):
    db = get_db()
    repo = WeatherRepository(db)
    rows = repo.get_recent_daily_peaks(city_key, limit)
    return [dict(r) for r in rows]


@app.get("/api/city/{city_key}/stats")
async def api_city_stats(city_key: str):
    db = get_db()
    repo = WeatherRepository(db)
    return repo.get_city_stats(city_key)


@app.get("/api/city/{city_key}/ml-results")
async def api_city_ml_results(city_key: str):
    db = get_db()
    repo = ModelResultRepository(db)
    results = repo.get_latest_results(city_key)
    # Also check in-memory cache
    trainer = get_ml_trainer()
    cached = trainer.get_cached(city_key)
    if not results and cached:
        results = [r.to_dict() for r in cached]
    return results


@app.post("/api/city/{city_key}/train-models")
async def api_city_train_models(city_key: str):
    db = get_db()
    model_repo = ModelResultRepository(db)
    trainer = get_ml_trainer()

    results = trainer.train(city_key)
    if not results:
        return {"error": "Not enough data to train (need ≥60 valid days with hourly observations)"}

    saved = []
    for r in results:
        d = r.to_dict()
        blob = trainer.pickle_model(city_key, r.model_name)
        model_repo.save_result(city_key, d, model_blob=blob)
        saved.append(d)
    return saved


@app.post("/api/backfill-all")
async def api_backfill_all(days: int = 365):
    """Trigger a 1-year backfill for ALL configured cities sequentially."""
    from src.weather.cities import CITIES
    job_id = str(uuid.uuid4())[:8]
    city_keys = list(CITIES.keys())
    _jobs[job_id] = {
        "status": "running", "type": "backfill_all",
        "total_cities": len(city_keys), "done_cities": 0,
        "current_city": None, "days": days,
        "started": time.time(),
    }
    asyncio.create_task(_run_backfill_all_job(job_id, city_keys, days))
    return {"job_id": job_id, "status": "running", "cities": len(city_keys), "days": days}


async def _run_backfill_all_job(job_id: str, city_keys: list[str], days: int) -> None:
    """Run backfills sequentially for all cities (reuses the single backfill semaphore)."""
    from src.db.repositories import JobRunRepository
    from src.weather.cities import get_city
    from src.weather.historical_loader import HistoricalLoader
    from src.weather.scraper import WUScraper

    db = get_db()
    job_repo = JobRunRepository(db)
    run_id = job_repo.create_run("backfill_all", metadata={"days": days, "cities": city_keys})

    total_stored = 0
    failed_cities: list[str] = []

    async with _backfill_semaphore:
        scraper = WUScraper(headless=True)
        await scraper.start()
        try:
            for idx, city_key in enumerate(city_keys):
                _jobs[job_id]["current_city"] = city_key
                _jobs[job_id]["done_cities"] = idx
                city = get_city(city_key)
                if city is None:
                    continue
                city_run_id = job_repo.create_run("backfill", city_key, metadata={"days": days})
                try:
                    repo = WeatherRepository(db)
                    loader = HistoricalLoader(scraper, repo)
                    await loader.backfill(city, days=days)
                    cnt = db.conn.execute(
                        "SELECT COUNT(*) as c FROM daily_peaks WHERE city_key = ?",
                        (city_key,),
                    ).fetchone()["c"]
                    total_stored += cnt
                    job_repo.finish_run(city_run_id, "success", records_processed=cnt)
                    log.info("backfill_all_city_done", city=city_key, done=idx + 1, total=len(city_keys))
                except Exception as exc:
                    failed_cities.append(city_key)
                    job_repo.finish_run(city_run_id, "error", error_message=str(exc))
                    log.error("backfill_all_city_failed", city=city_key, error=str(exc))
        finally:
            await scraper.stop()

    status = "success" if not failed_cities else "error"
    job_repo.finish_run(run_id, status,
                        records_processed=total_stored,
                        error_message=f"Failed: {failed_cities}" if failed_cities else None)
    _jobs[job_id] = {
        "status": "complete", "type": "backfill_all",
        "total_cities": len(city_keys), "done_cities": len(city_keys),
        "failed_cities": failed_cities, "days": days,
    }


@app.post("/api/train-all-models")
async def api_train_all_models():
    """Train ML models for all cities that have ≥60 valid days of data."""
    from src.weather.cities import CITIES

    db = get_db()
    model_repo = ModelResultRepository(db)
    trainer = get_ml_trainer()

    summary = {"trained": [], "skipped": [], "errors": []}

    for city_key in CITIES:
        try:
            results = trainer.train(city_key)
            if not results:
                summary["skipped"].append(city_key)
                continue
            for r in results:
                d = r.to_dict()
                blob = trainer.pickle_model(city_key, r.model_name)
                model_repo.save_result(city_key, d, model_blob=blob)
            best_auc = max(r.test_auc for r in results)
            summary["trained"].append({"city": city_key, "models": len(results), "best_auc": round(best_auc, 4)})
            log.info("train_all_city_done", city=city_key, models=len(results), best_auc=round(best_auc, 4))
        except Exception as exc:
            summary["errors"].append({"city": city_key, "error": str(exc)})
            log.error("train_all_city_failed", city=city_key, error=str(exc))

    return summary


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse("jobs.html", {"request": request})


@app.get("/api/jobs")
async def api_jobs(
    job_type: str | None = None,
    city_key: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    db = get_db()
    repo = JobRunRepository(db)
    rows = repo.list_runs(job_type=job_type, city_key=city_key, status=status, limit=limit, offset=offset)
    summary = repo.get_summary_stats()
    return {
        "runs": [dict(r) for r in rows],
        "total": summary["total_runs"],
        "summary": summary,
    }


@app.get("/api/jobs/{run_id}")
async def api_job_run(run_id: int):
    db = get_db()
    repo = JobRunRepository(db)
    row = repo.get_run(run_id)
    if not row:
        return {"error": "Not found"}
    return dict(row)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def shutdown():
    global _db
    if _db:
        _db.close()
        _db = None
