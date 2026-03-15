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
from src.db.repositories import MarketRepository, WeatherRepository
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
# Semaphore: only 1 Playwright browser at a time to avoid resource contention
_scrape_semaphore = asyncio.Semaphore(1)


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
        return None

    db = get_db()
    hist = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
        (city_key,),
    ).fetchone()["cnt"]

    if hist >= 5:
        _backfill_started.add(city_key)  # already has enough
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
    """Build enriched opportunity data for each active temperature event today."""
    from config import settings
    from src.model.features import compute_current_max
    from src.model.peak_predictor import PeakPredictor
    from src.polymarket.gamma_api import parse_brackets, parse_city_and_date
    from src.strategy.evaluator import EVEvaluator
    from src.utils.time_utils import current_hour_in_tz, today_in_tz
    from src.weather.cities import get_city

    db = get_db()
    weather_repo = WeatherRepository(db)
    market_repo = MarketRepository(db)
    evaluator = EVEvaluator(
        min_confidence=settings.min_confidence,
        min_expected_value=settings.min_expected_value,
    )
    predictor = PeakPredictor(db)

    # Get all active events
    active_events = market_repo.get_active_events()
    opportunities = []

    for event in active_events:
        city_key = event["city_key"]
        event_date = event["date"]
        city = get_city(city_key)
        if not city:
            continue

        today = today_in_tz(city.timezone)
        current_hour = current_hour_in_tz(city.timezone)
        is_today = event_date == today

        # Get brackets
        brackets = market_repo.get_brackets_for_event(event["event_id"])
        bracket_list = []
        for b in brackets:
            bracket_list.append({
                "id": b["id"],
                "label": b["bracket_label"],
                "yes_price": b["current_yes_price"] or 0,
                "no_price": b["current_no_price"] or 0,
                "token_id_yes": b["clob_token_id_yes"],
                "market_id": b["market_id"],
                "condition_id": b["condition_id"],
            })

        # Auto-backfill if this city lacks historical data
        backfill_job_id = _auto_backfill_if_needed(city_key)

        # Auto-scrape today if we don't have observations yet
        _auto_scrape_if_needed(city_key, city, event_date)

        # Get weather observations
        observations = weather_repo.get_hourly_observations(city_key, event_date)
        weather_data = None
        peak_info = None
        confidence_info = None
        recommendation = None
        auto_backfill_running = backfill_job_id is not None

        if observations:
            max_temp, peak_hour = compute_current_max(observations)
            weather_data = {
                "observation_count": len(observations),
                "max_temp": max_temp,
                "peak_hour": peak_hour,
                "latest_hour": observations[-1]["hour"],
                "latest_temp": observations[-1]["temperature"],
            }

            # Get historical peaks count
            hist_peaks = db.conn.execute(
                "SELECT COUNT(*) as cnt FROM daily_peaks WHERE city_key = ?",
                (city_key,),
            ).fetchone()["cnt"]

            if hist_peaks >= 5:
                try:
                    predictor.train(city_key, event_date, window_days=15, years_back=1)
                    prediction = predictor.predict_confidence(
                        city_key, current_hour, max_temp, observations
                    )
                    confidence_info = {
                        "confidence": round(prediction.confidence, 4),
                        "base_confidence": round(prediction.base_confidence, 4),
                        "mean_peak_hour": round(prediction.mean_peak_hour, 1),
                        "hours_past_mean": round(prediction.hours_past_mean, 1),
                    }

                    # Find matching bracket and evaluate
                    matching = evaluator.find_matching_bracket(
                        max_temp, [dict(b) for b in brackets]
                    )
                    if matching:
                        yes_price = matching.get("current_yes_price", 0) or 0
                        if yes_price > 0:
                            decision = evaluator.evaluate(
                                confidence=prediction.confidence,
                                market_price=yes_price,
                                current_max_temp=max_temp,
                                bracket_label=matching["bracket_label"],
                            )
                            recommendation = {
                                "action": decision.action,
                                "reason": decision.reason,
                                "bracket_label": matching["bracket_label"],
                                "bracket_id": matching["id"],
                                "market_price": yes_price,
                                "expected_value": round(decision.expected_value, 4),
                                "kelly_fraction": round(decision.kelly_fraction, 4),
                                "token_id_yes": matching.get("clob_token_id_yes"),
                            }
                except Exception as exc:
                    log.warning("opportunity_analysis_error", city=city_key, error=str(exc))

            peak_info = {
                "historical_days": hist_peaks,
                "has_history": hist_peaks >= 5,
            }

        slug = event["slug"]
        polymarket_url = f"https://polymarket.com/event/{slug}" if slug else None

        opportunities.append({
            "event_id": event["event_id"],
            "city_key": city_key,
            "city_name": city.display_name if city else city_key,
            "date": event_date,
            "title": event["title"],
            "slug": slug,
            "polymarket_url": polymarket_url,
            "is_today": is_today,
            "current_hour": current_hour if is_today else None,
            "timezone": city.timezone if city else None,
            "unit": city.unit if city else "celsius",
            "brackets": bracket_list,
            "weather": weather_data,
            "peak_info": peak_info,
            "confidence": confidence_info,
            "recommendation": recommendation,
            "auto_backfill_running": auto_backfill_running,
        })

    # Sort: today's events first, then by city
    opportunities.sort(key=lambda o: (not o["is_today"], o["date"], o["city_key"]))
    return opportunities


# ---------------------------------------------------------------------------
# Scrape / backfill / refresh actions
# ---------------------------------------------------------------------------

@app.post("/api/refresh-markets")
async def api_refresh_markets():
    """Re-poll Polymarket for latest events and prices."""
    db = get_db()
    market_repo = MarketRepository(db)

    from src.polymarket.market_monitor import MarketMonitor
    monitor = MarketMonitor(market_repo)
    count = await monitor.poll_all()
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
        scrape_date = date or today_in_tz(city.timezone)

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
                _jobs[job_id] = {
                    "status": "complete", "type": "scrape", "city": city.key,
                    "date": scrape_date, "observations": count,
                    "peak_temp": peak_obs.temperature, "peak_hour": peak_obs.hour,
                }
            else:
                _jobs[job_id] = {"status": "complete", "type": "scrape", "city": city.key, "observations": 0}

        except Exception as exc:
            log.error("scrape_job_failed", job_id=job_id, error=str(exc))
            _jobs[job_id] = {"status": "error", "type": "scrape", "error": str(exc)}


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
    """Background task: backfill historical data. Uses semaphore for 1-at-a-time."""
    from src.weather.historical_loader import HistoricalLoader
    from src.weather.scraper import WUScraper

    async with _scrape_semaphore:
        log.info("backfill_job_acquired_lock", job_id=job_id, city=city.key)
        db = get_db()
        repo = WeatherRepository(db)

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

            _jobs[job_id] = {
                "status": "complete", "type": "backfill", "city": city.key,
                "days": days, "total_peaks": total,
            }
        except Exception as exc:
            log.error("backfill_job_failed", job_id=job_id, error=str(exc))
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
# Shutdown
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def shutdown():
    global _db
    if _db:
        _db.close()
        _db = None
