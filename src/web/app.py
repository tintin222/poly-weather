"""FastAPI dashboard for poly-weather paper trading results."""

from __future__ import annotations

import asyncio
import dataclasses
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

# APScheduler for background periodic jobs
_scheduler = None

# WU hourly forecast cache: {city_key: {"fetched_at": float, "data": list[dict], "unit": str}}
_wu_forecast_cache: dict[str, dict] = {}

# AI bot state — persists only for this process lifetime (off by default)
_ai_bot_state: dict = {
    "enabled": True,
    "city_keys": [],        # active cities with today's event (refreshed each tick)
    "city_last_run": {},    # {city_key: float} — last analysis timestamp per city
    "last_run_at": 0.0,
    "runs_today": 0,
    "bets_today": 0,
    "last_city": None,
    "last_error": None,
    "last_min_bet_hour": None,
    "last_window": None,           # betting window string e.g. "07:00–12:00"
}

_gpt_bot_state: dict = {
    "enabled": True,
    "city_keys": [],
    "city_last_run": {},
    "last_run_at": 0.0,
    "last_city": None,
    "last_error": None,
    "last_min_bet_hour": None,
    "last_window": None,
}

_gemini_bot_state: dict = {
    "enabled": True,
    "city_keys": [],
    "city_last_run": {},
    "last_run_at": 0.0,
    "last_city": None,
    "last_error": None,
    "last_min_bet_hour": None,
    "last_window": None,
}

# NO agent state dicts
_claude_no_state: dict = {
    "enabled": True, "city_keys": [], "city_last_run": {},
    "last_run_at": 0.0, "last_city": None, "last_error": None,
    "last_min_bet_hour": None, "last_window": None,
}
_gpt_no_state: dict = {
    "enabled": True, "city_keys": [], "city_last_run": {},
    "last_run_at": 0.0, "last_city": None, "last_error": None,
    "last_min_bet_hour": None, "last_window": None,
}
_gemini_no_state: dict = {
    "enabled": True, "city_keys": [], "city_last_run": {},
    "last_run_at": 0.0, "last_city": None, "last_error": None,
    "last_min_bet_hour": None, "last_window": None,
}


def _tiered_bet_size(confidence: float, price: float) -> float:
    """Return USDC bet size based on confidence and edge. Returns 0.0 if below threshold.

    Tiers (confidence AND minimum edge both required):
      ≥88% conf, ≥25% edge  →  $200
      ≥80% conf, ≥18% edge  →  $100
      ≥75% conf, ≥12% edge  →  $40
      below any tier        →   $0  (skip)
    """
    edge = confidence - price
    if confidence >= 0.88 and edge >= 0.25:
        return 200.0
    if confidence >= 0.80 and edge >= 0.18:
        return 100.0
    if confidence >= 0.75 and edge >= 0.12:
        return 40.0
    return 0.0


def _no_bet_size(confidence: float, no_price: float) -> float:
    """Return USDC bet size for BUY_NO bets (safe, small-gain strategy).

      ≥95% conf, NO price 0.80-0.97 → $100 (3-20% return)
      ≥90% conf, NO price 0.80-0.97 → $40
      NO price > 0.97 or < 0.80     → $0
    """
    if no_price > 0.97 or no_price < 0.80:
        return 0.0
    if confidence >= 0.95:
        return 100.0
    if confidence >= 0.90:
        return 40.0
    return 0.0


def _min_bet_hour(db, city_key: str, n_days: int = 10) -> int:
    """Earliest hour betting is allowed = min(peak_hour) over last n_days.
    Falls back to 12 if fewer than 3 days of history.
    """
    rows = db.conn.execute(
        "SELECT peak_hour FROM daily_peaks "
        "WHERE city_key = ? AND peak_hour > 0 "
        "ORDER BY date DESC LIMIT ?",
        (city_key, n_days),
    ).fetchall()
    if len(rows) < 3:
        return 12
    return min(r["peak_hour"] for r in rows)


def _fetch_historical_context(db, city_key: str, n_days: int = 10) -> list[dict]:
    """Fetch n_days of historical hourly curves + peak info for LLM context."""
    peaks = db.conn.execute(
        "SELECT date, peak_temp, peak_hour, summary_high_temp "
        "FROM daily_peaks WHERE city_key = ? AND peak_hour > 0 "
        "ORDER BY date DESC LIMIT ?",
        (city_key, n_days),
    ).fetchall()
    result = []
    for p in peaks:
        hourly = db.conn.execute(
            "SELECT hour, temperature FROM hourly_observations "
            "WHERE city_key = ? AND date = ? AND hour BETWEEN 8 AND 20 "
            "ORDER BY hour",
            (city_key, p["date"]),
        ).fetchall()
        result.append({
            "date": p["date"],
            "peak_temp": p["peak_temp"],
            "peak_hour": p["peak_hour"],
            "summary_high_temp": p["summary_high_temp"],
            "hourly": [{"hour": h["hour"], "temperature": h["temperature"]} for h in hourly],
        })
    return result


def _get_best_active_event_brackets(db, city_key: str, today: str) -> tuple[list[dict], str]:
    """Return (brackets, event_date) for today's tradeable active event.

    Only looks at today's event — bots never bet on future dates since
    there would be no same-day observational data to inform the decision.
    Returns ([], today) if no tradeable event is found.
    """
    events = db.conn.execute(
        "SELECT event_id FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ?",
        (city_key, today),
    ).fetchall()
    brackets: list[dict] = []
    for ev in events:
        brows = db.conn.execute(
            "SELECT id, bracket_label, current_yes_price, current_no_price, "
            "clob_token_id_yes, clob_token_id_no FROM polymarket_brackets WHERE event_id = ?",
            (ev["event_id"],),
        ).fetchall()
        for b in brows:
            yp = b["current_yes_price"] or 0.0
            np_ = round(1.0 - yp, 4) if yp else 0.0
            brackets.append({
                "label": b["bracket_label"],
                "yes_price": yp,
                "no_price": np_,
                "yes_token_id": b["clob_token_id_yes"] or "",
                "no_token_id": b["clob_token_id_no"] or "",
                "db_id": b["id"],
                "event_id": ev["event_id"],
            })
    if any(0.05 <= b["yes_price"] <= 0.95 for b in brackets):
        return brackets, today
    return [], today


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

    # Recent bets — include placed time and profit-if-win
    recent_bets = conn.execute("""
        SELECT b.*,
               e.title as event_title,
               e.slug as event_slug,
               datetime(b.created_at, 'unixepoch') as placed_at,
               (b.size - b.cost) as profit_if_win
        FROM bets b
        LEFT JOIN polymarket_events e ON b.event_id = e.event_id
        ORDER BY b.created_at DESC
        LIMIT 50
    """).fetchall()

    # Today's watch: active markets merged with today's peaks and our latest bet per city
    from src.utils.time_utils import today_in_tz
    today = today_in_tz("UTC")
    today_status = conn.execute("""
        SELECT
            e.city_key,
            e.event_id,
            e.slug,
            e.date,
            dp.peak_temp,
            dp.peak_hour,
            dp.observation_count,
            b.bracket_label  as bet_bracket,
            b.price          as bet_price,
            b.cost           as bet_cost,
            (b.size - b.cost) as bet_profit_if_win,
            b.won            as bet_won,
            b.pnl            as bet_pnl,
            (SELECT COUNT(*) FROM polymarket_brackets br
             WHERE br.event_id = e.event_id) as bracket_count
        FROM polymarket_events e
        LEFT JOIN daily_peaks dp
               ON dp.city_key = e.city_key AND dp.date = e.date
        LEFT JOIN bets b ON b.id = (
            SELECT id FROM bets b2
            WHERE b2.city_key = e.city_key AND b2.date = e.date
            ORDER BY b2.created_at DESC LIMIT 1
        )
        WHERE e.status = 'active' AND e.date >= ?
        ORDER BY e.date, e.city_key
        LIMIT 60
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

    return templates.TemplateResponse(request, "dashboard.html", {
        "total_bets": total_bets,
        "paper_bets": paper_bets,
        "won_bets": won_bets,
        "lost_bets": lost_bets,
        "pending_bets": pending_bets,
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "recent_bets": recent_bets,
        "today_status": today_status,
        "today": today,
        "daily_pnl": daily_pnl,
        "observation_count": observation_count,
        "peak_count": peak_count,
        "cities_with_data": cities_with_data,
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
# Bets history page
# ---------------------------------------------------------------------------

@app.get("/bets", response_class=HTMLResponse)
async def bets_page(
    request: Request,
    date: str = "",
    city: str = "",
    bot: str = "",
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
):
    from src.weather.cities import CITIES
    db = get_db()
    conn = db.conn

    per_page = 50
    offset = (page - 1) * per_page

    conditions: list[str] = []
    params: list = []
    if date:
        conditions.append("b.date = ?")
        params.append(date)
    if city:
        conditions.append("b.city_key = ?")
        params.append(city)
    if bot:
        conditions.append("b.llm_source = ?")
        params.append(bot)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    allowed_sort = {"created_at": "b.created_at", "date": "b.date", "city": "b.city_key"}
    sort_col = allowed_sort.get(sort, "b.created_at")
    sort_dir = "ASC" if order == "asc" else "DESC"

    total = conn.execute(
        f"SELECT COUNT(*) as cnt FROM bets b {where}", params
    ).fetchone()["cnt"]

    bets = conn.execute(f"""
        SELECT b.*,
               (b.size - b.cost) as profit_if_win,
               datetime(b.created_at, 'unixepoch') as placed_at
        FROM bets b
        {where}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """, params + [per_page, offset]).fetchall()

    summary = conn.execute(f"""
        SELECT COUNT(*) as total_bets,
               SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) as losses,
               SUM(CASE WHEN won IS NULL THEN 1 ELSE 0 END) as pending,
               COALESCE(SUM(cost), 0) as total_staked,
               COALESCE(SUM(CASE WHEN won=1 THEN (size-cost) ELSE 0 END), 0) as won_profit,
               COALESCE(SUM(CASE WHEN won=0 THEN -cost ELSE 0 END), 0) as lost_cost,
               COALESCE(SUM(pnl), 0) as net_pnl
        FROM bets b {where}
    """, params).fetchone()

    city_keys = sorted(CITIES.keys())
    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(request, "bets.html", {
        "bets": bets,
        "summary": summary,
        "city_keys": city_keys,
        "filter_date": date,
        "filter_city": city,
        "filter_bot": bot,
        "sort": sort,
        "order": order,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "per_page": per_page,
    })


# ---------------------------------------------------------------------------
# Opportunities page
# ---------------------------------------------------------------------------

@app.get("/opportunities", response_class=HTMLResponse)
async def opportunities_page(request: Request):
    return templates.TemplateResponse(request, "opportunities.html")


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
                "best_bracket":       guidance.best_bracket,
                "best_bracket_price": guidance.best_bracket_price,
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

        # Use DB-stored WU URL if available (overrides static config)
        from src.polymarket.resolution_url_checker import get_effective_wu_url_path
        effective_path = get_effective_wu_url_path(city, db)
        if effective_path != city.wu_url_path:
            log.info("scrape_using_db_url", city=city.key, path=effective_path)
            city = dataclasses.replace(city, wu_url_path=effective_path)

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

        # Use DB-stored WU URL if available
        from src.polymarket.resolution_url_checker import get_effective_wu_url_path
        effective_path = get_effective_wu_url_path(city, db)
        if effective_path != city.wu_url_path:
            log.info("backfill_using_db_url", city=city.key, path=effective_path)
            city = dataclasses.replace(city, wu_url_path=effective_path)

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

    # Per-bot breakdown
    per_bot_rows = conn.execute("""
        SELECT llm_source,
               COUNT(*) as total_bets,
               SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) as losses,
               SUM(CASE WHEN won IS NULL THEN 1 ELSE 0 END) as pending_bets,
               COALESCE(SUM(pnl), 0) as total_pnl,
               COALESCE(AVG(CASE WHEN won IS NOT NULL THEN CAST(won AS REAL) END), 0) as win_rate
        FROM bets WHERE is_paper = 1
        GROUP BY llm_source
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
        "per_bot": [dict(r) for r in per_bot_rows],
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
    """Reset paper trading: delete all bets and analysis logs, keep config."""
    db = get_db()
    db.conn.execute("DELETE FROM bets")
    db.conn.execute("DELETE FROM llm_analysis_log")
    db.conn.commit()
    return {"status": "ok", "message": "All bets and analysis logs deleted. Balance reset."}


_BOT_SETTINGS_DEFAULTS = {
    "min_confidence": 0.75,
    "max_bet_usd": 200.0,
    "max_daily_exposure": 500.0,
}


def _load_bot_settings(conn, bot_name: str) -> dict:
    """Return bot settings dict, inserting defaults if row missing."""
    row = conn.execute("SELECT * FROM bot_settings WHERE bot_name = ?", (bot_name,)).fetchone()
    if row:
        return dict(row)
    conn.execute(
        "INSERT OR IGNORE INTO bot_settings (bot_name, min_confidence, max_bet_usd, max_daily_exposure, updated_at) VALUES (?, ?, ?, ?, ?)",
        (bot_name, _BOT_SETTINGS_DEFAULTS["min_confidence"], _BOT_SETTINGS_DEFAULTS["max_bet_usd"], _BOT_SETTINGS_DEFAULTS["max_daily_exposure"], int(time.time())),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM bot_settings WHERE bot_name = ?", (bot_name,)).fetchone())


@app.get("/api/bot-settings")
async def api_get_bot_settings():
    """Get per-bot configuration settings for all 3 bots."""
    db = get_db()
    return {bot: _load_bot_settings(db.conn, bot) for bot in ("claude", "gpt", "gemini")}


@app.post("/api/bot-settings/{bot_name}")
async def api_set_bot_settings(
    bot_name: str,
    min_confidence: float | None = None,
    max_bet_usd: float | None = None,
    max_daily_exposure: float | None = None,
):
    """Upsert per-bot configuration settings."""
    if bot_name not in ("claude", "gpt", "gemini"):
        return {"error": f"Unknown bot: {bot_name}"}
    db = get_db()
    conn = db.conn
    _load_bot_settings(conn, bot_name)  # ensure row exists
    updates = []
    params = []
    if min_confidence is not None:
        updates.append("min_confidence = ?")
        params.append(min_confidence)
    if max_bet_usd is not None:
        updates.append("max_bet_usd = ?")
        params.append(max_bet_usd)
    if max_daily_exposure is not None:
        updates.append("max_daily_exposure = ?")
        params.append(max_daily_exposure)
    if updates:
        updates.append("updated_at = ?")
        params.append(int(time.time()))
        params.append(bot_name)
        conn.execute(f"UPDATE bot_settings SET {', '.join(updates)} WHERE bot_name = ?", params)
        conn.commit()
    return {"status": "ok", "config": _load_bot_settings(conn, bot_name)}


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Config page for paper trading settings."""
    return templates.TemplateResponse(request, "config.html")


@app.get("/bots", response_class=HTMLResponse)
async def bots_page(request: Request):
    """Unified bots control page."""
    return templates.TemplateResponse(request, "bots.html")


# ---------------------------------------------------------------------------
# City detail page
# ---------------------------------------------------------------------------

@app.get("/city/{city_key}", response_class=HTMLResponse)
async def city_detail_page(city_key: str, request: Request):
    from src.weather.cities import get_city
    city = get_city(city_key)
    return templates.TemplateResponse(request, "city.html", {
        "city_key": city_key,
        "city": city,
    })


@app.get("/api/city/{city_key}/bets")
async def api_city_bets(city_key: str):
    db = get_db()
    rows = db.conn.execute(
        """SELECT b.*, e.slug, e.title, e.status as event_status
           FROM bets b
           LEFT JOIN polymarket_events e ON b.event_id = e.event_id
           WHERE b.city_key = ?
           ORDER BY b.date DESC, b.created_at DESC""",
        (city_key,),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/city/{city_key}/market")
async def api_city_market(city_key: str):
    """Active Polymarket event + brackets + bets for a city.

    Priority: today's active event > closest future active > most recent closed.
    """
    from src.weather.cities import get_city as _get_city
    from src.utils.time_utils import today_in_tz
    db = get_db()
    city_obj = _get_city(city_key)
    today = today_in_tz(city_obj.timezone) if city_obj else None

    event = None
    if today:
        # 1. Today's active event
        event = db.conn.execute(
            "SELECT * FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
            (city_key, today),
        ).fetchone()
    if not event:
        # 2. Closest future active event
        event = db.conn.execute(
            "SELECT * FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date >= ? ORDER BY date ASC LIMIT 1",
            (city_key, today or ''),
        ).fetchone()
    if not event:
        # 3. Most recent event (active or closed)
        event = db.conn.execute(
            "SELECT * FROM polymarket_events WHERE city_key = ? ORDER BY date DESC LIMIT 1",
            (city_key,),
        ).fetchone()
    if not event:
        return {"event": None, "brackets": [], "bets": []}
    brackets = db.conn.execute(
        "SELECT * FROM polymarket_brackets WHERE event_id = ? ORDER BY bracket_label",
        (event["event_id"],),
    ).fetchall()
    bets = db.conn.execute(
        "SELECT * FROM bets WHERE event_id = ? ORDER BY created_at DESC",
        (event["event_id"],),
    ).fetchall()
    return {
        "event": dict(event),
        "brackets": [dict(b) for b in brackets],
        "bets": [dict(b) for b in bets],
    }


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


@app.get("/api/city/{city_key}/peak-hours")
async def api_city_peak_hours(
    city_key: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 400,
):
    """For each date, return only the hour(s) that reached the daily max temperature.

    If two consecutive hours tied for the peak, both are returned so the user
    can see the plateau. Each row is enriched with metar_temp and om_temp where
    available for source comparison.
    """
    db = get_db()
    conditions = ["o.city_key = ?"]
    params: list = [city_key]
    if date_from:
        conditions.append("o.date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("o.date <= ?")
        params.append(date_to)
    where = " AND ".join(conditions)
    rows = db.conn.execute(
        f"""SELECT o.*, dp.summary_high_temp
            FROM hourly_observations o
            JOIN (
                SELECT date, MAX(temperature) AS peak_temp
                FROM hourly_observations
                WHERE city_key = ?
                GROUP BY date
            ) p ON o.date = p.date AND o.temperature = p.peak_temp
            LEFT JOIN daily_peaks dp ON dp.city_key = o.city_key AND dp.date = o.date
            WHERE {where}
            ORDER BY o.date DESC, o.hour ASC
            LIMIT ?""",
        [city_key, *params, limit],
    ).fetchall()

    result = [dict(r) for r in rows]
    if not result:
        return result

    from src.weather.cities import get_city
    city = get_city(city_key)
    if not city:
        return result

    dates = [r["date"] for r in result]
    start_date = min(dates)
    end_date = max(dates)

    from src.weather import metar as metar_mod, open_meteo

    async def _fetch_om():
        try:
            return await open_meteo.fetch_historical(city, start_date, end_date)
        except Exception as exc:
            log.warning("peak_hours_om_error", error=str(exc))
            return {}

    async def _fetch_metar():
        try:
            data = await metar_mod.fetch(city, hours=720)
            lookup: dict[tuple[str, int], float] = {}
            for obs in data["observations"]:
                key = (obs["date"], obs["hour"])
                lookup[key] = obs["temperature"]  # last reading in that hour wins
            return lookup
        except Exception as exc:
            log.warning("peak_hours_metar_error", error=str(exc))
            return {}

    om_lookup, metar_lookup = await asyncio.gather(_fetch_om(), _fetch_metar())

    for r in result:
        key = (r["date"], r["hour"])
        r["om_temp"] = om_lookup.get(key)
        r["metar_temp"] = metar_lookup.get(key)

    return result


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
async def api_city_train_models(city_key: str, target_date: str | None = None):
    db = get_db()
    model_repo = ModelResultRepository(db)
    trainer = get_ml_trainer()

    results = trainer.train(city_key, target_date=target_date)
    if not results:
        return {"error": "Not enough data to train (need ≥60 valid days with hourly observations)"}

    saved = []
    for r in results:
        d = r.to_dict()
        blob = trainer.pickle_model(city_key, r.model_name)
        model_repo.save_result(city_key, d, model_blob=blob)
        saved.append(d)
    return saved


@app.get("/api/city/{city_key}/metar")
async def api_city_metar(city_key: str, hours: int = 24):
    from src.weather.cities import get_city
    from src.weather import metar
    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}
    try:
        return await metar.fetch(city, hours=hours)
    except Exception as exc:
        log.error("metar_api_error", city=city_key, error=str(exc))
        return {"error": str(exc)}


@app.get("/api/city/{city_key}/open-meteo")
async def api_city_open_meteo(city_key: str):
    from src.weather.cities import get_city
    from src.weather import open_meteo
    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}
    try:
        return await open_meteo.fetch(city)
    except Exception as exc:
        log.error("open_meteo_api_error", city=city_key, error=str(exc))
        return {"error": str(exc)}


@app.get("/api/city/{city_key}/wu-forecast")
async def api_city_wu_forecast(city_key: str):
    """Return the cached WU hourly forecast for a city.

    Populated by the background scheduler every 10–30 minutes.
    Returns empty data if the city hasn't been scraped yet.
    """
    cached = _wu_forecast_cache.get(city_key)
    if not cached:
        return {"data": [], "fetched_at": None, "unit": None}
    return cached


@app.post("/api/city/{city_key}/llm-analyze")
async def api_city_llm_analyze(city_key: str, execute: bool = False):
    """Analyze today's weather data with Claude and return bracket betting recommendations.

    Args:
        city_key: City identifier.
        execute: If True, place paper bets for all actionable recommendations.

    Returns JSON with analysis narrative and per-bracket recommendations.
    """
    from config import settings
    from src.weather.cities import get_city
    from src.utils.time_utils import today_in_tz, current_hour_in_tz
    from src.weather import open_meteo, metar as metar_mod
    from src.strategy.llm_advisor import analyze as llm_analyze
    from src.polymarket.trader import PolymarketTrader

    if not settings.anthropic_api_key:
        return {"error": "ANTHROPIC_API_KEY not configured"}

    city = get_city(city_key)
    if not city:
        return {"error": f"Unknown city: {city_key}"}

    db = get_db()
    today = today_in_tz(city.timezone)
    current_hour = current_hour_in_tz(city.timezone)

    # --- Find best tradeable event (today or tomorrow if today is fully settled) ---
    brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
    if not brackets:
        return {"error": f"No active tradeable brackets found for {city_key} (today or tomorrow)"}

    # --- Gather WU observations (for event_date) ---
    wu_rows = db.conn.execute(
        "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
        "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
        (city_key, event_date),
    ).fetchall()
    wu_obs = [dict(r) for r in wu_rows]

    # --- WU summary high (for event_date) ---
    peak_row = db.conn.execute(
        "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
        (city_key, event_date),
    ).fetchone()
    summary_high = peak_row["summary_high_temp"] if peak_row else None

    # --- WU forecast from cache ---
    wu_fc_cached = _wu_forecast_cache.get(city_key, {})
    wu_forecast = wu_fc_cached.get("data", [])

    # --- Open-Meteo (filter to event_date) ---
    om_hourly: list[dict] = []
    try:
        om_data = await open_meteo.fetch(city)
        om_hourly = [
            {"hour": h["hour"], "temperature": h["temperature"], "is_forecast": h.get("is_forecast", False)}
            for h in om_data.get("hourly", []) if h.get("date") == event_date
        ]
    except Exception as exc:
        log.warning("llm_analyze_om_error", city=city_key, error=str(exc))

    # --- METAR (filter to event_date) ---
    metar_obs_list: list[dict] = []
    try:
        metar_data = await metar_mod.fetch(city, hours=24)
        metar_obs_list = [
            {"hour": o["hour"], "temperature": o["temperature"]}
            for o in metar_data.get("observations", [])
            if o.get("date") == event_date
        ]
    except Exception as exc:
        log.warning("llm_analyze_metar_error", city=city_key, error=str(exc))

    # --- Historical context (10-day hourly curves + peaks) ---
    historical_context = _fetch_historical_context(db, city_key)
    min_hour = _min_bet_hour(db, city_key)
    _manual_pos = db.conn.execute(
        "SELECT bracket_label, price, cost FROM bets "
        "WHERE city_key = ? AND date = ? AND side LIKE '%YES%' AND llm_source = 'claude' AND won IS NULL LIMIT 1",
        (city_key, event_date),
    ).fetchone()

    # --- Call LLM ---
    try:
        advice = await llm_analyze(
            api_key=settings.anthropic_api_key,
            city_name=city.display_name,
            unit=city.unit,
            today=event_date,
            current_local_hour=current_hour,
            wu_obs=wu_obs,
            wu_forecast=wu_forecast,
            om_hourly=om_hourly,
            metar_obs=metar_obs_list,
            summary_high=summary_high,
            brackets=brackets,
            historical_context=historical_context,
            min_bet_hour=min_hour,
            current_position=dict(_manual_pos) if _manual_pos else None,
        )
    except Exception as exc:
        log.error("llm_analyze_failed", city=city_key, error=str(exc))
        return {"error": f"LLM analysis failed: {exc}"}

    # --- Optionally execute bets ---
    executed: list[dict] = []
    if execute:
        trader = PolymarketTrader(settings)
        _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
        for rec in advice.recommendations:
            if rec.action == "SKIP" or not rec.token_id:
                continue
            if rec.price <= 0.01:
                continue
            # Cap: don't buy NO when it already costs >85¢ (tiny edge, ties up capital)
            if rec.action == "BUY_NO" and rec.price > 0.85:
                continue
            # Never bet NO against the market's top bracket
            if rec.action == "BUY_NO" and rec.bracket_label == _market_leader:
                continue
            # Cap: don't buy YES when already priced >92¢ (almost no upside)
            if rec.action == "BUY_YES" and rec.price > 0.92:
                continue
            # Guard: market says <8% chance — too low to bet YES against
            if rec.action == "BUY_YES" and rec.price < 0.08:
                continue
            bet_size = _tiered_bet_size(rec.confidence, rec.price)
            if bet_size == 0.0:
                continue
            size = round(bet_size / rec.price, 2)
            cost = round(rec.price * size, 2)
            order = trader.place_limit_buy(
                token_id=rec.token_id,
                price=rec.price,
                size=size,
            )
            db.conn.execute(
                """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                   side, token_id, price, size, cost, order_id, order_status,
                   current_max_temp, peak_hour, model_confidence, expected_value,
                   is_paper, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.event_id, rec.bracket_db_id, city_key, event_date,
                    rec.bracket_label, rec.action, rec.token_id,
                    rec.price, size, cost,
                    order.order_id, order.status,
                    advice.peak_temp_estimate or 0.0,
                    peak_row["peak_hour"] if peak_row else 0,
                    rec.confidence, rec.confidence - rec.price,
                    int(order.is_paper), int(time.time()),
                ),
            )
            db.conn.commit()
            executed.append({
                "bracket": rec.bracket_label,
                "action": rec.action,
                "price": rec.price,
                "size": size,
                "cost": cost,
                "order_id": order.order_id,
                "paper": order.is_paper,
            })
            log.info(
                "LLM_BET", city=city_key, bracket=rec.bracket_label,
                action=rec.action, price=rec.price, size=f"${bet_size:.0f}",
                conf=f"{rec.confidence:.0%}", edge=f"{rec.confidence - rec.price:.0%}",
                paper=order.is_paper,
            )

    return {
        "analysis": advice.analysis,
        "peak_temp_estimate": advice.peak_temp_estimate,
        "peak_confidence": advice.peak_confidence,
        "model": advice.model,
        "recommendations": [
            {
                "bracket": r.bracket_label,
                "action": r.action,
                "price": r.price,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
            }
            for r in advice.recommendations
        ],
        "executed": executed,
        "today": event_date,
        "current_hour": current_hour,
    }


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


@app.post("/api/backfill-date-range")
async def api_backfill_date_range(start_date: str, end_date: str):
    """Trigger a date-range backfill for ALL cities (e.g. March+April 2023).

    Query params:
        start_date: YYYY-MM-DD (inclusive)
        end_date:   YYYY-MM-DD (inclusive)

    Example: POST /api/backfill-date-range?start_date=2023-03-01&end_date=2023-04-30
    """
    from src.weather.cities import CITIES
    job_id = str(uuid.uuid4())[:8]
    city_keys = list(CITIES.keys())
    _jobs[job_id] = {
        "status": "running", "type": "backfill_date_range",
        "total_cities": len(city_keys), "done_cities": 0,
        "current_city": None,
        "start_date": start_date, "end_date": end_date,
        "started": time.time(),
    }
    asyncio.create_task(_run_backfill_date_range_job(job_id, city_keys, start_date, end_date))
    return {"job_id": job_id, "status": "running", "cities": len(city_keys),
            "start_date": start_date, "end_date": end_date}


async def _run_backfill_date_range_job(
    job_id: str, city_keys: list[str], start_date: str, end_date: str
) -> None:
    """Run date-range backfills sequentially for all cities."""
    from src.db.repositories import JobRunRepository
    from src.weather.cities import get_city

    db = get_db()
    job_repo = JobRunRepository(db)
    run_id = job_repo.create_run(
        "backfill_date_range",
        metadata={"start_date": start_date, "end_date": end_date, "cities": city_keys},
    )

    failed_cities: list[str] = []

    async with _backfill_semaphore:
        scraper = WUScraper(headless=True)
        await scraper.start()
        try:
            for idx, city_key in enumerate(city_keys):
                _jobs[job_id]["current_city"] = city_key
                city = get_city(city_key)
                if not city:
                    continue
                effective_path = get_effective_wu_url_path(city, db)
                if effective_path != city.wu_url_path:
                    city = dataclasses.replace(city, wu_url_path=effective_path)
                city_run_id = job_repo.create_run(
                    "backfill", city_key,
                    metadata={"start_date": start_date, "end_date": end_date},
                )
                try:
                    repo = WeatherRepository(db)
                    loader = HistoricalLoader(scraper, repo)
                    await loader.backfill_date_range(city, start_date, end_date)
                    cnt = db.conn.execute(
                        "SELECT COUNT(*) as c FROM daily_peaks WHERE city_key = ?",
                        (city_key,),
                    ).fetchone()["c"]
                    job_repo.finish_run(city_run_id, "success", records_processed=cnt)
                    log.info("backfill_date_range_city_done",
                             city=city_key, done=idx + 1, total=len(city_keys))
                except Exception as exc:
                    failed_cities.append(city_key)
                    job_repo.finish_run(city_run_id, "error", error_message=str(exc))
                    log.error("backfill_date_range_city_failed", city=city_key, error=str(exc))
                _jobs[job_id]["done_cities"] = idx + 1
        finally:
            await scraper.stop()

    job_repo.finish_run(
        run_id, "error" if failed_cities else "success",
        error_message=f"Failed: {failed_cities}" if failed_cities else None,
    )
    _jobs[job_id].update({
        "status": "complete",
        "failed_cities": failed_cities,
    })


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
                # Use DB-stored WU URL if available
                from src.polymarket.resolution_url_checker import get_effective_wu_url_path
                effective_path = get_effective_wu_url_path(city, db)
                if effective_path != city.wu_url_path:
                    log.info("backfill_all_using_db_url", city=city_key, path=effective_path)
                    city = dataclasses.replace(city, wu_url_path=effective_path)
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
    """Train ML models for all cities in a background task to avoid request timeout."""
    from src.weather.cities import CITIES

    job_id = f"train-all-{str(uuid.uuid4())[:6]}"
    _jobs[job_id] = {"status": "running", "type": "train_all", "trained": [], "skipped": [], "errors": []}

    async def _run():
        import datetime as _dt
        from src.weather.cities import CITIES as _CITIES
        db = get_db()
        model_repo = ModelResultRepository(db)
        trainer = get_ml_trainer()
        today_str = _dt.date.today().isoformat()
        for city_key in _CITIES:
            try:
                results = trainer.train(city_key, target_date=today_str)
                if not results:
                    _jobs[job_id]["skipped"].append(city_key)
                    continue
                for r in results:
                    d = r.to_dict()
                    blob = trainer.pickle_model(city_key, r.model_name)
                    model_repo.save_result(city_key, d, model_blob=blob)
                best_auc = max(r.test_auc for r in results)
                _jobs[job_id]["trained"].append({"city": city_key, "models": len(results), "best_auc": round(best_auc, 4)})
                log.info("train_all_city_done", city=city_key, best_auc=round(best_auc, 4))
            except Exception as exc:
                _jobs[job_id]["errors"].append({"city": city_key, "error": str(exc)})
                log.error("train_all_city_failed", city=city_key, error=str(exc))
        _jobs[job_id]["status"] = "complete"

    asyncio.create_task(_run())
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse(request, "jobs.html")


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
# Resolution URL tracker
# ---------------------------------------------------------------------------

@app.get("/resolution-urls", response_class=HTMLResponse)
async def resolution_urls_page(request: Request):
    return templates.TemplateResponse(request, "resolution_urls.html")


@app.get("/api/resolution-urls")
async def api_resolution_urls():
    """List all cities with their DB-stored and static WU URL paths."""
    from src.weather.cities import CITIES

    db = get_db()
    rows = {
        r["city_key"]: dict(r)
        for r in db.conn.execute("SELECT * FROM city_resolution_urls").fetchall()
    }

    result = []
    for city_key, city in CITIES.items():
        db_row = rows.get(city_key, {})
        wu_url_path_db = db_row.get("wu_url_path")
        matches_static = (wu_url_path_db == city.wu_url_path) if wu_url_path_db else None

        if not db_row:
            status = "unchecked"
        elif db_row.get("previous_url") and db_row["previous_url"] != wu_url_path_db:
            status = "changed"
        else:
            status = "ok"

        # METAR: uses wu_station_id directly — the same ICAO station as WU
        metar_url = (
            f"https://aviationweather.gov/metar/data"
            f"?ids={city.wu_station_id}&format=decoded&date=0&hours=24&taf=false"
        )

        # Open-Meteo: coordinate-based, nearest grid point to the city's lat/lon
        open_meteo_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={city.lat}&longitude={city.lon}"
            f"&current=temperature_2m,weather_code"
            f"&timezone={city.timezone}"
        )

        result.append({
            "city_key": city_key,
            "city_name": city.display_name,
            "wu_url_path_static": city.wu_url_path,
            "wu_url_path_db": wu_url_path_db,
            "wu_url": db_row.get("wu_url"),
            "previous_url": db_row.get("previous_url"),
            "changed_at": db_row.get("changed_at"),
            "last_checked_at": db_row.get("last_checked_at"),
            "check_count": db_row.get("check_count", 0),
            "source_slug": db_row.get("source_slug"),
            "matches_static": matches_static,
            "status": status,
            "metar_station": city.wu_station_id,
            "metar_url": metar_url,
            "open_meteo_url": open_meteo_url,
            "lat": city.lat,
            "lon": city.lon,
        })
    return result


@app.post("/api/admin/check-resolution-urls")
async def api_check_resolution_urls():
    """Trigger an immediate resolution URL check for all cities."""
    db = get_db()
    job_id = f"res-url-{uuid.uuid4().hex[:6]}"
    _jobs[job_id] = {"status": "running", "type": "resolution_url_check", "started": time.time()}

    async def _run():
        from src.polymarket.resolution_url_checker import check_all_cities
        try:
            results = await check_all_cities(db)
            _jobs[job_id].update({
                "status": "complete",
                "total": len(results),
                "changed": sum(1 for r in results if r.get("status") == "changed"),
                "not_found": sum(1 for r in results if r.get("status") == "not_found"),
                "results": results,
            })
        except Exception as exc:
            log.error("resolution_url_check_job_failed", error=str(exc))
            _jobs[job_id].update({"status": "error", "error": str(exc)})

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.post("/api/admin/fix-resolution-urls")
async def api_fix_resolution_urls():
    """Audit DB wu_url_path vs static cities.py config, return diff report.

    The DB-stored path overrides the static config for future scrapes.
    This endpoint surfaces any discrepancies so operators can investigate.
    """
    from src.weather.cities import CITIES

    db = get_db()
    rows = {
        r["city_key"]: dict(r)
        for r in db.conn.execute("SELECT * FROM city_resolution_urls").fetchall()
    }

    report = []
    for city_key, city in CITIES.items():
        db_row = rows.get(city_key, {})
        db_path = db_row.get("wu_url_path")
        static_path = city.wu_url_path
        if db_path is None:
            match = None
            status = "unchecked"
        elif db_path == static_path:
            match = True
            status = "match"
        else:
            match = False
            status = "mismatch"
        report.append({
            "city_key": city_key,
            "city_name": city.display_name,
            "static_path": static_path,
            "db_path": db_path,
            "match": match,
            "status": status,
        })

    mismatches = [r for r in report if r["match"] is False]
    log.info(
        "resolution_url_audit",
        total=len(report),
        mismatches=len(mismatches),
        unchecked=sum(1 for r in report if r["match"] is None),
    )
    return {
        "report": report,
        "total": len(report),
        "mismatches": len(mismatches),
        "unchecked": sum(1 for r in report if r["match"] is None),
    }


@app.post("/api/admin/clean-resolution-url-dots")
async def api_clean_resolution_url_dots():
    """Strip trailing dots from wu_url_path values caused by early regex bug.

    Safe to call multiple times — idempotent.
    """
    db = get_db()
    result = db.conn.execute(
        "SELECT COUNT(*) FROM city_resolution_urls WHERE wu_url_path LIKE '%.'"
    ).fetchone()
    count = result[0] if result else 0

    if count > 0:
        db.conn.execute("""
            UPDATE city_resolution_urls
            SET wu_url_path = rtrim(wu_url_path, './,''\")')
            WHERE wu_url_path LIKE '%.'
        """)
        db.conn.execute("""
            UPDATE city_resolution_urls
            SET wu_url = 'https://www.wunderground.com/history/daily/' || wu_url_path
            WHERE wu_url_path IS NOT NULL
        """)
        db.conn.commit()
        log.info("admin_clean_resolution_url_dots", rows_fixed=count)

    return {"ok": True, "rows_fixed": count}


# ---------------------------------------------------------------------------
# Admin migrations
# ---------------------------------------------------------------------------

@app.post("/api/admin/fix-celsius-rounding")
async def api_fix_celsius_rounding():
    """Round all stored Celsius temperatures to nearest integer.

    WU displays Celsius as whole numbers (e.g. 91°F → 33°C, not 32.8°C).
    Old scraped data was stored with 1 decimal place, causing bracket
    mismatches. This migration rounds hourly_observations and daily_peaks
    for all Celsius cities without re-scraping.
    """
    from src.weather.cities import CITIES
    celsius_keys = [c.key for c in CITIES.values() if c.unit == "celsius"]
    placeholders = ",".join("?" * len(celsius_keys))
    db = get_db()

    obs_result = db.conn.execute(
        f"""UPDATE hourly_observations
            SET temperature = ROUND(temperature),
                dew_point = CASE WHEN dew_point IS NOT NULL THEN ROUND(dew_point) ELSE NULL END
            WHERE city_key IN ({placeholders})""",
        celsius_keys,
    )
    obs_updated = obs_result.rowcount

    # Recompute peak_temp from the now-rounded hourly observations
    peaks_result = db.conn.execute(
        f"""UPDATE daily_peaks
            SET peak_temp = (
                SELECT MAX(temperature)
                FROM hourly_observations
                WHERE hourly_observations.city_key = daily_peaks.city_key
                  AND hourly_observations.date = daily_peaks.date
            )
            WHERE city_key IN ({placeholders})""",
        celsius_keys,
    )
    peaks_updated = peaks_result.rowcount
    db.conn.commit()

    log.info("celsius_rounding_fixed", obs_updated=obs_updated, peaks_updated=peaks_updated, cities=celsius_keys)
    return {
        "ok": True,
        "obs_updated": obs_updated,
        "peaks_updated": peaks_updated,
        "cities": celsius_keys,
    }


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _scheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    _scheduler = AsyncIOScheduler()

    async def _run_resolution_url_check():
        db = get_db()
        from src.polymarket.resolution_url_checker import check_all_cities
        try:
            await check_all_cities(db)
        except Exception as exc:
            log.error("scheduled_resolution_check_failed", error=str(exc))

    _scheduler.add_job(
        _run_resolution_url_check,
        "interval",
        hours=3,
        id="resolution_url_check",
    )

    async def _run_wu_forecast_job():
        """Scrape WU hourly forecasts for all cities, respecting adaptive intervals."""
        import dataclasses as _dc
        from src.weather.cities import CITIES
        from src.weather.scraper import WUScraper
        from src.utils.time_utils import current_hour_in_tz
        from src.polymarket.resolution_url_checker import get_effective_wu_url_path

        db = get_db()
        now = time.time()
        cities_to_scrape = []

        for city in CITIES.values():
            last = _wu_forecast_cache.get(city.key, {}).get("fetched_at", 0)
            elapsed = now - last
            city_hour = current_hour_in_tz(city.timezone)

            row = db.conn.execute(
                "SELECT AVG(peak_hour) as avg FROM daily_peaks WHERE city_key = ? AND peak_hour > 0",
                (city.key,),
            ).fetchone()
            avg_peak = int(row["avg"]) if row and row["avg"] else 14

            near_peak = abs(city_hour - avg_peak) <= 1
            threshold = 600 if near_peak else 1800  # 10 min vs 30 min

            if elapsed >= threshold:
                cities_to_scrape.append(city)

        if not cities_to_scrape:
            return

        log.info("wu_forecast_job_start", cities=len(cities_to_scrape))
        scraper = WUScraper(headless=True)
        await scraper.start()
        try:
            for city in cities_to_scrape:
                effective_path = get_effective_wu_url_path(city, db)
                if effective_path != city.wu_url_path:
                    city = _dc.replace(city, wu_url_path=effective_path)
                try:
                    data = await scraper.scrape_forecast_hourly(city)
                    _wu_forecast_cache[city.key] = {
                        "fetched_at": time.time(),
                        "data": data,
                        "unit": city.unit,
                    }
                except Exception as exc:
                    log.warning("wu_forecast_job_city_failed", city=city.key, error=str(exc))
        finally:
            await scraper.stop()
        log.info("wu_forecast_job_done", cities=len(cities_to_scrape))

    _scheduler.add_job(
        _run_wu_forecast_job,
        "interval",
        minutes=10,
        id="wu_forecast_poll",
    )

    async def _run_market_poll_and_resolve():
        """Poll Polymarket for latest prices and resolve any settled bets."""
        db = get_db()
        market_repo = MarketRepository(db)
        from src.polymarket.market_monitor import MarketMonitor
        monitor = MarketMonitor(market_repo)
        try:
            count = await monitor.poll_all()
            log.info("market_poll_done", events=count)
        except Exception as exc:
            log.error("market_poll_failed", error=str(exc))

    _scheduler.add_job(
        _run_market_poll_and_resolve,
        "interval",
        minutes=30,
        id="market_poll_resolve",
    )

    async def _run_llm_auto_bet():
        """Run LLM analysis + auto-bet for ALL active cities each tick (5-min per-city cooldown)."""
        from config import settings
        from src.weather.cities import get_city, CITIES
        from src.utils.time_utils import today_in_tz, current_hour_in_tz
        from src.weather import open_meteo, metar as metar_mod
        from src.strategy.llm_advisor import analyze as llm_analyze
        from src.polymarket.trader import PolymarketTrader
        import json as _json

        if not _ai_bot_state["enabled"]:
            return
        if not settings.anthropic_api_key:
            return

        db = get_db()
        now = time.time()
        _COOLDOWN = 300  # analyze each city at most every 5 minutes

        # Refresh city list every tick
        active_keys: list[str] = []
        for ck, city_obj in CITIES.items():
            _today = today_in_tz(city_obj.timezone)
            row = db.conn.execute(
                "SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
                (ck, _today),
            ).fetchone()
            if row:
                active_keys.append(ck)
        _ai_bot_state["city_keys"] = active_keys
        if not active_keys:
            return

        for city_key in active_keys:
            # Per-city cooldown: skip if analyzed within the last 5 minutes
            if now - _ai_bot_state["city_last_run"].get(city_key, 0) < _COOLDOWN:
                continue
            # Mark as checked immediately so parallel ticks don't double-fire
            _ai_bot_state["city_last_run"][city_key] = now
            _ai_bot_state["last_city"] = city_key

            city = get_city(city_key)
            if not city:
                continue

            today = today_in_tz(city.timezone)
            current_hour = current_hour_in_tz(city.timezone)

            if current_hour < 7:
                continue

            try:
                # Find best tradeable event — today only
                brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
                if not brackets:
                    continue

                # WU observations (for the event date)
                wu_rows = db.conn.execute(
                    "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
                    "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
                    (city_key, event_date),
                ).fetchall()
                wu_obs = [dict(r) for r in wu_rows]

                # WU summary high (for event date)
                peak_row = db.conn.execute(
                    "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
                    (city_key, event_date),
                ).fetchone()
                summary_high = peak_row["summary_high_temp"] if peak_row else None

                # Guard: observed max before 8 AM is an overnight low, not the daily peak
                if peak_row and peak_row["peak_hour"] < 8:
                    continue

                # WU forecast from cache
                wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])

                # Open-Meteo (filter to event_date so tomorrow forecasts align)
                om_hourly: list[dict] = []
                try:
                    om_data = await open_meteo.fetch(city)
                    om_hourly = [
                        {"hour": h["hour"], "temperature": h["temperature"], "is_forecast": h.get("is_forecast", False)}
                        for h in om_data.get("hourly", []) if h.get("date") == event_date
                    ]
                except Exception:
                    pass

                # METAR (filter to event_date)
                metar_obs_list: list[dict] = []
                try:
                    metar_data = await metar_mod.fetch(city, hours=24)
                    metar_obs_list = [
                        {"hour": o["hour"], "temperature": o["temperature"]}
                        for o in metar_data.get("observations", [])
                        if o.get("date") == event_date
                    ]
                except Exception:
                    pass

                # Historical context (10-day hourly curves + peaks)
                historical_context = _fetch_historical_context(db, city_key)
                min_hour = _min_bet_hour(db, city_key)
                _ai_bot_state["last_min_bet_hour"] = min_hour
                _ai_bot_state["last_window"] = f"{min_hour:02d}:00+"
                if current_hour < min_hour:
                    continue

                _claude_pos = db.conn.execute(
                    "SELECT id, bracket_label, token_id, price, size, cost FROM bets "
                    "WHERE city_key = ? AND date = ? AND side LIKE '%YES%' AND llm_source = 'claude' AND won IS NULL LIMIT 1",
                    (city_key, event_date),
                ).fetchone()
                _claude_pos_dict = dict(_claude_pos) if _claude_pos else None

                advice = await llm_analyze(
                    api_key=settings.anthropic_api_key,
                    city_name=city.display_name,
                    unit=city.unit,
                    today=event_date,
                    current_local_hour=current_hour,
                    wu_obs=wu_obs,
                    wu_forecast=wu_forecast,
                    om_hourly=om_hourly,
                    metar_obs=metar_obs_list,
                    summary_high=summary_high,
                    brackets=brackets,
                    historical_context=historical_context,
                    min_bet_hour=min_hour,
                    current_position=_claude_pos_dict,
                )

                executed_count = 0
                yes_placed_this_run = False
                trader = PolymarketTrader(settings)
                _claude_cfg = _load_bot_settings(db.conn, "claude")
                _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
                _bracket_prices = {b["label"]: b["yes_price"] for b in brackets}
                for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                    if rec.action == "SKIP" or not rec.token_id:
                        continue
                    if rec.price <= 0.01:
                        continue
                    if rec.action != "BUY_YES":
                        continue  # YES agent: only BUY_YES
                    if rec.price > 0.92:
                        continue
                    if rec.price < 0.08:
                        continue  # market says <8% chance; too low to bet YES against
                    if yes_placed_this_run:
                        continue
                    if _claude_pos:
                            if _claude_pos["bracket_label"] == rec.bracket_label:
                                continue  # already holding this bracket
                            # Sell old position, then buy new one
                            old_sell_price = _bracket_prices.get(_claude_pos["bracket_label"], 0)
                            if old_sell_price > 0:
                                sell_order = trader.place_limit_sell(
                                    token_id=_claude_pos["token_id"], price=old_sell_price, size=_claude_pos["size"])
                                sell_pnl = round(old_sell_price * _claude_pos["size"] - _claude_pos["cost"], 2)
                                db.conn.execute(
                                    "UPDATE bets SET won = 2, pnl = ?, resolved_at = ? WHERE id = ?",
                                    (sell_pnl, int(time.time()), _claude_pos["id"]))
                                db.conn.commit()
                                _claude_pos = None  # cleared
                                log.info("POSITION_SOLD", city=city_key, old_bracket=_claude_pos_dict["bracket_label"],
                                         new_bracket=rec.bracket_label, sell_pnl=sell_pnl)
                            else:
                                continue  # can't sell at 0
                    if rec.confidence < _claude_cfg["min_confidence"]:
                        continue
                    bet_size = _tiered_bet_size(rec.confidence, rec.price)
                    if bet_size == 0.0:
                        continue
                    bet_size = min(bet_size, _claude_cfg["max_bet_usd"])
                    existing = db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? AND llm_source = 'claude' LIMIT 1",
                        (city_key, event_date, rec.bracket_label, rec.action),
                    ).fetchone()
                    if existing:
                        continue
                    # Cross-bot dedup: skip if any bot already holds the opposing side
                    opposite_side = "BUY_NO" if rec.action == "BUY_YES" else "BUY_YES"
                    if db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? LIMIT 1",
                        (city_key, event_date, rec.bracket_label, opposite_side),
                    ).fetchone():
                        continue

                    size = round(bet_size / rec.price, 2)
                    cost = round(rec.price * size, 2)
                    today_spent = db.conn.execute(
                        "SELECT COALESCE(SUM(cost), 0) FROM bets WHERE llm_source = 'claude' AND date = ?",
                        (event_date,),
                    ).fetchone()[0]
                    if today_spent + cost > _claude_cfg["max_daily_exposure"]:
                        break
                    order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                    db.conn.execute(
                        """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                           side, token_id, price, size, cost, order_id, order_status,
                           current_max_temp, peak_hour, model_confidence, expected_value,
                           is_paper, created_at, llm_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            rec.event_id, rec.bracket_db_id, city_key, event_date,
                            rec.bracket_label, rec.action, rec.token_id,
                            rec.price, size, cost,
                            order.order_id, order.status,
                            advice.peak_temp_estimate or 0.0,
                            peak_row["peak_hour"] if peak_row else 0,
                            rec.confidence, rec.confidence - rec.price,
                            int(order.is_paper), int(time.time()), "claude",
                        ),
                    )
                    db.conn.commit()
                    executed_count += 1
                    if rec.action == "BUY_YES":
                        yes_placed_this_run = True
                    log.info(
                        "AI_BOT_BET", city=city_key, bracket=rec.bracket_label,
                        action=rec.action, price=rec.price, size=f"${bet_size:.0f}",
                        conf=f"{rec.confidence:.0%}", edge=f"{rec.confidence - rec.price:.0%}",
                    )

                recs_json = _json.dumps([
                    {"bracket": r.bracket_label, "action": r.action,
                     "confidence": r.confidence, "price": r.price, "reasoning": r.reasoning}
                    for r in advice.recommendations
                ])
                db.conn.execute(
                    """INSERT INTO llm_analysis_log
                       (city_key, date, analysis_time, hour_local, peak_estimate, peak_confidence,
                        analysis_text, recs_json, bets_executed, model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (city_key, event_date, int(time.time()), current_hour,
                     advice.peak_temp_estimate, advice.peak_confidence,
                     advice.analysis, recs_json, executed_count, advice.model),
                )
                db.conn.commit()

                _ai_bot_state["last_run_at"] = time.time()
                _ai_bot_state["runs_today"] += 1
                _ai_bot_state["bets_today"] += executed_count
                _ai_bot_state["last_error"] = None
                log.info("ai_bot_tick", city=city_key, event_date=event_date, bets=executed_count, hour=current_hour)

            except Exception as exc:
                _ai_bot_state["last_error"] = str(exc)
                log.error("ai_bot_tick_failed", city=city_key, error=str(exc))

    _scheduler.add_job(
        _run_llm_auto_bet,
        "interval",
        minutes=1,
        id="llm_auto_bet",
    )

    async def _run_gpt_auto_bet() -> None:
        """Run GPT analysis + auto-bet for ALL active cities each tick (5-min per-city cooldown)."""
        from config import settings
        from src.weather.cities import get_city, CITIES
        from src.utils.time_utils import today_in_tz, current_hour_in_tz
        from src.weather import open_meteo, metar as metar_mod
        from src.strategy.openai_advisor import analyze as gpt_analyze
        from src.polymarket.trader import PolymarketTrader
        import json as _json

        if not _gpt_bot_state["enabled"]:
            return
        if not settings.openai_api_key:
            return

        db = get_db()
        now = time.time()
        _COOLDOWN = 300

        active_keys: list[str] = []
        for ck, city_obj in CITIES.items():
            _today = today_in_tz(city_obj.timezone)
            row = db.conn.execute(
                "SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
                (ck, _today),
            ).fetchone()
            if row:
                active_keys.append(ck)
        _gpt_bot_state["city_keys"] = active_keys
        if not active_keys:
            return

        for city_key in active_keys:
            if now - _gpt_bot_state["city_last_run"].get(city_key, 0) < _COOLDOWN:
                continue
            _gpt_bot_state["city_last_run"][city_key] = now
            _gpt_bot_state["last_city"] = city_key

            city = get_city(city_key)
            if not city:
                continue

            today = today_in_tz(city.timezone)
            current_hour = current_hour_in_tz(city.timezone)

            if current_hour < 7:
                continue

            try:
                brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
                if not brackets:
                    continue

                wu_rows = db.conn.execute(
                    "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
                    "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
                    (city_key, event_date),
                ).fetchall()
                wu_obs = [dict(r) for r in wu_rows]

                peak_row = db.conn.execute(
                    "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
                    (city_key, event_date),
                ).fetchone()
                summary_high = peak_row["summary_high_temp"] if peak_row and peak_row["summary_high_temp"] else (
                    peak_row["peak_temp"] if peak_row else None
                )

                # Guard: observed max before 8 AM is an overnight low, not the daily peak
                if peak_row and peak_row["peak_hour"] < 8:
                    continue

                wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])

                om_hourly: list[dict] = []
                try:
                    om_data = await open_meteo.fetch(city)
                    om_hourly = om_data.get("hourly", [])
                except Exception:
                    pass

                metar_obs_list: list[dict] = []
                try:
                    metar_data = await metar_mod.fetch(city, hours=24)
                    metar_obs_list = [
                        {"hour": o["hour"], "temperature": o["temperature"]}
                        for o in metar_data.get("observations", [])
                        if o.get("date") == event_date
                    ]
                except Exception:
                    pass

                historical_context = _fetch_historical_context(db, city_key)
                min_hour = _min_bet_hour(db, city_key)
                _gpt_bot_state["last_min_bet_hour"] = min_hour
                _gpt_bot_state["last_window"] = f"{min_hour:02d}:00+"
                if current_hour < min_hour:
                    continue

                _gpt_pos = db.conn.execute(
                    "SELECT id, bracket_label, token_id, price, size, cost FROM bets "
                    "WHERE city_key = ? AND date = ? AND side LIKE '%YES%' AND llm_source = 'gpt' AND won IS NULL LIMIT 1",
                    (city_key, event_date),
                ).fetchone()
                _gpt_pos_dict = dict(_gpt_pos) if _gpt_pos else None

                advice = await gpt_analyze(
                    api_key=settings.openai_api_key,
                    city_name=city.display_name,
                    unit=city.unit,
                    today=event_date,
                    current_local_hour=current_hour,
                    wu_obs=wu_obs,
                    wu_forecast=wu_forecast,
                    om_hourly=om_hourly,
                    metar_obs=metar_obs_list,
                    summary_high=summary_high,
                    brackets=brackets,
                    historical_context=historical_context,
                    min_bet_hour=min_hour,
                    current_position=_gpt_pos_dict,
                )

                executed_count = 0
                yes_placed_this_run = False
                trader = PolymarketTrader(settings)
                _gpt_cfg = _load_bot_settings(db.conn, "gpt")
                _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
                _bracket_prices = {b["label"]: b["yes_price"] for b in brackets}
                for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                    if rec.action == "SKIP" or not rec.token_id:
                        continue
                    if rec.price <= 0.01:
                        continue
                    if rec.action != "BUY_YES":
                        continue  # YES agent: only BUY_YES
                    if rec.price > 0.92:
                        continue
                    if rec.price < 0.08:
                        continue
                    if yes_placed_this_run:
                        continue
                    if _gpt_pos:
                            if _gpt_pos["bracket_label"] == rec.bracket_label:
                                continue
                            old_sell_price = _bracket_prices.get(_gpt_pos["bracket_label"], 0)
                            if old_sell_price > 0:
                                sell_order = trader.place_limit_sell(
                                    token_id=_gpt_pos["token_id"], price=old_sell_price, size=_gpt_pos["size"])
                                sell_pnl = round(old_sell_price * _gpt_pos["size"] - _gpt_pos["cost"], 2)
                                db.conn.execute(
                                    "UPDATE bets SET won = 2, pnl = ?, resolved_at = ? WHERE id = ?",
                                    (sell_pnl, int(time.time()), _gpt_pos["id"]))
                                db.conn.commit()
                                log.info("POSITION_SOLD", city=city_key, old_bracket=_gpt_pos_dict["bracket_label"],
                                         new_bracket=rec.bracket_label, sell_pnl=sell_pnl)
                                _gpt_pos = None
                            else:
                                continue
                    if rec.confidence < _gpt_cfg["min_confidence"]:
                        continue
                    bet_size = _tiered_bet_size(rec.confidence, rec.price)
                    if bet_size == 0.0:
                        continue
                    bet_size = min(bet_size, _gpt_cfg["max_bet_usd"])
                    existing = db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? AND llm_source = 'gpt' LIMIT 1",
                        (city_key, event_date, rec.bracket_label, rec.action),
                    ).fetchone()
                    if existing:
                        continue
                    # Cross-bot dedup: skip if any bot already holds the opposing side
                    opposite_side = "BUY_NO" if rec.action == "BUY_YES" else "BUY_YES"
                    if db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? LIMIT 1",
                        (city_key, event_date, rec.bracket_label, opposite_side),
                    ).fetchone():
                        continue

                    size = round(bet_size / rec.price, 2)
                    cost = round(rec.price * size, 2)
                    today_spent = db.conn.execute(
                        "SELECT COALESCE(SUM(cost), 0) FROM bets WHERE llm_source = 'gpt' AND date = ?",
                        (event_date,),
                    ).fetchone()[0]
                    if today_spent + cost > _gpt_cfg["max_daily_exposure"]:
                        break
                    order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                    db.conn.execute(
                        """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                           side, token_id, price, size, cost, order_id, order_status,
                           current_max_temp, peak_hour, model_confidence, expected_value,
                           is_paper, created_at, llm_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            rec.event_id, rec.bracket_db_id, city_key, event_date,
                            rec.bracket_label, rec.action, rec.token_id,
                            rec.price, size, cost,
                            order.order_id, order.status,
                            advice.peak_temp_estimate or 0.0,
                            peak_row["peak_hour"] if peak_row else 0,
                            rec.confidence, rec.confidence - rec.price,
                            int(order.is_paper), int(time.time()), "gpt",
                        ),
                    )
                    db.conn.commit()
                    executed_count += 1
                    if rec.action == "BUY_YES":
                        yes_placed_this_run = True
                    log.info(
                        "GPT_BOT_BET", city=city_key, bracket=rec.bracket_label,
                        action=rec.action, price=rec.price, size=f"${bet_size:.0f}",
                        conf=f"{rec.confidence:.0%}", edge=f"{rec.confidence - rec.price:.0%}",
                    )

                recs_json = _json.dumps([
                    {"bracket": r.bracket_label, "action": r.action,
                     "confidence": r.confidence, "price": r.price, "reasoning": r.reasoning}
                    for r in advice.recommendations
                ])
                db.conn.execute(
                    """INSERT INTO llm_analysis_log
                       (city_key, date, analysis_time, hour_local, peak_estimate, peak_confidence,
                        analysis_text, recs_json, bets_executed, model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (city_key, event_date, int(time.time()), current_hour,
                     advice.peak_temp_estimate, advice.peak_confidence,
                     advice.analysis, recs_json, executed_count, advice.model),
                )
                db.conn.commit()

                _gpt_bot_state["last_run_at"] = time.time()
                _gpt_bot_state["last_error"] = None
                log.info("gpt_bot_tick", city=city_key, event_date=event_date, bets=executed_count, hour=current_hour)

            except Exception as exc:
                _gpt_bot_state["last_error"] = str(exc)
                log.error("gpt_bot_tick_failed", city=city_key, error=str(exc))

    _scheduler.add_job(
        _run_gpt_auto_bet,
        "interval",
        minutes=1,
        seconds=30,
        id="gpt_auto_bet",
    )

    async def _run_gemini_auto_bet() -> None:
        """Run Gemini analysis + auto-bet for ALL active cities each tick (5-min per-city cooldown)."""
        from config import settings
        from src.weather.cities import get_city, CITIES
        from src.utils.time_utils import today_in_tz, current_hour_in_tz
        from src.weather import open_meteo, metar as metar_mod
        from src.strategy.gemini_advisor import analyze as gemini_analyze
        from src.polymarket.trader import PolymarketTrader
        import json as _json

        if not _gemini_bot_state["enabled"]:
            return
        if not settings.gemini_api_key:
            return

        db = get_db()
        now = time.time()
        _COOLDOWN = 300

        active_keys: list[str] = []
        for ck, city_obj in CITIES.items():
            _today = today_in_tz(city_obj.timezone)
            row = db.conn.execute(
                "SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
                (ck, _today),
            ).fetchone()
            if row:
                active_keys.append(ck)
        _gemini_bot_state["city_keys"] = active_keys
        if not active_keys:
            return

        for city_key in active_keys:
            if now - _gemini_bot_state["city_last_run"].get(city_key, 0) < _COOLDOWN:
                continue
            _gemini_bot_state["city_last_run"][city_key] = now
            _gemini_bot_state["last_city"] = city_key

            city = get_city(city_key)
            if not city:
                continue

            today = today_in_tz(city.timezone)
            current_hour = current_hour_in_tz(city.timezone)

            if current_hour < 7:
                continue

            try:
                brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
                if not brackets:
                    continue

                wu_rows = db.conn.execute(
                    "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
                    "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
                    (city_key, event_date),
                ).fetchall()
                wu_obs = [dict(r) for r in wu_rows]

                peak_row = db.conn.execute(
                    "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
                    (city_key, event_date),
                ).fetchone()
                summary_high = peak_row["summary_high_temp"] if peak_row and peak_row["summary_high_temp"] else (
                    peak_row["peak_temp"] if peak_row else None
                )

                # Guard: observed max before 8 AM is an overnight low, not the daily peak
                if peak_row and peak_row["peak_hour"] < 8:
                    continue

                wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])

                om_hourly: list[dict] = []
                try:
                    om_data = await open_meteo.fetch(city)
                    om_hourly = om_data.get("hourly", [])
                except Exception:
                    pass

                metar_obs_list: list[dict] = []
                try:
                    metar_data = await metar_mod.fetch(city, hours=24)
                    metar_obs_list = [
                        {"hour": o["hour"], "temperature": o["temperature"]}
                        for o in metar_data.get("observations", [])
                        if o.get("date") == event_date
                    ]
                except Exception:
                    pass

                historical_context = _fetch_historical_context(db, city_key)
                min_hour = _min_bet_hour(db, city_key)
                _gemini_bot_state["last_min_bet_hour"] = min_hour
                _gemini_bot_state["last_window"] = f"{min_hour:02d}:00+"
                if current_hour < min_hour:
                    continue

                _gemini_pos = db.conn.execute(
                    "SELECT id, bracket_label, token_id, price, size, cost FROM bets "
                    "WHERE city_key = ? AND date = ? AND side LIKE '%YES%' AND llm_source = 'gemini' AND won IS NULL LIMIT 1",
                    (city_key, event_date),
                ).fetchone()
                _gemini_pos_dict = dict(_gemini_pos) if _gemini_pos else None

                advice = await gemini_analyze(
                    api_key=settings.gemini_api_key,
                    city_name=city.display_name,
                    unit=city.unit,
                    today=event_date,
                    current_local_hour=current_hour,
                    wu_obs=wu_obs,
                    wu_forecast=wu_forecast,
                    om_hourly=om_hourly,
                    metar_obs=metar_obs_list,
                    summary_high=summary_high,
                    brackets=brackets,
                    historical_context=historical_context,
                    min_bet_hour=min_hour,
                    current_position=_gemini_pos_dict,
                )

                executed_count = 0
                yes_placed_this_run = False
                trader = PolymarketTrader(settings)
                _gemini_cfg = _load_bot_settings(db.conn, "gemini")
                _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
                _bracket_prices = {b["label"]: b["yes_price"] for b in brackets}
                for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                    if rec.action == "SKIP" or not rec.token_id:
                        continue
                    if rec.price <= 0.01:
                        continue
                    if rec.action != "BUY_YES":
                        continue  # YES agent: only BUY_YES
                    if rec.price > 0.92:
                        continue
                    if rec.price < 0.08:
                        continue
                    if yes_placed_this_run:
                        continue
                    if _gemini_pos:
                            if _gemini_pos["bracket_label"] == rec.bracket_label:
                                continue
                            old_sell_price = _bracket_prices.get(_gemini_pos["bracket_label"], 0)
                            if old_sell_price > 0:
                                sell_order = trader.place_limit_sell(
                                    token_id=_gemini_pos["token_id"], price=old_sell_price, size=_gemini_pos["size"])
                                sell_pnl = round(old_sell_price * _gemini_pos["size"] - _gemini_pos["cost"], 2)
                                db.conn.execute(
                                    "UPDATE bets SET won = 2, pnl = ?, resolved_at = ? WHERE id = ?",
                                    (sell_pnl, int(time.time()), _gemini_pos["id"]))
                                db.conn.commit()
                                log.info("POSITION_SOLD", city=city_key, old_bracket=_gemini_pos_dict["bracket_label"],
                                         new_bracket=rec.bracket_label, sell_pnl=sell_pnl)
                                _gemini_pos = None
                            else:
                                continue
                    if rec.confidence < _gemini_cfg["min_confidence"]:
                        continue
                    bet_size = _tiered_bet_size(rec.confidence, rec.price)
                    if bet_size == 0.0:
                        continue
                    bet_size = min(bet_size, _gemini_cfg["max_bet_usd"])
                    existing = db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? AND llm_source = 'gemini' LIMIT 1",
                        (city_key, event_date, rec.bracket_label, rec.action),
                    ).fetchone()
                    if existing:
                        continue
                    # Cross-bot dedup: skip if any bot already holds the opposing side
                    opposite_side = "BUY_NO" if rec.action == "BUY_YES" else "BUY_YES"
                    if db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? LIMIT 1",
                        (city_key, event_date, rec.bracket_label, opposite_side),
                    ).fetchone():
                        continue

                    size = round(bet_size / rec.price, 2)
                    cost = round(rec.price * size, 2)
                    today_spent = db.conn.execute(
                        "SELECT COALESCE(SUM(cost), 0) FROM bets WHERE llm_source = 'gemini' AND date = ?",
                        (event_date,),
                    ).fetchone()[0]
                    if today_spent + cost > _gemini_cfg["max_daily_exposure"]:
                        break
                    order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                    db.conn.execute(
                        """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                           side, token_id, price, size, cost, order_id, order_status,
                           current_max_temp, peak_hour, model_confidence, expected_value,
                           is_paper, created_at, llm_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            rec.event_id, rec.bracket_db_id, city_key, event_date,
                            rec.bracket_label, rec.action, rec.token_id,
                            rec.price, size, cost,
                            order.order_id, order.status,
                            advice.peak_temp_estimate or 0.0,
                            peak_row["peak_hour"] if peak_row else 0,
                            rec.confidence, rec.confidence - rec.price,
                            int(order.is_paper), int(time.time()), "gemini",
                        ),
                    )
                    db.conn.commit()
                    executed_count += 1
                    if rec.action == "BUY_YES":
                        yes_placed_this_run = True
                    log.info(
                        "GEMINI_BOT_BET", city=city_key, bracket=rec.bracket_label,
                        action=rec.action, price=rec.price, size=f"${bet_size:.0f}",
                        conf=f"{rec.confidence:.0%}", edge=f"{rec.confidence - rec.price:.0%}",
                    )

                recs_json = _json.dumps([
                    {"bracket": r.bracket_label, "action": r.action,
                     "confidence": r.confidence, "price": r.price, "reasoning": r.reasoning}
                    for r in advice.recommendations
                ])
                db.conn.execute(
                    """INSERT INTO llm_analysis_log
                       (city_key, date, analysis_time, hour_local, peak_estimate, peak_confidence,
                        analysis_text, recs_json, bets_executed, model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (city_key, event_date, int(time.time()), current_hour,
                     advice.peak_temp_estimate, advice.peak_confidence,
                     advice.analysis, recs_json, executed_count, advice.model),
                )
                db.conn.commit()

                _gemini_bot_state["last_run_at"] = time.time()
                _gemini_bot_state["last_error"] = None
                log.info("gemini_bot_tick", city=city_key, event_date=event_date, bets=executed_count, hour=current_hour)

            except Exception as exc:
                _gemini_bot_state["last_error"] = str(exc)
                log.error("gemini_bot_tick_failed", city=city_key, error=str(exc))

    _scheduler.add_job(
        _run_gemini_auto_bet,
        "interval",
        minutes=1,
        seconds=45,
        id="gemini_auto_bet",
    )

    # ---- NO Agent Bot Loops ----
    async def _run_claude_no_bot():
        """Claude NO agent: BUY_NO on clearly excluded brackets for safe small gains."""
        from config import settings
        from src.weather.cities import get_city, CITIES
        from src.utils.time_utils import today_in_tz, current_hour_in_tz
        from src.weather import open_meteo, metar as metar_mod
        from src.strategy.llm_advisor import analyze as llm_analyze
        from src.polymarket.trader import PolymarketTrader
        import json as _json

        if not _claude_no_state["enabled"]:
            return
        if not settings.anthropic_api_key:
            return

        db = get_db()
        now = time.time()
        _COOLDOWN = 300

        active_keys: list[str] = []
        for ck, city_obj in CITIES.items():
            _today = today_in_tz(city_obj.timezone)
            row = db.conn.execute(
                "SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
                (ck, _today),
            ).fetchone()
            if row:
                active_keys.append(ck)
        _claude_no_state["city_keys"] = active_keys
        if not active_keys:
            return

        for city_key in active_keys:
            if now - _claude_no_state["city_last_run"].get(city_key, 0) < _COOLDOWN:
                continue
            _claude_no_state["city_last_run"][city_key] = now
            _claude_no_state["last_city"] = city_key

            city = get_city(city_key)
            if not city:
                continue

            today = today_in_tz(city.timezone)
            current_hour = current_hour_in_tz(city.timezone)
            if current_hour < 7:
                continue

            try:
                brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
                if not brackets:
                    continue

                wu_rows = db.conn.execute(
                    "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
                    "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
                    (city_key, event_date),
                ).fetchall()
                wu_obs = [dict(r) for r in wu_rows]

                peak_row = db.conn.execute(
                    "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
                    (city_key, event_date),
                ).fetchone()
                summary_high = peak_row["summary_high_temp"] if peak_row and peak_row["summary_high_temp"] else (
                    peak_row["peak_temp"] if peak_row else None
                )
                if peak_row and peak_row["peak_hour"] < 8:
                    continue

                wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])
                om_hourly: list[dict] = []
                try:
                    om_data = await open_meteo.fetch(city)
                    om_hourly = [
                        {"hour": h["hour"], "temperature": h["temperature"], "is_forecast": h.get("is_forecast", False)}
                        for h in om_data.get("hourly", []) if h.get("date") == event_date
                    ]
                except Exception:
                    pass

                metar_obs_list: list[dict] = []
                try:
                    metar_data = await metar_mod.fetch(city, hours=24)
                    metar_obs_list = [
                        {"hour": o["hour"], "temperature": o["temperature"]}
                        for o in metar_data.get("observations", []) if o.get("date") == event_date
                    ]
                except Exception:
                    pass

                historical_context = _fetch_historical_context(db, city_key)
                min_hour = _min_bet_hour(db, city_key)
                _claude_no_state["last_min_bet_hour"] = min_hour
                _claude_no_state["last_window"] = f"{min_hour:02d}:00+"
                if current_hour < min_hour:
                    continue

                advice = await llm_analyze(
                    api_key=settings.anthropic_api_key,
                    city_name=city.display_name, unit=city.unit, today=event_date,
                    current_local_hour=current_hour, wu_obs=wu_obs, wu_forecast=wu_forecast,
                    om_hourly=om_hourly, metar_obs=metar_obs_list, summary_high=summary_high,
                    brackets=brackets, historical_context=historical_context, min_bet_hour=min_hour,
                )

                executed_count = 0
                trader = PolymarketTrader(settings)
                _no_cfg = _load_bot_settings(db.conn, "claude-no")
                _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
                for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                    if rec.action != "BUY_NO":
                        continue  # NO agent: only BUY_NO
                    if not rec.token_id or rec.price <= 0.01:
                        continue
                    if rec.bracket_label == _market_leader:
                        continue  # never bet NO on market leader
                    if rec.confidence < _no_cfg["min_confidence"]:
                        continue
                    bet_size = _no_bet_size(rec.confidence, rec.price)
                    if bet_size == 0.0:
                        continue
                    bet_size = min(bet_size, _no_cfg["max_bet_usd"])
                    existing = db.conn.execute(
                        "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? AND llm_source = 'claude-no' LIMIT 1",
                        (city_key, event_date, rec.bracket_label, rec.action),
                    ).fetchone()
                    if existing:
                        continue
                    size = round(bet_size / rec.price, 2)
                    cost = round(rec.price * size, 2)
                    today_spent = db.conn.execute(
                        "SELECT COALESCE(SUM(cost), 0) FROM bets WHERE llm_source = 'claude-no' AND date = ?",
                        (event_date,),
                    ).fetchone()[0]
                    if today_spent + cost > _no_cfg["max_daily_exposure"]:
                        break
                    order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                    db.conn.execute(
                        """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                           side, token_id, price, size, cost, order_id, order_status,
                           current_max_temp, peak_hour, model_confidence, expected_value,
                           is_paper, created_at, llm_source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (rec.event_id, rec.bracket_db_id, city_key, event_date,
                         rec.bracket_label, rec.action, rec.token_id,
                         rec.price, size, cost, order.order_id, order.status,
                         advice.peak_temp_estimate or 0.0,
                         peak_row["peak_hour"] if peak_row else 0,
                         rec.confidence, rec.confidence - rec.price,
                         int(order.is_paper), int(time.time()), "claude-no"),
                    )
                    db.conn.commit()
                    executed_count += 1
                    log.info("CLAUDE_NO_BET", city=city_key, bracket=rec.bracket_label,
                             price=rec.price, conf=f"{rec.confidence:.0%}")

                _claude_no_state["last_run_at"] = time.time()
                _claude_no_state["last_error"] = None

            except Exception as exc:
                _claude_no_state["last_error"] = str(exc)
                log.error("claude_no_tick_failed", city=city_key, error=str(exc))

    async def _run_gpt_no_bot():
        """GPT NO agent: BUY_NO on clearly excluded brackets."""
        from config import settings
        from src.weather.cities import get_city, CITIES
        from src.utils.time_utils import today_in_tz, current_hour_in_tz
        from src.weather import open_meteo, metar as metar_mod
        from src.strategy.openai_advisor import analyze as gpt_analyze
        from src.polymarket.trader import PolymarketTrader

        if not _gpt_no_state["enabled"] or not settings.openai_api_key:
            return
        db = get_db()
        now = time.time()
        _COOLDOWN = 300
        active_keys = [ck for ck, c in CITIES.items()
                       if db.conn.execute("SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
                                          (ck, today_in_tz(c.timezone))).fetchone()]
        _gpt_no_state["city_keys"] = active_keys
        if not active_keys:
            return
        for city_key in active_keys:
            if now - _gpt_no_state["city_last_run"].get(city_key, 0) < _COOLDOWN:
                continue
            _gpt_no_state["city_last_run"][city_key] = now
            _gpt_no_state["last_city"] = city_key
            city = get_city(city_key)
            if not city:
                continue
            today = today_in_tz(city.timezone)
            current_hour = current_hour_in_tz(city.timezone)
            if current_hour < 7:
                continue
            try:
                brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
                if not brackets:
                    continue
                wu_rows = db.conn.execute(
                    "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
                    "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
                    (city_key, event_date)).fetchall()
                wu_obs = [dict(r) for r in wu_rows]
                peak_row = db.conn.execute(
                    "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
                    (city_key, event_date)).fetchone()
                summary_high = peak_row["summary_high_temp"] if peak_row and peak_row["summary_high_temp"] else (
                    peak_row["peak_temp"] if peak_row else None)
                if peak_row and peak_row["peak_hour"] < 8:
                    continue
                wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])
                om_hourly: list[dict] = []
                try:
                    om_data = await open_meteo.fetch(city)
                    om_hourly = om_data.get("hourly", [])
                except Exception:
                    pass
                metar_obs_list: list[dict] = []
                try:
                    metar_data = await metar_mod.fetch(city, hours=24)
                    metar_obs_list = [{"hour": o["hour"], "temperature": o["temperature"]}
                                     for o in metar_data.get("observations", []) if o.get("date") == event_date]
                except Exception:
                    pass
                historical_context = _fetch_historical_context(db, city_key)
                min_hour = _min_bet_hour(db, city_key)
                _gpt_no_state["last_min_bet_hour"] = min_hour
                _gpt_no_state["last_window"] = f"{min_hour:02d}:00+"
                if current_hour < min_hour:
                    continue
                advice = await gpt_analyze(
                    api_key=settings.openai_api_key, city_name=city.display_name, unit=city.unit,
                    today=event_date, current_local_hour=current_hour, wu_obs=wu_obs, wu_forecast=wu_forecast,
                    om_hourly=om_hourly, metar_obs=metar_obs_list, summary_high=summary_high,
                    brackets=brackets, historical_context=historical_context, min_bet_hour=min_hour)
                executed_count = 0
                trader = PolymarketTrader(settings)
                _no_cfg = _load_bot_settings(db.conn, "gpt-no")
                _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
                for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                    if rec.action != "BUY_NO" or not rec.token_id or rec.price <= 0.01:
                        continue
                    if rec.bracket_label == _market_leader:
                        continue
                    if rec.confidence < _no_cfg["min_confidence"]:
                        continue
                    bet_size = _no_bet_size(rec.confidence, rec.price)
                    if bet_size == 0.0:
                        continue
                    bet_size = min(bet_size, _no_cfg["max_bet_usd"])
                    if db.conn.execute("SELECT 1 FROM bets WHERE city_key=? AND date=? AND bracket_label=? AND side=? AND llm_source='gpt-no' LIMIT 1",
                                       (city_key, event_date, rec.bracket_label, rec.action)).fetchone():
                        continue
                    size = round(bet_size / rec.price, 2)
                    cost = round(rec.price * size, 2)
                    today_spent = db.conn.execute("SELECT COALESCE(SUM(cost),0) FROM bets WHERE llm_source='gpt-no' AND date=?", (event_date,)).fetchone()[0]
                    if today_spent + cost > _no_cfg["max_daily_exposure"]:
                        break
                    order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                    db.conn.execute(
                        """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                           side, token_id, price, size, cost, order_id, order_status,
                           current_max_temp, peak_hour, model_confidence, expected_value,
                           is_paper, created_at, llm_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (rec.event_id, rec.bracket_db_id, city_key, event_date, rec.bracket_label,
                         rec.action, rec.token_id, rec.price, size, cost, order.order_id, order.status,
                         advice.peak_temp_estimate or 0.0, peak_row["peak_hour"] if peak_row else 0,
                         rec.confidence, rec.confidence - rec.price, int(order.is_paper), int(time.time()), "gpt-no"))
                    db.conn.commit()
                    executed_count += 1
                    log.info("GPT_NO_BET", city=city_key, bracket=rec.bracket_label, price=rec.price, conf=f"{rec.confidence:.0%}")
                _gpt_no_state["last_run_at"] = time.time()
                _gpt_no_state["last_error"] = None
            except Exception as exc:
                _gpt_no_state["last_error"] = str(exc)
                log.error("gpt_no_tick_failed", city=city_key, error=str(exc))

    async def _run_gemini_no_bot():
        """Gemini NO agent: BUY_NO on clearly excluded brackets."""
        from config import settings
        from src.weather.cities import get_city, CITIES
        from src.utils.time_utils import today_in_tz, current_hour_in_tz
        from src.weather import open_meteo, metar as metar_mod
        from src.strategy.gemini_advisor import analyze as gemini_analyze
        from src.polymarket.trader import PolymarketTrader

        if not _gemini_no_state["enabled"] or not settings.gemini_api_key:
            return
        db = get_db()
        now = time.time()
        _COOLDOWN = 300
        active_keys = [ck for ck, c in CITIES.items()
                       if db.conn.execute("SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
                                          (ck, today_in_tz(c.timezone))).fetchone()]
        _gemini_no_state["city_keys"] = active_keys
        if not active_keys:
            return
        for city_key in active_keys:
            if now - _gemini_no_state["city_last_run"].get(city_key, 0) < _COOLDOWN:
                continue
            _gemini_no_state["city_last_run"][city_key] = now
            _gemini_no_state["last_city"] = city_key
            city = get_city(city_key)
            if not city:
                continue
            today = today_in_tz(city.timezone)
            current_hour = current_hour_in_tz(city.timezone)
            if current_hour < 7:
                continue
            try:
                brackets, event_date = _get_best_active_event_brackets(db, city_key, today)
                if not brackets:
                    continue
                wu_rows = db.conn.execute(
                    "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
                    "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
                    (city_key, event_date)).fetchall()
                wu_obs = [dict(r) for r in wu_rows]
                peak_row = db.conn.execute(
                    "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
                    (city_key, event_date)).fetchone()
                summary_high = peak_row["summary_high_temp"] if peak_row and peak_row["summary_high_temp"] else (
                    peak_row["peak_temp"] if peak_row else None)
                if peak_row and peak_row["peak_hour"] < 8:
                    continue
                wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])
                om_hourly: list[dict] = []
                try:
                    om_data = await open_meteo.fetch(city)
                    om_hourly = om_data.get("hourly", [])
                except Exception:
                    pass
                metar_obs_list: list[dict] = []
                try:
                    metar_data = await metar_mod.fetch(city, hours=24)
                    metar_obs_list = [{"hour": o["hour"], "temperature": o["temperature"]}
                                     for o in metar_data.get("observations", []) if o.get("date") == event_date]
                except Exception:
                    pass
                historical_context = _fetch_historical_context(db, city_key)
                min_hour = _min_bet_hour(db, city_key)
                _gemini_no_state["last_min_bet_hour"] = min_hour
                _gemini_no_state["last_window"] = f"{min_hour:02d}:00+"
                if current_hour < min_hour:
                    continue
                advice = await gemini_analyze(
                    api_key=settings.gemini_api_key, city_name=city.display_name, unit=city.unit,
                    today=event_date, current_local_hour=current_hour, wu_obs=wu_obs, wu_forecast=wu_forecast,
                    om_hourly=om_hourly, metar_obs=metar_obs_list, summary_high=summary_high,
                    brackets=brackets, historical_context=historical_context, min_bet_hour=min_hour)
                executed_count = 0
                trader = PolymarketTrader(settings)
                _no_cfg = _load_bot_settings(db.conn, "gemini-no")
                _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""
                for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                    if rec.action != "BUY_NO" or not rec.token_id or rec.price <= 0.01:
                        continue
                    if rec.bracket_label == _market_leader:
                        continue
                    if rec.confidence < _no_cfg["min_confidence"]:
                        continue
                    bet_size = _no_bet_size(rec.confidence, rec.price)
                    if bet_size == 0.0:
                        continue
                    bet_size = min(bet_size, _no_cfg["max_bet_usd"])
                    if db.conn.execute("SELECT 1 FROM bets WHERE city_key=? AND date=? AND bracket_label=? AND side=? AND llm_source='gemini-no' LIMIT 1",
                                       (city_key, event_date, rec.bracket_label, rec.action)).fetchone():
                        continue
                    size = round(bet_size / rec.price, 2)
                    cost = round(rec.price * size, 2)
                    today_spent = db.conn.execute("SELECT COALESCE(SUM(cost),0) FROM bets WHERE llm_source='gemini-no' AND date=?", (event_date,)).fetchone()[0]
                    if today_spent + cost > _no_cfg["max_daily_exposure"]:
                        break
                    order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                    db.conn.execute(
                        """INSERT INTO bets (event_id, bracket_id, city_key, date, bracket_label,
                           side, token_id, price, size, cost, order_id, order_status,
                           current_max_temp, peak_hour, model_confidence, expected_value,
                           is_paper, created_at, llm_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (rec.event_id, rec.bracket_db_id, city_key, event_date, rec.bracket_label,
                         rec.action, rec.token_id, rec.price, size, cost, order.order_id, order.status,
                         advice.peak_temp_estimate or 0.0, peak_row["peak_hour"] if peak_row else 0,
                         rec.confidence, rec.confidence - rec.price, int(order.is_paper), int(time.time()), "gemini-no"))
                    db.conn.commit()
                    executed_count += 1
                    log.info("GEMINI_NO_BET", city=city_key, bracket=rec.bracket_label, price=rec.price, conf=f"{rec.confidence:.0%}")
                _gemini_no_state["last_run_at"] = time.time()
                _gemini_no_state["last_error"] = None
            except Exception as exc:
                _gemini_no_state["last_error"] = str(exc)
                log.error("gemini_no_tick_failed", city=city_key, error=str(exc))

    _scheduler.add_job(_run_claude_no_bot, "interval", minutes=1, seconds=10, id="claude_no_bot")
    _scheduler.add_job(_run_gpt_no_bot, "interval", minutes=1, seconds=25, id="gpt_no_bot")
    _scheduler.add_job(_run_gemini_no_bot, "interval", minutes=1, seconds=50, id="gemini_no_bot")

    _scheduler.start()
    log.info("scheduler_started")

    # Trigger an initial check 30 seconds after startup
    asyncio.create_task(_delayed_resolution_check(30))
    # Trigger initial WU forecast scrape 60 seconds after startup
    asyncio.create_task(_delayed_wu_forecast(60))
    # Trigger initial market poll + bet resolution 90 seconds after startup
    asyncio.create_task(_delayed_market_poll(90))


async def _delayed_resolution_check(delay_seconds: int) -> None:
    await asyncio.sleep(delay_seconds)
    db = get_db()
    from src.polymarket.resolution_url_checker import check_all_cities
    try:
        await check_all_cities(db)
    except Exception as exc:
        log.error("startup_resolution_check_failed", error=str(exc))


async def _delayed_market_poll(delay_seconds: int) -> None:
    """Run market poll + bet resolution once after startup delay."""
    await asyncio.sleep(delay_seconds)
    db = get_db()
    market_repo = MarketRepository(db)
    from src.polymarket.market_monitor import MarketMonitor
    monitor = MarketMonitor(market_repo)
    try:
        count = await monitor.poll_all()
        log.info("startup_market_poll_done", events=count)
    except Exception as exc:
        log.error("startup_market_poll_failed", error=str(exc))


async def _delayed_wu_forecast(delay_seconds: int) -> None:
    """Run the WU forecast job once after a delay (initial warm-up)."""
    await asyncio.sleep(delay_seconds)
    import dataclasses as _dc
    from src.weather.cities import CITIES
    from src.weather.scraper import WUScraper
    from src.polymarket.resolution_url_checker import get_effective_wu_url_path

    db = get_db()
    log.info("wu_forecast_initial_start")
    scraper = WUScraper(headless=True)
    await scraper.start()
    try:
        for city in CITIES.values():
            effective_path = get_effective_wu_url_path(city, db)
            if effective_path != city.wu_url_path:
                city = _dc.replace(city, wu_url_path=effective_path)
            try:
                data = await scraper.scrape_forecast_hourly(city)
                _wu_forecast_cache[city.key] = {
                    "fetched_at": time.time(),
                    "data": data,
                    "unit": city.unit,
                }
            except Exception as exc:
                log.warning("wu_forecast_initial_city_failed", city=city.key, error=str(exc))
    finally:
        await scraper.stop()
    log.info("wu_forecast_initial_done", cities=len(CITIES))


@app.get("/ai-betting")
async def ai_betting_page(request: Request):
    return templates.TemplateResponse(request, "ai_betting.html")


@app.get("/api/ai-bot/status")
async def api_ai_bot_status():
    from src.weather.cities import CITIES
    db = get_db()
    # Count today's runs and bets from DB (survives restarts)
    today_counts = {}
    for ck, city in CITIES.items():
        from src.utils.time_utils import today_in_tz
        td = today_in_tz(city.timezone)
        today_counts[ck] = td
    # Use a rough today (UTC date) for DB counts
    import datetime
    today_utc = datetime.date.today().isoformat()
    runs_today = db.conn.execute(
        "SELECT COUNT(*) FROM llm_analysis_log WHERE date >= ?", (today_utc,)
    ).fetchone()[0]
    bets_today = db.conn.execute(
        "SELECT COUNT(*) FROM bets WHERE date >= ? AND is_paper = 1", (today_utc,)
    ).fetchone()[0]
    return {
        "enabled": _ai_bot_state["enabled"],
        "active_city_count": len(_ai_bot_state["city_keys"]),
        "active_cities": _ai_bot_state["city_keys"],
        "last_run_at": _ai_bot_state["last_run_at"],
        "last_city": _ai_bot_state["last_city"],
        "last_error": _ai_bot_state["last_error"],
        "last_min_bet_hour": _ai_bot_state["last_min_bet_hour"],
        "last_window": _ai_bot_state["last_window"],
        "runs_today": runs_today,
        "bets_today": bets_today,
    }


@app.post("/api/ai-bot/toggle")
async def api_ai_bot_toggle(enabled: bool):
    _ai_bot_state["enabled"] = enabled
    if enabled:
        # Reset city list and cooldowns so it re-evaluates all cities immediately
        _ai_bot_state["city_keys"] = []
        _ai_bot_state["city_last_run"] = {}
    log.info("ai_bot_toggled", enabled=enabled)
    return {"enabled": _ai_bot_state["enabled"]}


@app.get("/api/ai-bot/log")
async def api_ai_bot_log(limit: int = 50):
    import json as _json
    db = get_db()
    rows = db.conn.execute(
        """SELECT id, city_key, date, analysis_time, hour_local, peak_estimate,
                  peak_confidence, analysis_text, recs_json, bets_executed, model
           FROM llm_analysis_log ORDER BY analysis_time DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    result = []
    for r in rows:
        entry = dict(r)
        try:
            entry["recommendations"] = _json.loads(r["recs_json"] or "[]")
        except Exception:
            entry["recommendations"] = []
        del entry["recs_json"]
        # Attach actual bets placed within 2 minutes of this analysis
        bet_rows = db.conn.execute(
            """SELECT bracket_label, side, size, price FROM bets
               WHERE city_key = ? AND date = ?
                 AND created_at BETWEEN ? AND ?
               ORDER BY id""",
            (r["city_key"], r["date"],
             r["analysis_time"] - 120, r["analysis_time"] + 120),
        ).fetchall()
        entry["placed_bets"] = [dict(b) for b in bet_rows]

        # Attach current live bracket prices from polymarket_brackets
        mkt_rows = db.conn.execute(
            """SELECT pb.bracket_label, pb.current_yes_price
               FROM polymarket_brackets pb
               JOIN polymarket_events pe ON pb.event_id = pe.event_id
               WHERE pe.city_key = ? AND pe.date = ?
               ORDER BY pb.bracket_label""",
            (r["city_key"], r["date"]),
        ).fetchall()
        entry["market_brackets"] = [
            {"label": m["bracket_label"], "yes_price": m["current_yes_price"] or 0.0}
            for m in mkt_rows
        ]
        result.append(entry)
    return {"log": result}


@app.get("/gpt-betting")
async def gpt_betting_page(request: Request):
    return templates.TemplateResponse(request, "gpt_betting.html")


@app.get("/api/gpt-bot/status")
async def api_gpt_bot_status():
    db = get_db()
    import datetime
    today_utc = datetime.date.today().isoformat()
    runs_today = db.conn.execute(
        "SELECT COUNT(*) FROM llm_analysis_log WHERE date >= ? AND model LIKE 'gpt%'", (today_utc,)
    ).fetchone()[0]
    bets_today = db.conn.execute(
        "SELECT COUNT(*) FROM bets WHERE date >= ? AND llm_source = 'gpt'", (today_utc,)
    ).fetchone()[0]
    return {
        "enabled": _gpt_bot_state["enabled"],
        "active_city_count": len(_gpt_bot_state["city_keys"]),
        "active_cities": _gpt_bot_state["city_keys"],
        "last_run_at": _gpt_bot_state["last_run_at"],
        "last_city": _gpt_bot_state["last_city"],
        "last_error": _gpt_bot_state["last_error"],
        "last_min_bet_hour": _gpt_bot_state["last_min_bet_hour"],
        "last_window": _gpt_bot_state["last_window"],
        "runs_today": runs_today,
        "bets_today": bets_today,
    }


@app.post("/api/gpt-bot/toggle")
async def api_gpt_bot_toggle(enabled: bool):
    _gpt_bot_state["enabled"] = enabled
    if enabled:
        _gpt_bot_state["city_keys"] = []
        _gpt_bot_state["city_last_run"] = {}
    log.info("gpt_bot_toggled", enabled=enabled)
    return {"enabled": _gpt_bot_state["enabled"]}


@app.get("/api/gpt-bot/log")
async def api_gpt_bot_log(limit: int = 50):
    import json as _json
    db = get_db()
    rows = db.conn.execute(
        """SELECT id, city_key, date, analysis_time, hour_local, peak_estimate,
                  peak_confidence, analysis_text, recs_json, bets_executed, model
           FROM llm_analysis_log WHERE model LIKE 'gpt%'
           ORDER BY analysis_time DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    result = []
    for r in rows:
        entry = dict(r)
        try:
            entry["recommendations"] = _json.loads(r["recs_json"] or "[]")
        except Exception:
            entry["recommendations"] = []
        del entry["recs_json"]
        bet_rows = db.conn.execute(
            """SELECT bracket_label, side, size, price FROM bets
               WHERE city_key = ? AND date = ? AND llm_source = 'gpt'
                 AND created_at BETWEEN ? AND ?
               ORDER BY id""",
            (r["city_key"], r["date"],
             r["analysis_time"] - 120, r["analysis_time"] + 120),
        ).fetchall()
        entry["placed_bets"] = [dict(b) for b in bet_rows]
        mkt_rows = db.conn.execute(
            """SELECT pb.bracket_label, pb.current_yes_price
               FROM polymarket_brackets pb
               JOIN polymarket_events pe ON pb.event_id = pe.event_id
               WHERE pe.city_key = ? AND pe.date = ?
               ORDER BY pb.bracket_label""",
            (r["city_key"], r["date"]),
        ).fetchall()
        entry["market_brackets"] = [
            {"label": m["bracket_label"], "yes_price": m["current_yes_price"] or 0.0}
            for m in mkt_rows
        ]
        result.append(entry)
    return {"log": result}


@app.get("/gemini-betting")
async def gemini_betting_page(request: Request):
    return templates.TemplateResponse(request, "gemini_betting.html")


@app.get("/api/gemini-bot/status")
async def api_gemini_bot_status():
    db = get_db()
    import datetime
    today_utc = datetime.date.today().isoformat()
    runs_today = db.conn.execute(
        "SELECT COUNT(*) FROM llm_analysis_log WHERE date >= ? AND model LIKE 'gemini%'", (today_utc,)
    ).fetchone()[0]
    bets_today = db.conn.execute(
        "SELECT COUNT(*) FROM bets WHERE date >= ? AND llm_source = 'gemini'", (today_utc,)
    ).fetchone()[0]
    return {
        "enabled": _gemini_bot_state["enabled"],
        "active_city_count": len(_gemini_bot_state["city_keys"]),
        "active_cities": _gemini_bot_state["city_keys"],
        "last_run_at": _gemini_bot_state["last_run_at"],
        "last_city": _gemini_bot_state["last_city"],
        "last_error": _gemini_bot_state["last_error"],
        "last_min_bet_hour": _gemini_bot_state["last_min_bet_hour"],
        "last_window": _gemini_bot_state["last_window"],
        "runs_today": runs_today,
        "bets_today": bets_today,
    }


@app.post("/api/gemini-bot/toggle")
async def api_gemini_bot_toggle(enabled: bool):
    _gemini_bot_state["enabled"] = enabled
    if enabled:
        _gemini_bot_state["city_keys"] = []
        _gemini_bot_state["city_last_run"] = {}
    log.info("gemini_bot_toggled", enabled=enabled)
    return {"enabled": _gemini_bot_state["enabled"]}


@app.get("/api/gemini-bot/log")
async def api_gemini_bot_log(limit: int = 50):
    import json as _json
    db = get_db()
    rows = db.conn.execute(
        """SELECT id, city_key, date, analysis_time, hour_local, peak_estimate,
                  peak_confidence, analysis_text, recs_json, bets_executed, model
           FROM llm_analysis_log WHERE model LIKE 'gemini%'
           ORDER BY analysis_time DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    result = []
    for r in rows:
        entry = dict(r)
        try:
            entry["recommendations"] = _json.loads(r["recs_json"] or "[]")
        except Exception:
            entry["recommendations"] = []
        del entry["recs_json"]
        bet_rows = db.conn.execute(
            """SELECT bracket_label, side, size, price FROM bets
               WHERE city_key = ? AND date = ? AND llm_source = 'gemini'
                 AND created_at BETWEEN ? AND ?
               ORDER BY id""",
            (r["city_key"], r["date"],
             r["analysis_time"] - 120, r["analysis_time"] + 120),
        ).fetchall()
        entry["placed_bets"] = [dict(b) for b in bet_rows]
        mkt_rows = db.conn.execute(
            """SELECT pb.bracket_label, pb.current_yes_price
               FROM polymarket_brackets pb
               JOIN polymarket_events pe ON pb.event_id = pe.event_id
               WHERE pe.city_key = ? AND pe.date = ?
               ORDER BY pb.bracket_label""",
            (r["city_key"], r["date"]),
        ).fetchall()
        entry["market_brackets"] = [
            {"label": m["bracket_label"], "yes_price": m["current_yes_price"] or 0.0}
            for m in mkt_rows
        ]
        result.append(entry)
    return {"log": result}


# ---- NO Bot Status/Toggle Endpoints ----
@app.get("/api/claude-no/status")
async def api_claude_no_status():
    return {
        "enabled": _claude_no_state["enabled"],
        "last_run_at": _claude_no_state["last_run_at"],
        "last_city": _claude_no_state["last_city"],
        "last_error": _claude_no_state["last_error"],
        "last_min_bet_hour": _claude_no_state["last_min_bet_hour"],
        "last_window": _claude_no_state["last_window"],
    }

@app.post("/api/claude-no/toggle")
async def api_claude_no_toggle(enabled: bool):
    _claude_no_state["enabled"] = enabled
    if enabled:
        _claude_no_state["city_last_run"] = {}
    return {"enabled": _claude_no_state["enabled"]}

@app.get("/api/gpt-no/status")
async def api_gpt_no_status():
    return {
        "enabled": _gpt_no_state["enabled"],
        "last_run_at": _gpt_no_state["last_run_at"],
        "last_city": _gpt_no_state["last_city"],
        "last_error": _gpt_no_state["last_error"],
        "last_min_bet_hour": _gpt_no_state["last_min_bet_hour"],
        "last_window": _gpt_no_state["last_window"],
    }

@app.post("/api/gpt-no/toggle")
async def api_gpt_no_toggle(enabled: bool):
    _gpt_no_state["enabled"] = enabled
    if enabled:
        _gpt_no_state["city_last_run"] = {}
    return {"enabled": _gpt_no_state["enabled"]}

@app.get("/api/gemini-no/status")
async def api_gemini_no_status():
    return {
        "enabled": _gemini_no_state["enabled"],
        "last_run_at": _gemini_no_state["last_run_at"],
        "last_city": _gemini_no_state["last_city"],
        "last_error": _gemini_no_state["last_error"],
        "last_min_bet_hour": _gemini_no_state["last_min_bet_hour"],
        "last_window": _gemini_no_state["last_window"],
    }

@app.post("/api/gemini-no/toggle")
async def api_gemini_no_toggle(enabled: bool):
    _gemini_no_state["enabled"] = enabled
    if enabled:
        _gemini_no_state["city_last_run"] = {}
    return {"enabled": _gemini_no_state["enabled"]}


@app.get("/pnl-comparison")
async def pnl_comparison_page(request: Request):
    return templates.TemplateResponse(request, "pnl_comparison.html")


@app.get("/api/pnl-comparison")
async def api_pnl_comparison():
    db = get_db()
    summary = db.conn.execute("""
        SELECT llm_source,
               COUNT(*) as total_bets,
               COALESCE(SUM(cost), 0) as total_cost,
               SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN won = 0 THEN 1 ELSE 0 END) as losses,
               SUM(CASE WHEN won IS NULL THEN 1 ELSE 0 END) as pending,
               COALESCE(SUM(pnl), 0) as total_pnl,
               COALESCE(AVG(CASE WHEN won IS NOT NULL THEN won END), 0) as win_rate
        FROM bets WHERE is_paper = 1
        GROUP BY llm_source
    """).fetchall()
    daily = db.conn.execute("""
        SELECT llm_source, date,
               COUNT(*) as bets,
               COALESCE(SUM(pnl), 0) as pnl,
               COALESCE(SUM(cost), 0) as cost
        FROM bets WHERE is_paper = 1
        GROUP BY llm_source, date ORDER BY date DESC
    """).fetchall()
    return {
        "summary": [dict(r) for r in summary],
        "daily": [dict(r) for r in daily],
    }


@app.on_event("shutdown")
async def shutdown():
    global _db, _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
    if _db:
        _db.close()
        _db = None
