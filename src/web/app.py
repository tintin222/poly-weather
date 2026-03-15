"""FastAPI dashboard for poly-weather paper trading results."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.db.database import Database
from src.utils.logger import get_logger

log = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="poly-weather", docs_url="/api/docs")

# Database singleton - initialized on startup
_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        from config import settings
        _db = Database(settings.db_path)
        _db.initialize()
    return _db


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


@app.on_event("shutdown")
async def shutdown():
    global _db
    if _db:
        _db.close()
        _db = None
