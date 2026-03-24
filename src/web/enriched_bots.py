"""Enriched weather bots — same LLMs as originals but with OpenWeather + WeatherAPI data.

Creates 6 new bots (claude+, gpt+, gemini+ YES and claude+-no, gpt+-no, gemini+-no NO)
that run alongside the original 6 bots. Zero modifications to existing bot code.
"""

from __future__ import annotations

import time
import json as _json
from typing import Callable, Any, Awaitable

from fastapi import APIRouter, HTTPException

from src.utils.logger import get_logger

log = get_logger(__name__)

enriched_router = APIRouter()

# ---------------------------------------------------------------------------
# State dicts — one per enriched bot
# ---------------------------------------------------------------------------

def _make_state() -> dict:
    return {
        "enabled": True,
        "city_keys": [],
        "city_last_run": {},
        "runs_today": 0,
        "bets_today": 0,
        "last_run_at": 0.0,
        "last_city": None,
        "last_error": None,
        "last_min_bet_hour": None,
        "last_window": None,
    }

_enriched_states: dict[str, dict] = {
    "claude+": {**_make_state(), "enabled": False},
    "gpt+": {**_make_state(), "enabled": False},
    "gemini+": _make_state(),
    "claude+-no": {**_make_state(), "enabled": False},
    "gpt+-no": {**_make_state(), "enabled": False},
    "gemini+-no": _make_state(),
}

_SLUG_TO_NAME = {
    "claude-plus": "claude+",
    "gpt-plus": "gpt+",
    "gemini-plus": "gemini+",
    "claude-plus-no": "claude+-no",
    "gpt-plus-no": "gpt+-no",
    "gemini-plus-no": "gemini+-no",
}

_COOLDOWN = 300  # 5 minutes per city


# ---------------------------------------------------------------------------
# Shared weather fetch helpers
# ---------------------------------------------------------------------------

async def _fetch_all_weather(city, city_key: str, event_date: str, db):
    """Fetch all 6 weather sources. Returns (wu_obs, summary_high, peak_row, wu_forecast, om_hourly, metar_obs, ow_data, wa_data)."""
    from src.weather import open_meteo, metar as metar_mod
    from src.weather import openweather, weatherapi_client
    from config import settings

    # WU observations
    wu_rows = db.conn.execute(
        "SELECT hour, temperature, humidity, dew_point, condition, wind_speed, wind_direction "
        "FROM hourly_observations WHERE city_key = ? AND date = ? ORDER BY hour",
        (city_key, event_date),
    ).fetchall()
    wu_obs = [dict(r) for r in wu_rows]

    # WU peak/summary
    peak_row = db.conn.execute(
        "SELECT peak_temp, peak_hour, summary_high_temp FROM daily_peaks WHERE city_key = ? AND date = ?",
        (city_key, event_date),
    ).fetchone()
    summary_high = peak_row["summary_high_temp"] if peak_row else None

    # WU forecast cache
    from src.web.app import _wu_forecast_cache
    wu_forecast = _wu_forecast_cache.get(city_key, {}).get("data", [])

    # Open-Meteo
    om_hourly: list[dict] = []
    try:
        om_data = await open_meteo.fetch(city)
        om_hourly = [
            {"hour": h["hour"], "temperature": h["temperature"], "is_forecast": h.get("is_forecast", False)}
            for h in om_data.get("hourly", []) if h.get("date") == event_date
        ]
    except Exception:
        pass

    # METAR
    metar_obs_list: list[dict] = []
    try:
        metar_data = await metar_mod.fetch(city, hours=24)
        metar_obs_list = [
            {"hour": o["hour"], "temperature": o["temperature"]}
            for o in metar_data.get("observations", []) if o.get("date") == event_date
        ]
    except Exception:
        pass

    # OpenWeather (new)
    ow_data: dict = {}
    try:
        ow_data = await openweather.fetch(city, settings.openweather_api_key)
    except Exception:
        pass

    # WeatherAPI (new)
    wa_data: dict = {}
    try:
        wa_data = await weatherapi_client.fetch(city, settings.weatherapi_api_key)
    except Exception:
        pass

    return wu_obs, summary_high, peak_row, wu_forecast, om_hourly, metar_obs_list, ow_data, wa_data


# ---------------------------------------------------------------------------
# Generic YES bot tick
# ---------------------------------------------------------------------------

async def _enriched_yes_tick(
    bot_name: str,
    llm_source: str,
    analyze_fn: Callable[..., Awaitable],
    state: dict,
    api_key_attr: str,  # e.g. "anthropic_api_key"
):
    """Generic enriched YES bot — same logic as original YES bots + extra weather data."""
    from config import settings
    from src.weather.cities import get_city, CITIES
    from src.utils.time_utils import today_in_tz, current_hour_in_tz
    from src.polymarket.trader import PolymarketTrader
    from src.web.app import (
        get_db, _get_best_active_event_brackets, _fetch_historical_context,
        _min_bet_hour, _tiered_bet_size, _load_bot_settings,
    )

    if not state["enabled"]:
        return
    api_key = getattr(settings, api_key_attr, "")
    if not api_key:
        return

    db = get_db()
    now = time.time()

    # Refresh active city list
    active_keys: list[str] = []
    for ck, city_obj in CITIES.items():
        _today = today_in_tz(city_obj.timezone)
        row = db.conn.execute(
            "SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
            (ck, _today),
        ).fetchone()
        if row:
            active_keys.append(ck)
    state["city_keys"] = active_keys
    if not active_keys:
        return

    for city_key in active_keys:
        if now - state["city_last_run"].get(city_key, 0) < _COOLDOWN:
            continue
        state["city_last_run"][city_key] = now
        state["last_city"] = city_key

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

            wu_obs, summary_high, peak_row, wu_forecast, om_hourly, metar_obs_list, ow_data, wa_data = \
                await _fetch_all_weather(city, city_key, event_date, db)

            if peak_row and peak_row["peak_hour"] < 8:
                continue

            historical_context = _fetch_historical_context(db, city_key)
            min_hour = _min_bet_hour(db, city_key)
            state["last_min_bet_hour"] = min_hour
            state["last_window"] = f"{min_hour:02d}:00+"
            if current_hour < min_hour:
                continue

            # Fetch current position for this enriched bot
            pos = db.conn.execute(
                "SELECT id, bracket_label, token_id, price, size, cost FROM bets "
                "WHERE city_key = ? AND date = ? AND side LIKE '%YES%' AND llm_source = ? AND won IS NULL LIMIT 1",
                (city_key, event_date, llm_source),
            ).fetchone()
            pos_dict = dict(pos) if pos else None

            advice = await analyze_fn(
                api_key=api_key,
                city_name=city.display_name, unit=city.unit, today=event_date,
                current_local_hour=current_hour, wu_obs=wu_obs, wu_forecast=wu_forecast,
                om_hourly=om_hourly, metar_obs=metar_obs_list, summary_high=summary_high,
                brackets=brackets, historical_context=historical_context, min_bet_hour=min_hour,
                current_position=pos_dict,
                openweather_data=ow_data, weatherapi_data=wa_data,
            )

            executed_count = 0
            yes_placed_this_run = False
            # Enriched bots always paper trade for A/B comparison
            paper_settings = settings.model_copy(update={"paper_mode": True})
            trader = PolymarketTrader(paper_settings)
            bot_cfg = _load_bot_settings(db.conn, bot_name)
            _bracket_prices = {b["label"]: b["yes_price"] for b in brackets}

            for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                if rec.action == "SKIP" or not rec.token_id:
                    continue
                if rec.price <= 0.01:
                    continue
                if rec.action != "BUY_YES":
                    continue
                if rec.price > 0.92 or rec.price < 0.08:
                    continue
                if yes_placed_this_run:
                    continue

                # Position switching
                if pos:
                    if pos["bracket_label"] == rec.bracket_label:
                        continue
                    old_sell_price = _bracket_prices.get(pos["bracket_label"], 0)
                    if old_sell_price > 0:
                        sell_order = trader.place_limit_sell(
                            token_id=pos["token_id"], price=old_sell_price, size=pos["size"])
                        if not sell_order.success:
                            log.warning("ENRICHED_SELL_FAILED", bot=bot_name, city=city_key,
                                        bracket=pos["bracket_label"], error=sell_order.error_msg)
                            continue
                        sell_pnl = round(old_sell_price * pos["size"] - pos["cost"], 2)
                        db.conn.execute(
                            "UPDATE bets SET won = 2, pnl = ?, resolved_at = ? WHERE id = ?",
                            (sell_pnl, int(time.time()), pos["id"]))
                        db.conn.commit()
                        log.info("ENRICHED_POSITION_SOLD", bot=bot_name, city=city_key,
                                 old_bracket=pos_dict["bracket_label"], new_bracket=rec.bracket_label, sell_pnl=sell_pnl)
                        pos = None
                    else:
                        continue

                if rec.confidence < bot_cfg["min_confidence"]:
                    continue
                bet_size = _tiered_bet_size(rec.confidence, rec.price)
                if bet_size == 0.0:
                    continue
                bet_size = min(bet_size, bot_cfg["max_bet_usd"])

                existing = db.conn.execute(
                    "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? AND llm_source = ? LIMIT 1",
                    (city_key, event_date, rec.bracket_label, rec.action, llm_source),
                ).fetchone()
                if existing:
                    continue

                # Cross-bot dedup
                opposite_side = "BUY_NO" if rec.action == "BUY_YES" else "BUY_YES"
                if db.conn.execute(
                    "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? LIMIT 1",
                    (city_key, event_date, rec.bracket_label, opposite_side),
                ).fetchone():
                    continue

                size = round(bet_size / rec.price, 2)
                cost = round(rec.price * size, 2)
                today_spent = db.conn.execute(
                    "SELECT COALESCE(SUM(cost), 0) FROM bets WHERE llm_source = ? AND date = ?",
                    (llm_source, event_date),
                ).fetchone()[0]
                if today_spent + cost > bot_cfg["max_daily_exposure"]:
                    break

                order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                if not order.success:
                    log.warning("ENRICHED_YES_ORDER_FAILED", bot=bot_name, city=city_key,
                                bracket=rec.bracket_label, error=order.error_msg)
                    continue
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
                     int(order.is_paper), int(time.time()), llm_source),
                )
                db.conn.commit()
                executed_count += 1
                if rec.action == "BUY_YES":
                    yes_placed_this_run = True
                log.info("ENRICHED_YES_BET", bot=bot_name, city=city_key, bracket=rec.bracket_label,
                         price=rec.price, size=f"${bet_size:.0f}", conf=f"{rec.confidence:.0%}")

            # Log analysis
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
                 advice.analysis, recs_json, executed_count, f"{advice.model}+"),
            )
            db.conn.commit()

            state["last_run_at"] = time.time()
            state["runs_today"] += 1
            state["bets_today"] += executed_count
            state["last_error"] = None
            log.info("enriched_yes_tick", bot=bot_name, city=city_key, event_date=event_date,
                     bets=executed_count, hour=current_hour)

        except Exception as exc:
            state["last_error"] = str(exc)
            log.error("enriched_yes_tick_failed", bot=bot_name, city=city_key, error=str(exc))


# ---------------------------------------------------------------------------
# Generic NO bot tick
# ---------------------------------------------------------------------------

async def _enriched_no_tick(
    bot_name: str,
    llm_source: str,
    analyze_fn: Callable[..., Awaitable],
    state: dict,
    api_key_attr: str,
):
    """Generic enriched NO bot — same logic as original NO bots + extra weather data."""
    from config import settings
    from src.weather.cities import get_city, CITIES
    from src.utils.time_utils import today_in_tz, current_hour_in_tz
    from src.polymarket.trader import PolymarketTrader
    from src.web.app import (
        get_db, _get_best_active_event_brackets, _fetch_historical_context,
        _min_bet_hour, _no_bet_size, _load_bot_settings,
    )

    if not state["enabled"]:
        return
    api_key = getattr(settings, api_key_attr, "")
    if not api_key:
        return

    db = get_db()
    now = time.time()

    active_keys: list[str] = []
    for ck, city_obj in CITIES.items():
        _today = today_in_tz(city_obj.timezone)
        row = db.conn.execute(
            "SELECT 1 FROM polymarket_events WHERE city_key = ? AND status = 'active' AND date = ? LIMIT 1",
            (ck, _today),
        ).fetchone()
        if row:
            active_keys.append(ck)
    state["city_keys"] = active_keys
    if not active_keys:
        return

    for city_key in active_keys:
        if now - state["city_last_run"].get(city_key, 0) < _COOLDOWN:
            continue
        state["city_last_run"][city_key] = now
        state["last_city"] = city_key

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

            wu_obs, summary_high, peak_row, wu_forecast, om_hourly, metar_obs_list, ow_data, wa_data = \
                await _fetch_all_weather(city, city_key, event_date, db)

            # Use peak_temp as fallback for summary_high (NO bot pattern)
            if not summary_high and peak_row:
                summary_high = peak_row["peak_temp"]
            if peak_row and peak_row["peak_hour"] < 8:
                continue

            historical_context = _fetch_historical_context(db, city_key)
            min_hour = _min_bet_hour(db, city_key)
            state["last_min_bet_hour"] = min_hour
            state["last_window"] = f"{min_hour:02d}:00+"
            if current_hour < min_hour:
                continue

            advice = await analyze_fn(
                api_key=api_key,
                city_name=city.display_name, unit=city.unit, today=event_date,
                current_local_hour=current_hour, wu_obs=wu_obs, wu_forecast=wu_forecast,
                om_hourly=om_hourly, metar_obs=metar_obs_list, summary_high=summary_high,
                brackets=brackets, historical_context=historical_context, min_bet_hour=min_hour,
                openweather_data=ow_data, weatherapi_data=wa_data,
            )

            executed_count = 0
            # Enriched bots always paper trade for A/B comparison
            paper_settings = settings.model_copy(update={"paper_mode": True})
            trader = PolymarketTrader(paper_settings)
            no_cfg = _load_bot_settings(db.conn, bot_name)
            _market_leader = max(brackets, key=lambda b: b["yes_price"])["label"] if brackets else ""

            for rec in sorted(advice.recommendations, key=lambda r: r.confidence, reverse=True):
                if rec.action != "BUY_NO":
                    continue
                if not rec.token_id or rec.price <= 0.01:
                    continue
                if rec.bracket_label == _market_leader:
                    continue
                if rec.confidence < no_cfg["min_confidence"]:
                    continue
                bet_size = _no_bet_size(rec.confidence, rec.price)
                if bet_size == 0.0:
                    continue
                bet_size = min(bet_size, no_cfg["max_bet_usd"])

                existing = db.conn.execute(
                    "SELECT 1 FROM bets WHERE city_key = ? AND date = ? AND bracket_label = ? AND side = ? AND llm_source = ? LIMIT 1",
                    (city_key, event_date, rec.bracket_label, rec.action, llm_source),
                ).fetchone()
                if existing:
                    continue

                size = round(bet_size / rec.price, 2)
                cost = round(rec.price * size, 2)
                today_spent = db.conn.execute(
                    "SELECT COALESCE(SUM(cost), 0) FROM bets WHERE llm_source = ? AND date = ?",
                    (llm_source, event_date),
                ).fetchone()[0]
                if today_spent + cost > no_cfg["max_daily_exposure"]:
                    break

                order = trader.place_limit_buy(token_id=rec.token_id, price=rec.price, size=size)
                if not order.success:
                    log.warning("ENRICHED_NO_ORDER_FAILED", bot=bot_name, city=city_key,
                                bracket=rec.bracket_label, error=order.error_msg)
                    continue
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
                     int(order.is_paper), int(time.time()), llm_source),
                )
                db.conn.commit()
                executed_count += 1
                log.info("ENRICHED_NO_BET", bot=bot_name, city=city_key, bracket=rec.bracket_label,
                         price=rec.price, conf=f"{rec.confidence:.0%}")

            state["last_run_at"] = time.time()
            state["last_error"] = None

        except Exception as exc:
            state["last_error"] = str(exc)
            log.error("enriched_no_tick_failed", bot=bot_name, city=city_key, error=str(exc))


# ---------------------------------------------------------------------------
# Concrete bot wrappers (thin — call the generic tick)
# ---------------------------------------------------------------------------

async def _run_claude_plus_yes():
    from config import settings
    from src.strategy.enriched_advisors import analyze_claude
    await _enriched_yes_tick("claude+", "claude+", analyze_claude, _enriched_states["claude+"], "anthropic_api_key")

async def _run_gpt_plus_yes():
    from config import settings
    from src.strategy.enriched_advisors import analyze_gpt
    await _enriched_yes_tick("gpt+", "gpt+", analyze_gpt, _enriched_states["gpt+"], "openai_api_key")

async def _run_gemini_plus_yes():
    from config import settings
    from src.strategy.enriched_advisors import analyze_gemini
    await _enriched_yes_tick("gemini+", "gemini+", analyze_gemini, _enriched_states["gemini+"], "gemini_api_key")

async def _run_claude_plus_no():
    from config import settings
    from src.strategy.enriched_advisors import analyze_claude
    await _enriched_no_tick("claude+-no", "claude+-no", analyze_claude, _enriched_states["claude+-no"], "anthropic_api_key")

async def _run_gpt_plus_no():
    from config import settings
    from src.strategy.enriched_advisors import analyze_gpt
    await _enriched_no_tick("gpt+-no", "gpt+-no", analyze_gpt, _enriched_states["gpt+-no"], "openai_api_key")

async def _run_gemini_plus_no():
    from config import settings
    from src.strategy.enriched_advisors import analyze_gemini
    await _enriched_no_tick("gemini+-no", "gemini+-no", analyze_gemini, _enriched_states["gemini+-no"], "gemini_api_key")


# ---------------------------------------------------------------------------
# Scheduler registration
# ---------------------------------------------------------------------------

def register_enriched_jobs(scheduler):
    """Register all 6 enriched bot jobs. Called from app.py startup."""
    scheduler.add_job(_run_claude_plus_yes, "interval", minutes=2, seconds=5, id="enriched_claude_plus_yes")
    scheduler.add_job(_run_claude_plus_no, "interval", minutes=2, seconds=15, id="enriched_claude_plus_no")
    scheduler.add_job(_run_gpt_plus_yes, "interval", minutes=2, seconds=25, id="enriched_gpt_plus_yes")
    scheduler.add_job(_run_gpt_plus_no, "interval", minutes=2, seconds=35, id="enriched_gpt_plus_no")
    scheduler.add_job(_run_gemini_plus_yes, "interval", minutes=2, seconds=45, id="enriched_gemini_plus_yes")
    scheduler.add_job(_run_gemini_plus_no, "interval", minutes=2, seconds=55, id="enriched_gemini_plus_no")
    log.info("enriched_bots_registered", count=6)


# ---------------------------------------------------------------------------
# API endpoints (generic for all 6 bots)
# ---------------------------------------------------------------------------

@enriched_router.get("/api/enriched-bots/status")
async def enriched_bots_status():
    """Return status for all 6 enriched bots."""
    out = {}
    for name, state in _enriched_states.items():
        out[name] = {
            "enabled": state["enabled"],
            "last_run_at": state["last_run_at"],
            "runs_today": state["runs_today"],
            "bets_today": state["bets_today"],
            "last_city": state["last_city"],
            "last_error": state["last_error"],
            "active_cities": len(state["city_keys"]),
            "last_min_bet_hour": state["last_min_bet_hour"],
            "last_window": state["last_window"],
        }
    return out


@enriched_router.post("/api/enriched-bots/{bot_slug}/toggle")
async def enriched_bot_toggle(bot_slug: str, enabled: bool = True):
    """Toggle an enriched bot on/off."""
    name = _SLUG_TO_NAME.get(bot_slug)
    if not name:
        raise HTTPException(404, detail=f"Unknown enriched bot: {bot_slug}")
    state = _enriched_states[name]
    state["enabled"] = enabled
    if enabled:
        state["city_keys"] = []
        state["city_last_run"] = {}
    log.info("enriched_bot_toggled", bot=name, enabled=enabled)
    return {"bot": name, "enabled": enabled}


@enriched_router.get("/api/enriched-bots/settings")
async def enriched_bots_settings():
    """Return bot_settings for all enriched bots."""
    from src.web.app import get_db, _load_bot_settings
    db = get_db()
    return {name: _load_bot_settings(db.conn, name) for name in _enriched_states}
