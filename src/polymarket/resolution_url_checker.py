"""Polymarket resolution URL tracker for Weather Underground station monitoring.

Scrapes the official WU resolution URL from each Polymarket temperature event
and stores it in city_resolution_urls, detecting station changes over time.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import httpx

from src.polymarket.gamma_api import get_temperature_events
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.db.database import Database
    from src.weather.cities import CityConfig

log = get_logger(__name__)

_WU_URL_RE = re.compile(
    r'https?://www\.wunderground\.com/history/daily/([\w\-\./%]+?)(?:/date/|\s|"|\}|$)'
)

# Cities whose Polymarket slug fragment differs from display_name.lower().replace(" ", "-")
_CITY_SLUG_OVERRIDES: dict[str, str] = {
    "NYC": "nyc",
}


def _city_to_slug(city: "CityConfig") -> str:
    """Convert city to Polymarket slug fragment."""
    return _CITY_SLUG_OVERRIDES.get(city.key, city.display_name.lower().replace(" ", "-"))


def _extract_wu_url(text: str) -> tuple[str, str] | None:
    """Extract WU full URL and path from text. Returns (full_url, path) or None."""
    match = _WU_URL_RE.search(text)
    if not match:
        return None
    # Strip trailing punctuation that may be captured from surrounding description text
    path = match.group(1).rstrip("/.,'\")")
    full_url = f"https://www.wunderground.com/history/daily/{path}"
    return full_url, path


async def _fallback_fetch_from_polymarket(slug: str) -> tuple[str, str] | None:
    """Fetch Polymarket event page and extract WU URL from __NEXT_DATA__ JSON."""
    import json

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(
                f"https://polymarket.com/event/{slug}",
                headers={"User-Agent": "Mozilla/5.0 (compatible; poly-weather/1.0)"},
            )
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        log.warning("fallback_fetch_error", slug=slug, error=str(exc))
        return None

    # Try __NEXT_DATA__ JSON block first
    next_data_match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            result = _extract_wu_url(json.dumps(data))
            if result:
                return result
        except (json.JSONDecodeError, Exception):
            pass

    # Fall back to raw HTML scan
    return _extract_wu_url(html)


async def check_city(city: "CityConfig", db: "Database") -> dict:
    """Fetch the Polymarket resolution URL for a city, upsert in DB.

    Returns a status dict with keys: city_key, status, wu_url, wu_url_path,
    changed, source_slug, error (on failure).
    """
    city_slug_prefix = f"highest-temperature-in-{_city_to_slug(city)}-on-"

    wu_url: str | None = None
    wu_url_path: str | None = None
    source_slug: str | None = None

    # Primary path: search Gamma API for the city's temperature events
    try:
        events = await get_temperature_events(active_only=False, limit=200)
        matching = [e for e in events if e.slug.startswith(city_slug_prefix)]

        if matching:
            # Sort by end_date descending to get most recent event
            def _event_key(e):
                return e.end_date or e.closed_time or e.start_date or ""

            matching.sort(key=_event_key, reverse=True)
            latest = matching[0]
            source_slug = latest.slug

            if latest.description:
                result = _extract_wu_url(latest.description)
                if result:
                    wu_url, wu_url_path = result
    except Exception as exc:
        log.warning("gamma_api_error_for_city", city=city.key, error=str(exc))

    # Fallback path: fetch polymarket.com event page directly
    if not wu_url_path:
        # Try with the most recent date if we found a slug prefix match earlier
        if source_slug:
            result = await _fallback_fetch_from_polymarket(source_slug)
            if result:
                wu_url, wu_url_path = result
        else:
            # Construct slug for today and try
            from datetime import datetime
            now = datetime.utcnow()
            month_name = now.strftime("%B").lower()
            day = now.day
            year = now.year
            slug = f"{city_slug_prefix}{month_name}-{day}-{year}"
            result = await _fallback_fetch_from_polymarket(slug)
            if result:
                wu_url, wu_url_path = result
                source_slug = slug

    if not wu_url_path:
        log.warning(
            "resolution_url_not_found",
            city=city.key,
            slug_prefix=city_slug_prefix,
        )
        return {"city_key": city.key, "status": "not_found"}

    # Upsert into DB
    now_ts = int(time.time())
    existing = db.conn.execute(
        "SELECT wu_url_path, check_count FROM city_resolution_urls WHERE city_key = ?",
        (city.key,),
    ).fetchone()

    changed = False
    if existing is None:
        db.conn.execute(
            """INSERT INTO city_resolution_urls
               (city_key, wu_url, wu_url_path, last_checked_at, check_count, source_slug)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (city.key, wu_url, wu_url_path, now_ts, source_slug),
        )
    elif existing["wu_url_path"] and existing["wu_url_path"] != wu_url_path:
        changed = True
        db.conn.execute(
            """UPDATE city_resolution_urls
               SET wu_url = ?, wu_url_path = ?, previous_url = wu_url_path,
                   changed_at = ?, last_checked_at = ?,
                   check_count = check_count + 1, source_slug = ?
               WHERE city_key = ?""",
            (wu_url, wu_url_path, now_ts, now_ts, source_slug, city.key),
        )
    else:
        db.conn.execute(
            """UPDATE city_resolution_urls
               SET wu_url = ?, wu_url_path = ?, last_checked_at = ?,
                   check_count = check_count + 1, source_slug = ?
               WHERE city_key = ?""",
            (wu_url, wu_url_path, now_ts, source_slug, city.key),
        )
    db.conn.commit()

    log.info(
        "resolution_url_checked",
        city=city.key,
        wu_url_path=wu_url_path,
        changed=changed,
        source_slug=source_slug,
    )

    return {
        "city_key": city.key,
        "wu_url": wu_url,
        "wu_url_path": wu_url_path,
        "changed": changed,
        "source_slug": source_slug,
        "status": "changed" if changed else "ok",
    }


async def check_all_cities(db: "Database") -> list[dict]:
    """Check resolution URLs for all configured cities."""
    from src.weather.cities import CITIES

    results = []
    for city in CITIES.values():
        result = await check_city(city, db)
        results.append(result)

    ok_count = sum(1 for r in results if r.get("status") in ("ok", "changed"))
    changed_count = sum(1 for r in results if r.get("status") == "changed")
    log.info(
        "resolution_url_check_complete",
        total=len(results),
        ok=ok_count,
        changed=changed_count,
    )
    return results


def get_effective_wu_url_path(city: "CityConfig", db: "Database") -> str:
    """Return DB-stored wu_url_path if available, else static config value."""
    row = db.conn.execute(
        "SELECT wu_url_path FROM city_resolution_urls"
        " WHERE city_key = ? AND wu_url_path IS NOT NULL",
        (city.key,),
    ).fetchone()
    return row["wu_url_path"] if row else city.wu_url_path
