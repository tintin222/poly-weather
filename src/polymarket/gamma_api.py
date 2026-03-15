"""Gamma API client for discovering Polymarket temperature events.

Ported from copy-cat/src/api/gamma-api.ts.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx

from src.polymarket.models import BracketInfo, GammaEvent, GammaMarket
from src.utils.logger import get_logger
from src.utils.rate_limiter import gamma_api_limiter

log = get_logger(__name__)

BASE_URL = "https://gamma-api.polymarket.com"

def _new_client() -> httpx.AsyncClient:
    """Create a fresh HTTP client (avoids event loop binding issues)."""
    return httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30.0,
        headers={"Accept": "application/json"},
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def get_temperature_events(
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[GammaEvent]:
    """Fetch temperature market events from the Gamma API.

    Uses the ``tag_slug=temperature`` filter which is what Polymarket
    uses to tag all temperature-related events.
    """
    await gamma_api_limiter.acquire()

    params: dict[str, str] = {
        "tag_slug": "temperature",
        "limit": str(limit),
        "offset": str(offset),
    }
    if active_only:
        params["active"] = "true"
        params["closed"] = "false"

    try:
        async with _new_client() as client:
            resp = await client.get("/events", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("gamma_api_events_error", status=exc.response.status_code, detail=str(exc))
        return []
    except httpx.RequestError as exc:
        log.error("gamma_api_request_error", detail=str(exc))
        return []

    if not isinstance(data, list):
        log.warning("gamma_api_unexpected_response", data_type=type(data).__name__)
        return []

    events: list[GammaEvent] = []
    for item in data:
        try:
            events.append(GammaEvent.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            log.warning("gamma_event_parse_error", error=str(exc), raw_keys=list(item.keys()) if isinstance(item, dict) else None)
    return events


async def get_event(event_id: str) -> GammaEvent | None:
    """Fetch a single event by its ID."""
    await gamma_api_limiter.acquire()
    try:
        async with _new_client() as client:
            resp = await client.get(f"/events/{event_id}")
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("gamma_api_event_error", event_id=event_id, status=exc.response.status_code)
        return None
    except httpx.RequestError as exc:
        log.error("gamma_api_request_error", event_id=event_id, detail=str(exc))
        return None
    if not data:
        return None

    try:
        return GammaEvent.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("gamma_event_parse_error", event_id=event_id, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches slugs like "highest-temperature-in-ankara-on-march-15-2026"
_SLUG_PATTERN = re.compile(
    r"highest-temperature-in-(?P<city>[\w-]+)-on-(?P<month>\w+)-(?P<day>\d+)-(?P<year>\d{4})",
    re.IGNORECASE,
)

# Month name to number mapping
_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_city_and_date(event: GammaEvent) -> tuple[str, str] | None:
    """Extract city name and date from an event's slug or title.

    Returns a tuple of (city_display_name, "YYYY-MM-DD") or None if parsing fails.

    Examples:
        slug "highest-temperature-in-ankara-on-march-15-2026" -> ("Ankara", "2026-03-15")
        slug "highest-temperature-in-new-york-city-on-march-15-2026" -> ("New York City", "2026-03-15")
    """
    slug = event.slug or ""
    m = _SLUG_PATTERN.search(slug)
    if not m:
        # Try the title as a fallback
        title_slug = event.title.lower().replace(" ", "-")
        m = _SLUG_PATTERN.search(title_slug)
    if not m:
        log.debug("cannot_parse_slug", slug=slug, title=event.title)
        return None

    raw_city = m.group("city")
    month_name = m.group("month").lower()
    day = int(m.group("day"))
    year = int(m.group("year"))

    month_num = _MONTH_MAP.get(month_name)
    if month_num is None:
        log.warning("unknown_month", month=month_name, slug=slug)
        return None

    # Convert slug-style city name to display name (e.g. "new-york-city" -> "New York City")
    city_display = raw_city.replace("-", " ").title()
    date_str = f"{year}-{month_num:02d}-{day:02d}"

    return city_display, date_str


def _extract_threshold(label: str) -> float | None:
    """Extract the numeric threshold from a bracket label.

    Examples:
        "29.5+" -> 29.5
        "< 30.0" -> 30.0
        "30.0 - 31.9" -> 30.0  (lower bound)
        "32\u00b0+" -> 32.0
    """
    # Remove degree symbols and common suffixes
    cleaned = label.replace("\u00b0", "").replace("\u00b0F", "").replace("\u00b0C", "").strip()

    # Try to find a number
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return None


def parse_brackets(event: GammaEvent) -> list[BracketInfo]:
    """Extract bracket information from an event's markets.

    Each market in a temperature event represents a temperature bracket
    (e.g. "29.5+" or "30.0 - 31.9"). This function extracts the bracket
    details including token IDs and current prices.
    """
    brackets: list[BracketInfo] = []

    for market in event.markets:
        # Use groupItemTitle if available, otherwise question
        label = market.group_item_title or market.question or ""

        # Parse CLOB token IDs – typically [yes_token, no_token]
        token_ids = market.parsed_clob_token_ids()
        token_id_yes = token_ids[0] if len(token_ids) > 0 else None
        token_id_no = token_ids[1] if len(token_ids) > 1 else None

        # Parse outcome prices – typically [yes_price, no_price]
        prices = market.parsed_outcome_prices()
        yes_price = prices[0] if len(prices) > 0 else 0.0
        no_price = prices[1] if len(prices) > 1 else 0.0

        threshold = _extract_threshold(label)

        brackets.append(
            BracketInfo(
                market_id=market.market_id,
                condition_id=market.condition_id,
                label=label,
                group_item_title=market.group_item_title,
                threshold_temp=threshold,
                token_id_yes=token_id_yes,
                token_id_no=token_id_no,
                yes_price=yes_price,
                no_price=no_price,
                neg_risk=market.neg_risk,
                neg_risk_market_id=market.neg_risk_market_id,
            )
        )

    return brackets


