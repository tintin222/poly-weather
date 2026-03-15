"""CLOB API client for fetching prices and order book data.

Ported from copy-cat/src/api/clob-api.ts.
"""

from __future__ import annotations

import asyncio

import httpx

from src.utils.logger import get_logger
from src.utils.rate_limiter import clob_api_limiter

log = get_logger(__name__)

BASE_URL = "https://clob.polymarket.com"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
    return _client


# ---------------------------------------------------------------------------
# Price endpoints
# ---------------------------------------------------------------------------

async def get_price(token_id: str, side: str = "BUY") -> float:
    """Get the current price for a token.

    Args:
        token_id: The CLOB token ID.
        side: "BUY" or "SELL".

    Returns:
        The price as a float (0.0 - 1.0).
    """
    await clob_api_limiter.acquire()
    client = _get_client()

    try:
        resp = await client.get("/price", params={"token_id": token_id, "side": side})
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price", 0))
    except httpx.HTTPStatusError as exc:
        log.error("clob_price_error", token_id=token_id[:16], status=exc.response.status_code)
        return 0.0
    except (httpx.RequestError, ValueError, KeyError) as exc:
        log.error("clob_price_error", token_id=token_id[:16], detail=str(exc))
        return 0.0


async def get_midpoint(token_id: str) -> float:
    """Get the midpoint price for a token.

    Returns:
        The midpoint price as a float (0.0 - 1.0).
    """
    await clob_api_limiter.acquire()
    client = _get_client()

    try:
        resp = await client.get("/midpoint", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("mid", 0))
    except httpx.HTTPStatusError as exc:
        log.error("clob_midpoint_error", token_id=token_id[:16], status=exc.response.status_code)
        return 0.0
    except (httpx.RequestError, ValueError, KeyError) as exc:
        log.error("clob_midpoint_error", token_id=token_id[:16], detail=str(exc))
        return 0.0


async def get_order_book(token_id: str) -> dict:
    """Get the full order book for a token.

    Returns:
        Dict with keys: market, asset_id, bids, asks, timestamp.
        Each bid/ask is {price: str, size: str}.
    """
    await clob_api_limiter.acquire()
    client = _get_client()

    try:
        resp = await client.get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("clob_book_error", token_id=token_id[:16], status=exc.response.status_code)
        return {}
    except httpx.RequestError as exc:
        log.error("clob_book_error", token_id=token_id[:16], detail=str(exc))
        return {}


async def get_prices_batch(token_ids: list[str], side: str = "BUY") -> dict[str, float]:
    """Fetch prices for multiple tokens concurrently.

    Args:
        token_ids: List of CLOB token IDs.
        side: "BUY" or "SELL".

    Returns:
        Dict mapping token_id -> price.
    """
    if not token_ids:
        return {}

    tasks = [get_price(tid, side=side) for tid in token_ids]
    prices = await asyncio.gather(*tasks, return_exceptions=True)

    result: dict[str, float] = {}
    for tid, price in zip(token_ids, prices):
        if isinstance(price, Exception):
            log.warning("batch_price_error", token_id=tid[:16], error=str(price))
            result[tid] = 0.0
        else:
            result[tid] = price
    return result


async def get_last_trade_price(token_id: str) -> float:
    """Get the last traded price for a token."""
    await clob_api_limiter.acquire()
    client = _get_client()

    try:
        resp = await client.get("/last-trade-price", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price", 0))
    except httpx.HTTPStatusError as exc:
        log.error("clob_last_trade_error", token_id=token_id[:16], status=exc.response.status_code)
        return 0.0
    except (httpx.RequestError, ValueError, KeyError) as exc:
        log.error("clob_last_trade_error", token_id=token_id[:16], detail=str(exc))
        return 0.0


async def close() -> None:
    """Close the shared HTTP client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
