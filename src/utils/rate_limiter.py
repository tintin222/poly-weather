"""Token-bucket rate limiter (ported from copy-cat/src/api/rate-limiter.ts)."""

from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimiter:
    """Async token-bucket rate limiter with queue-based backoff."""

    def __init__(self, max_tokens: int, refill_interval_ms: int) -> None:
        self._max_tokens = max_tokens
        self._refill_interval_s = refill_interval_ms / 1000.0
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._waiters: deque[asyncio.Future[None]] = deque()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = (elapsed / self._refill_interval_s) * self._max_tokens
        if tokens_to_add > 0:
            self._tokens = min(self._max_tokens, self._tokens + tokens_to_add)
            self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        self._refill()
        if self._tokens >= 1:
            self._tokens -= 1
            return

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._waiters.append(future)

        wait_time = self._refill_interval_s / self._max_tokens
        await asyncio.sleep(wait_time)

        self._refill()
        if not future.done():
            future.set_result(None)
        self._waiters.popleft()
        self._tokens -= 1

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


# Pre-configured limiters for Polymarket APIs
gamma_api_limiter = RateLimiter(250, 10_000)  # 250 of 300 req/10s
clob_api_limiter = RateLimiter(400, 10_000)   # 400 of 500 req/10s
data_api_limiter = RateLimiter(180, 10_000)   # 180 of 200 req/10s
