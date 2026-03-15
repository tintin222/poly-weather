"""Risk management for position sizing and exposure limits."""

from __future__ import annotations

from dataclasses import dataclass

from src.db.repositories import MarketRepository
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PositionSize:
    approved: bool
    size: float  # number of shares
    cost: float  # total cost = price * size
    reason: str


class RiskManager:
    """Manages position sizing and daily exposure limits."""

    def __init__(
        self,
        market_repo: MarketRepository,
        max_bet_size: float = 50.0,
        max_daily_exposure: float = 200.0,
    ) -> None:
        self._market_repo = market_repo
        self._max_bet_size = max_bet_size
        self._max_daily_exposure = max_daily_exposure

    def compute_position(
        self,
        date: str,
        price: float,
        kelly_fraction: float,
    ) -> PositionSize:
        """Compute position size given Kelly fraction and risk limits.

        Args:
            date: Trading date (YYYY-MM-DD) for exposure tracking.
            price: Price per share (0-1).
            kelly_fraction: Suggested Kelly fraction (0-0.25).

        Returns:
            PositionSize with approved size or rejection reason.
        """
        if price <= 0 or price >= 1.0:
            return PositionSize(approved=False, size=0, cost=0, reason="Invalid price")

        # Current daily exposure
        current_exposure = self._market_repo.get_daily_exposure(date)
        remaining = self._max_daily_exposure - current_exposure

        if remaining <= 0:
            return PositionSize(
                approved=False, size=0, cost=0,
                reason=f"Daily limit reached: ${current_exposure:.2f} / ${self._max_daily_exposure:.2f}",
            )

        # Kelly-based sizing: fraction of max_bet_size
        bet_amount = kelly_fraction * self._max_bet_size
        bet_amount = min(bet_amount, self._max_bet_size, remaining)

        # Ensure minimum viable bet ($1)
        if bet_amount < 1.0:
            return PositionSize(
                approved=False, size=0, cost=0,
                reason=f"Bet too small: ${bet_amount:.2f}",
            )

        size = bet_amount / price  # number of shares
        cost = bet_amount

        log.info(
            "position_computed",
            price=price,
            kelly=round(kelly_fraction, 4),
            size=round(size, 2),
            cost=round(cost, 2),
            daily_exposure=round(current_exposure + cost, 2),
        )

        return PositionSize(
            approved=True,
            size=round(size, 2),
            cost=round(cost, 2),
            reason=f"Kelly={kelly_fraction:.2%}, exposure=${current_exposure + cost:.2f}/${self._max_daily_exposure:.2f}",
        )
