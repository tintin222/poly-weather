"""Trading module using py-clob-client for Polymarket order placement.

Supports both live trading and paper trading modes.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.polymarket.models import OrderResult
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from config import Settings

log = get_logger(__name__)


class PolymarketTrader:
    """Manages order placement on Polymarket via the CLOB client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._paper_mode = settings.paper_mode
        self._clob_client = None

        if not self._paper_mode:
            self._init_clob_client()

    def _init_clob_client(self) -> None:
        """Initialize the py-clob-client for live trading."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            host = "https://clob.polymarket.com"
            creds = ApiCreds(
                api_key=self._settings.poly_api_key,
                api_secret=self._settings.poly_api_secret,
                api_passphrase=self._settings.poly_api_passphrase,
            )

            self._clob_client = ClobClient(
                host,
                key=self._settings.poly_private_key,
                chain_id=137,  # Polygon mainnet
                signature_type=self._settings.poly_signature_type,
                creds=creds,
            )
            log.info("clob_client_initialized", paper_mode=False)
        except Exception as exc:  # noqa: BLE001
            log.error("clob_client_init_error", error=str(exc))
            raise

    def place_limit_buy(
        self,
        token_id: str,
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> OrderResult:
        """Place a limit buy order.

        Args:
            token_id: The CLOB token ID to buy.
            price: Limit price (0.0 - 1.0).
            size: Number of shares to buy.
            tick_size: Minimum price increment ("0.01" or "0.001").
            neg_risk: Whether the market uses negative risk.

        Returns:
            OrderResult with order details.
        """
        if self._paper_mode:
            return self._paper_trade(token_id, price, size, "BUY")

        if self._clob_client is None:
            return OrderResult(
                success=False,
                error_msg="CLOB client not initialized",
            )

        try:
            from py_clob_client.order_builder.constants import BUY

            order_args = {
                "token_id": token_id,
                "price": price,
                "size": size,
                "side": BUY,
            }
            if tick_size:
                order_args["tick_size"] = tick_size

            if neg_risk:
                signed_order = self._clob_client.create_and_post_order(order_args)
            else:
                signed_order = self._clob_client.create_and_post_order(order_args)

            order_id = ""
            status = "UNKNOWN"
            success = False

            if isinstance(signed_order, dict):
                order_id = signed_order.get("orderID", signed_order.get("id", ""))
                status = signed_order.get("status", "SUBMITTED")
                success = signed_order.get("success", status != "ERROR")
                error_msg = signed_order.get("errorMsg")
            else:
                # Some versions of the client return the response differently
                order_id = str(signed_order) if signed_order else ""
                status = "SUBMITTED"
                success = bool(signed_order)
                error_msg = None

            log.info(
                "order_placed",
                order_id=order_id,
                token_id=token_id[:16],
                price=price,
                size=size,
                status=status,
            )

            return OrderResult(
                order_id=order_id,
                status=status,
                success=success,
                error_msg=error_msg,
                is_paper=False,
            )

        except Exception as exc:  # noqa: BLE001
            log.error("order_error", token_id=token_id[:16], price=price, size=size, error=str(exc))
            return OrderResult(
                success=False,
                error_msg=str(exc),
            )

    def place_limit_sell(
        self,
        token_id: str,
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> OrderResult:
        """Place a limit sell order (close an existing position)."""
        if self._paper_mode:
            return self._paper_trade(token_id, price, size, "SELL")

        if self._clob_client is None:
            return OrderResult(success=False, error_msg="CLOB client not initialized")

        try:
            from py_clob_client.order_builder.constants import SELL

            order_args = {
                "token_id": token_id,
                "price": price,
                "size": size,
                "side": SELL,
            }
            if tick_size:
                order_args["tick_size"] = tick_size

            signed_order = self._clob_client.create_and_post_order(order_args)

            order_id = ""
            status = "UNKNOWN"
            success = False

            if isinstance(signed_order, dict):
                order_id = signed_order.get("orderID", signed_order.get("id", ""))
                status = signed_order.get("status", "SUBMITTED")
                success = signed_order.get("success", status != "ERROR")
                error_msg = signed_order.get("errorMsg")
            else:
                order_id = str(signed_order) if signed_order else ""
                status = "SUBMITTED"
                success = bool(signed_order)
                error_msg = None

            log.info(
                "sell_order_placed",
                order_id=order_id,
                token_id=token_id[:16],
                price=price,
                size=size,
                status=status,
            )

            return OrderResult(
                order_id=order_id,
                status=status,
                success=success,
                error_msg=error_msg,
                is_paper=False,
            )

        except Exception as exc:  # noqa: BLE001
            log.error("sell_order_error", token_id=token_id[:16], price=price, size=size, error=str(exc))
            return OrderResult(success=False, error_msg=str(exc))

    def get_open_orders(self) -> list[dict]:
        """Get all open orders.

        Returns:
            List of order dicts from the CLOB API.
        """
        if self._paper_mode:
            log.info("paper_get_open_orders")
            return []

        if self._clob_client is None:
            log.error("clob_client_not_initialized")
            return []

        try:
            orders = self._clob_client.get_orders()
            if isinstance(orders, list):
                return orders
            return []
        except Exception as exc:  # noqa: BLE001
            log.error("get_orders_error", error=str(exc))
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        if self._paper_mode:
            log.info("paper_cancel_order", order_id=order_id)
            return True

        if self._clob_client is None:
            log.error("clob_client_not_initialized")
            return False

        try:
            result = self._clob_client.cancel(order_id)
            success = bool(result)
            log.info("order_cancelled", order_id=order_id, success=success)
            return success
        except Exception as exc:  # noqa: BLE001
            log.error("cancel_order_error", order_id=order_id, error=str(exc))
            return False

    def _paper_trade(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
    ) -> OrderResult:
        """Simulate a trade in paper trading mode."""
        paper_order_id = f"PAPER-{uuid.uuid4().hex[:12]}"

        log.info(
            "paper_trade",
            order_id=paper_order_id,
            token_id=token_id[:16],
            side=side,
            price=price,
            size=size,
            cost=round(price * size, 4),
        )

        return OrderResult(
            order_id=paper_order_id,
            status="PAPER_FILLED",
            success=True,
            is_paper=True,
        )
