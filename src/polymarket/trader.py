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

    def _get_book_price(self, token_id: str, side: str) -> tuple[float, str]:
        """Get the best price from the order book and the market's tick size.

        For BUY: returns best ask (lowest sell offer) — we match it to fill instantly.
        For SELL: returns best bid (highest buy offer).
        Returns (price, tick_size_str).
        """
        book = self._clob_client.get_order_book(token_id)
        tick = book.tick_size or "0.01"

        if side == "BUY":
            if book.asks:
                return float(book.asks[0].price), tick
        else:
            if book.bids:
                return float(book.bids[0].price), tick

        return 0.0, tick

    def _round_for_tick(self, price: float, size: float, tick_size: str) -> tuple[float, float]:
        """Round price and size to the precision required by the market's tick size."""
        import math
        # tick_size determines price precision: "0.01" → 2 decimals, "0.001" → 3 decimals
        price_decimals = len(tick_size.split(".")[-1]) if "." in tick_size else 2
        clean_price = round(price, price_decimals)
        # Size: floor to (price_decimals + 2) decimals to stay within maker amount limits
        size_decimals = price_decimals + 2
        factor = 10 ** size_decimals
        clean_size = math.floor(size * factor) / factor
        return clean_price, clean_size

    def place_limit_buy(
        self,
        token_id: str,
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> OrderResult:
        """Place a limit buy order at the best available ask price for instant fill.

        Fetches the order book to get the real best ask and tick size,
        then places a limit order with properly rounded price/size.
        """
        if self._paper_mode:
            return self._paper_trade(token_id, price, size, "BUY")

        if self._clob_client is None:
            return OrderResult(success=False, error_msg="CLOB client not initialized")

        try:
            from py_clob_client.order_builder.constants import BUY
            from py_clob_client.clob_types import OrderArgs

            # Get best ask price and tick size from order book
            book_price, market_tick = self._get_book_price(token_id, "BUY")
            if book_price <= 0:
                return OrderResult(success=False, error_msg="No asks in order book")

            # Use the book price (what sellers are offering)
            use_price = book_price
            use_size = size

            # Round to market precision
            clean_price, clean_size = self._round_for_tick(use_price, use_size, market_tick)
            if clean_size < 5:
                # Polymarket minimum is 5 shares — adjust size up
                clean_size = 5.0

            order_args = OrderArgs(
                token_id=token_id,
                price=clean_price,
                size=clean_size,
                side=BUY,
            )

            signed_order = self._clob_client.create_and_post_order(order_args)
            return self._parse_order_response(signed_order, token_id, clean_price, clean_size, "order_placed")

        except Exception as exc:  # noqa: BLE001
            log.error("order_error", token_id=token_id[:16], price=price, size=size, error=str(exc))
            return OrderResult(success=False, error_msg=str(exc))

    def place_limit_sell(
        self,
        token_id: str,
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> OrderResult:
        """Place a limit sell order at the best available bid price for instant fill."""
        if self._paper_mode:
            return self._paper_trade(token_id, price, size, "SELL")

        if self._clob_client is None:
            return OrderResult(success=False, error_msg="CLOB client not initialized")

        try:
            from py_clob_client.order_builder.constants import SELL
            from py_clob_client.clob_types import OrderArgs

            # Get best bid price and tick size from order book
            book_price, market_tick = self._get_book_price(token_id, "SELL")
            if book_price <= 0:
                return OrderResult(success=False, error_msg="No bids in order book")

            # Round to market precision
            clean_price, clean_size = self._round_for_tick(book_price, size, market_tick)

            order_args = OrderArgs(
                token_id=token_id,
                price=clean_price,
                size=clean_size,
                side=SELL,
            )

            signed_order = self._clob_client.create_and_post_order(order_args)
            return self._parse_order_response(signed_order, token_id, clean_price, clean_size, "sell_order_placed")

        except Exception as exc:  # noqa: BLE001
            log.error("sell_order_error", token_id=token_id[:16], price=price, size=size, error=str(exc))
            return OrderResult(success=False, error_msg=str(exc))

    def _parse_order_response(self, resp, token_id: str, price: float, size: float, log_event: str) -> OrderResult:
        """Parse order response from CLOB API."""
        order_id = ""
        status = "UNKNOWN"
        success = False
        error_msg = None

        if isinstance(resp, dict):
            order_id = resp.get("orderID", resp.get("id", ""))
            status = resp.get("status", "SUBMITTED")
            success = resp.get("success", status != "ERROR")
            error_msg = resp.get("errorMsg")
        else:
            order_id = str(resp) if resp else ""
            status = "SUBMITTED"
            success = bool(resp)

        log.info(log_event, order_id=order_id, token_id=token_id[:16],
                 price=price, size=size, status=status)

        return OrderResult(
            order_id=order_id, status=status, success=success,
            error_msg=error_msg, is_paper=False,
        )

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
