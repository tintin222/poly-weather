"""Market monitor for discovering and tracking Polymarket temperature events.

Periodically polls the Gamma API for new temperature events, stores them
in the database, and updates bracket prices via the CLOB API.
"""

from __future__ import annotations

from src.db.repositories import MarketRepository
from src.polymarket import clob_api, gamma_api
from src.polymarket.gamma_api import parse_brackets, parse_city_and_date
from src.utils.logger import get_logger
from src.weather.cities import get_city

log = get_logger(__name__)


class MarketMonitor:
    """Discovers temperature events and keeps bracket prices up to date."""

    def __init__(self, market_repo: MarketRepository) -> None:
        self._market_repo = market_repo

    async def poll_temperature_events(self) -> int:
        """Fetch active temperature events and store/update them in the DB.

        Returns:
            Number of events processed.
        """
        events = await gamma_api.get_temperature_events(active_only=True)
        log.info("polled_temperature_events", count=len(events))

        processed = 0
        for event in events:
            parsed = parse_city_and_date(event)
            if parsed is None:
                log.debug("skipping_unparseable_event", event_id=event.event_id, title=event.title)
                continue

            city_display, date_str = parsed

            # Try to match to a known city
            city_config = get_city(city_display)
            city_key = city_config.key if city_config else city_display

            # Determine status
            status = "active"
            if event.closed:
                status = "closed"
            elif not event.active:
                status = "inactive"

            # Upsert the event
            self._market_repo.upsert_event(
                event_id=event.event_id,
                city_key=city_key,
                date=date_str,
                title=event.title,
                slug=event.slug,
                status=status,
                total_volume=float(event.volume) if event.volume else None,
                total_liquidity=float(event.liquidity) if event.liquidity else None,
            )

            # Parse and store brackets
            brackets = parse_brackets(event)
            for bracket in brackets:
                self._market_repo.upsert_bracket(
                    event_id=event.event_id,
                    market_id=bracket.market_id,
                    condition_id=bracket.condition_id,
                    bracket_label=bracket.label,
                    clob_token_id_yes=bracket.token_id_yes,
                    clob_token_id_no=bracket.token_id_no,
                    current_yes_price=bracket.yes_price,
                    current_no_price=bracket.no_price,
                    neg_risk=bracket.neg_risk,
                )

            processed += 1

        log.info("events_processed", count=processed)
        return processed

    async def update_bracket_prices(self, event_id: str) -> int:
        """Update current prices for all brackets of an event from the CLOB API.

        Returns:
            Number of brackets updated.
        """
        brackets = self._market_repo.get_brackets_for_event(event_id)
        if not brackets:
            log.debug("no_brackets_for_event", event_id=event_id)
            return 0

        # Collect all yes-token IDs for batch price fetch
        yes_token_ids: list[str] = []
        bracket_rows = []
        for b in brackets:
            tid = b["clob_token_id_yes"]
            if tid:
                yes_token_ids.append(tid)
                bracket_rows.append(b)

        if not yes_token_ids:
            return 0

        # Batch fetch prices
        prices = await clob_api.get_prices_batch(yes_token_ids)

        updated = 0
        for b, tid in zip(bracket_rows, yes_token_ids):
            yes_price = prices.get(tid, 0.0)

            # Don't overwrite existing price with 0 (CLOB 404 = no liquidity, not price=0)
            if yes_price <= 0 and (b["current_yes_price"] or 0) > 0:
                continue

            no_price = round(1.0 - yes_price, 4) if yes_price > 0 else 0.0

            self._market_repo.upsert_bracket(
                event_id=b["event_id"],
                market_id=b["market_id"],
                condition_id=b["condition_id"],
                bracket_label=b["bracket_label"],
                clob_token_id_yes=b["clob_token_id_yes"],
                clob_token_id_no=b["clob_token_id_no"],
                current_yes_price=yes_price,
                current_no_price=no_price,
                neg_risk=bool(b["neg_risk"]),
            )
            updated += 1

        log.info("bracket_prices_updated", event_id=event_id, count=updated)
        return updated

    async def poll_all(self) -> None:
        """Full poll cycle: discover events then update prices for all active events."""
        # Step 1: Discover / update events
        await self.poll_temperature_events()

        # Step 2: Update prices for all active events
        active_events = self._market_repo.get_active_events()
        for event_row in active_events:
            try:
                await self.update_bracket_prices(event_row["event_id"])
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "price_update_error",
                    event_id=event_row["event_id"],
                    error=str(exc),
                )
