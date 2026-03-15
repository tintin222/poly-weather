"""Market monitor for discovering and tracking Polymarket temperature events.

Polls the Gamma API for temperature events and stores brackets with their
outcome prices. The Gamma API is the authoritative price source — it returns
inline outcomePrices for each market which match the Polymarket UI.
"""

from __future__ import annotations

from src.db.repositories import MarketRepository
from src.polymarket import gamma_api
from src.polymarket.gamma_api import parse_brackets, parse_city_and_date
from src.utils.logger import get_logger
from src.weather.cities import get_city

log = get_logger(__name__)


class MarketMonitor:
    """Discovers temperature events and keeps bracket prices up to date."""

    def __init__(self, market_repo: MarketRepository) -> None:
        self._market_repo = market_repo

    async def poll_all(self) -> int:
        """Full poll cycle: fetch events from Gamma API with prices.

        The Gamma API returns outcomePrices inline with each market,
        so a single call gives us both events and current prices.
        No separate CLOB price fetches needed.
        """
        # Fetch active events (with prices)
        active_events = await gamma_api.get_temperature_events(active_only=True)
        # Also fetch recent closed to detect settlements
        closed_events = await gamma_api.get_temperature_events(active_only=False, limit=50)

        active_ids = {e.event_id for e in active_events}
        all_events = list(active_events)
        for e in closed_events:
            if e.event_id not in active_ids:
                all_events.append(e)

        log.info("polled_events", active=len(active_events), total=len(all_events))

        processed = 0
        for event in all_events:
            parsed = parse_city_and_date(event)
            if parsed is None:
                continue

            city_display, date_str = parsed
            city_config = get_city(city_display)
            city_key = city_config.key if city_config else city_display

            status = "active"
            if event.closed:
                status = "closed"
            elif not event.active:
                status = "inactive"

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

            # Gamma API outcomePrices are the authoritative price source
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

        # Mark DB events as closed if no longer in active set
        db_active = self._market_repo.get_active_events()
        for db_event in db_active:
            if db_event["event_id"] not in active_ids:
                self._market_repo.upsert_event(
                    event_id=db_event["event_id"],
                    city_key=db_event["city_key"],
                    date=db_event["date"],
                    title=db_event["title"],
                    slug=db_event["slug"],
                    status="closed",
                )

        log.info("events_processed", count=processed)
        return processed
