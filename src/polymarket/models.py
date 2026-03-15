"""Pydantic models for Polymarket API data."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GammaMarket(BaseModel):
    """A single market (bracket) within a Gamma event."""

    market_id: str = Field(alias="id")
    question: str = ""
    condition_id: str = Field(alias="conditionId", default="")
    slug: str = ""
    outcomes: str = "[]"  # JSON string like '["Yes","No"]'
    outcome_prices: str = Field(alias="outcomePrices", default="[]")  # JSON string
    clob_token_ids: str | None = Field(alias="clobTokenIds", default=None)  # JSON string
    active: bool = True
    closed: bool = False
    accepting_orders: bool | None = Field(alias="acceptingOrders", default=None)
    volume: str | None = None
    liquidity: str | None = None
    group_item_title: str | None = Field(alias="groupItemTitle", default=None)
    group_item_threshold: str | float | None = Field(alias="groupItemThreshold", default=None)
    neg_risk: bool = Field(alias="negRisk", default=False)
    neg_risk_market_id: str | None = Field(alias="negRiskMarketID", default=None)
    neg_risk_request_id: str | None = Field(alias="negRiskRequestID", default=None)
    volume: float | str | None = None
    liquidity: float | str | None = None

    model_config = {"populate_by_name": True}

    def parsed_outcomes(self) -> list[str]:
        """Parse the outcomes JSON string."""
        try:
            return json.loads(self.outcomes)
        except (json.JSONDecodeError, TypeError):
            return []

    def parsed_outcome_prices(self) -> list[float]:
        """Parse the outcome_prices JSON string."""
        try:
            return [float(p) for p in json.loads(self.outcome_prices)]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def parsed_clob_token_ids(self) -> list[str]:
        """Parse the clob_token_ids JSON string."""
        if not self.clob_token_ids:
            return []
        try:
            return json.loads(self.clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            return []


class GammaEvent(BaseModel):
    """An event from the Gamma API, containing multiple markets (brackets)."""

    event_id: str = Field(alias="id")
    title: str = ""
    slug: str = ""
    description: str | None = None
    start_date: str | None = Field(alias="startDate", default=None)
    end_date: str | None = Field(alias="endDate", default=None)
    closed_time: str | None = Field(alias="closedTime", default=None)
    active: bool = True
    closed: bool = False
    volume: float | str | None = None
    liquidity: float | str | None = None
    markets: list[GammaMarket] = Field(default_factory=list)
    neg_risk: bool = Field(alias="negRisk", default=False)

    model_config = {"populate_by_name": True}

    @field_validator("markets", mode="before")
    @classmethod
    def parse_markets(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        return v


class BracketInfo(BaseModel):
    """Parsed bracket information extracted from a GammaMarket within an event."""

    market_id: str
    condition_id: str
    label: str  # e.g. "29.5+" or "< 30.0"
    group_item_title: str | None = None
    threshold_temp: float | None = None  # extracted numeric threshold
    token_id_yes: str | None = None
    token_id_no: str | None = None
    yes_price: float = 0.0
    no_price: float = 0.0
    neg_risk: bool = False
    neg_risk_market_id: str | None = None


class OrderResult(BaseModel):
    """Result from placing a trade order."""

    order_id: str = ""
    status: str = ""  # "LIVE", "MATCHED", "DELAYED", etc.
    success: bool = False
    error_msg: str | None = None
    is_paper: bool = False
