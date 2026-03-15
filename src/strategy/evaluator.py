"""Expected value evaluator for betting decisions.

Combines model confidence (P(peak has occurred)) with market prices
to determine whether a bet has positive expected value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BetDecision:
    action: str  # 'BUY', 'STRONG_BUY', 'SKIP'
    reason: str
    confidence: float = 0.0
    market_price: float = 0.0
    expected_value: float = 0.0
    kelly_fraction: float = 0.0
    bracket_label: str = ""


class EVEvaluator:
    """Evaluates expected value of betting on a temperature bracket."""

    def __init__(
        self,
        min_confidence: float = 0.85,
        min_expected_value: float = 0.03,
    ) -> None:
        self._min_confidence = min_confidence
        self._min_ev = min_expected_value

    def evaluate(
        self,
        confidence: float,
        market_price: float,
        current_max_temp: float,
        bracket_label: str,
    ) -> BetDecision:
        """Compute EV and decide whether to bet.

        EV = confidence - market_price
        If confidence=0.95 and price=0.88, EV = +7% -> BET
        If confidence=0.85 and price=0.90, EV = -5% -> SKIP
        """
        # Check if current max temp matches this bracket
        if not self._temp_matches_bracket(current_max_temp, bracket_label):
            return BetDecision(
                action="SKIP",
                reason=f"Temp {current_max_temp} does not match bracket '{bracket_label}'",
                bracket_label=bracket_label,
            )

        if market_price <= 0 or market_price >= 1.0:
            return BetDecision(
                action="SKIP",
                reason=f"Invalid market price: {market_price}",
                bracket_label=bracket_label,
            )

        if confidence < self._min_confidence:
            return BetDecision(
                action="SKIP",
                reason=f"Low confidence: {confidence:.2%} < {self._min_confidence:.2%}",
                confidence=confidence,
                market_price=market_price,
                bracket_label=bracket_label,
            )

        expected_value = confidence - market_price

        if expected_value < self._min_ev:
            return BetDecision(
                action="SKIP",
                reason=f"Low EV: {expected_value:.2%} < {self._min_ev:.2%}",
                confidence=confidence,
                market_price=market_price,
                expected_value=expected_value,
                bracket_label=bracket_label,
            )

        # Kelly criterion for position sizing: f* = (p*b - q) / b
        # where b = (1/price - 1), p = confidence, q = 1 - confidence
        b = (1.0 / market_price) - 1.0
        q = 1.0 - confidence
        kelly = (confidence * b - q) / b if b > 0 else 0
        kelly = max(0.0, min(kelly, 0.25))  # cap at 25%

        action = "STRONG_BUY" if expected_value > 0.10 else "BUY"

        log.info(
            "bet_evaluated",
            action=action,
            bracket=bracket_label,
            confidence=round(confidence, 4),
            price=market_price,
            ev=round(expected_value, 4),
            kelly=round(kelly, 4),
        )

        return BetDecision(
            action=action,
            reason=f"EV={expected_value:.2%}, Kelly={kelly:.2%}",
            confidence=confidence,
            market_price=market_price,
            expected_value=expected_value,
            kelly_fraction=kelly,
            bracket_label=bracket_label,
        )

    def _temp_matches_bracket(self, temp: float, label: str) -> bool:
        """Check if a temperature value matches a bracket label.

        Handles formats like: '14°C', '42-43°F', '16°C or higher',
        '13°C or below', '54°F or higher'
        """
        label_lower = label.lower()
        # Remove degree symbols and unit
        cleaned = label.replace("°C", "").replace("°F", "").strip()

        rounded = round(temp)

        # "X or higher" - rounded temp must be >= X
        if "or higher" in label_lower:
            nums = re.findall(r"-?\d+\.?\d*", cleaned)
            if nums:
                return rounded >= round(float(nums[0]))

        # "X or below" - rounded temp must be <= X
        if "or below" in label_lower:
            nums = re.findall(r"-?\d+\.?\d*", cleaned)
            if nums:
                return rounded <= round(float(nums[0]))

        # Range: "X-Y" (e.g., "42-43")
        range_match = re.match(r"(-?\d+\.?\d*)\s*[-–]\s*(-?\d+\.?\d*)", cleaned)
        if range_match:
            low, high = round(float(range_match.group(1))), round(float(range_match.group(2)))
            return low <= rounded <= high

        # Exact match: "14" means temp rounds to 14
        nums = re.findall(r"-?\d+\.?\d*", cleaned)
        if nums:
            return rounded == round(float(nums[0]))

        return False

    def find_matching_bracket(
        self,
        current_max_temp: float,
        brackets: list[dict],
    ) -> dict | None:
        """Find the bracket that matches the current max temperature.

        Args:
            current_max_temp: Current observed max temperature.
            brackets: List of bracket dicts with 'bracket_label', 'current_yes_price', etc.

        Returns:
            The matching bracket dict, or None.
        """
        for bracket in brackets:
            label = bracket.get("bracket_label", "")
            if self._temp_matches_bracket(current_max_temp, label):
                return bracket
        return None
