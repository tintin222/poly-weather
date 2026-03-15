"""Bracket probability analyzer.

Uses historical peak temperatures to compute the probability that today's
max will fall into each Polymarket bracket. Compares to market prices to
find edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BracketEdge:
    """Analysis of a single bracket's edge vs the market."""
    label: str
    our_probability: float      # our model's P(max lands in this bracket)
    market_probability: float   # market's implied probability (= yes_price)
    edge: float                 # our_prob - market_prob (positive = underpriced)
    ev_per_dollar: float        # expected profit per $1 bet
    verdict: str                # 'BUY', 'AVOID', 'NEUTRAL'
    is_best_bet: bool = False


@dataclass
class BracketGuidance:
    """Full guidance for a city's temperature market."""
    city_key: str
    date: str
    current_temp: float | None
    current_hour: int | None
    historical_avg_high: float
    historical_median_high: float
    historical_min_high: float
    historical_max_high: float
    predicted_bracket: str | None   # bracket most likely to win
    confidence_peak_passed: float | None
    brackets: list[BracketEdge] = field(default_factory=list)
    best_bet: BracketEdge | None = None
    summary: str = ""               # human-readable guidance


def parse_bracket_range(label: str) -> tuple[float | None, float | None]:
    """Parse a bracket label into (low, high) bounds.

    Returns (None, X) for "X or below", (X, None) for "X or higher",
    (X, Y) for "X-Y", (X, X) for exact "X".
    """
    label_clean = label.replace("°C", "").replace("°F", "").strip()

    if "or below" in label.lower():
        nums = re.findall(r"-?\d+\.?\d*", label_clean)
        if nums:
            return None, float(nums[0])

    if "or higher" in label.lower():
        nums = re.findall(r"-?\d+\.?\d*", label_clean)
        if nums:
            return float(nums[0]), None

    range_match = re.match(r"(-?\d+\.?\d*)\s*[-–]\s*(-?\d+\.?\d*)", label_clean)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    nums = re.findall(r"-?\d+\.?\d*", label_clean)
    if nums:
        v = float(nums[0])
        return v, v

    return None, None


def temp_in_bracket(temp: float, low: float | None, high: float | None) -> bool:
    """Check if a temperature falls within a bracket range.

    Uses rounding to match how Polymarket resolves:
    - "14°C" means the daily high rounds to 14 (13.5-14.4°C)
    - "6°C or below" means round(high) <= 6
    - "16°C or higher" means round(high) >= 16
    - "42-43°F" means round(high) is 42 or 43
    """
    rounded = round(temp)
    if low is None and high is not None:
        return rounded <= round(high)
    if high is None and low is not None:
        return rounded >= round(low)
    if low is not None and high is not None:
        if low == high:  # single value bracket like "14°C"
            return rounded == round(low)
        else:  # range bracket like "42-43°F"
            return round(low) <= rounded <= round(high)
    return False


def _forecast_probability(
    forecast_high: float,
    low: float | None,
    high: float | None,
    std: float = 1.5,
) -> float:
    """Compute probability of a bracket given a forecast high.

    Models forecast error as a normal distribution centered at forecast_high
    with standard deviation `std` (default 1.5°C).
    """
    from scipy.stats import norm

    # For "X or below"
    if low is None and high is not None:
        return float(norm.cdf(high + 0.5, loc=forecast_high, scale=std))

    # For "X or higher"
    if high is None and low is not None:
        return float(1 - norm.cdf(low - 0.5, loc=forecast_high, scale=std))

    if low is not None and high is not None:
        if low == high:
            # Single bracket "14°C" → range [13.5, 14.5)
            return float(
                norm.cdf(low + 0.5, loc=forecast_high, scale=std)
                - norm.cdf(low - 0.5, loc=forecast_high, scale=std)
            )
        else:
            # Range bracket "42-43°F" → range [42, 44)
            return float(
                norm.cdf(high + 0.5, loc=forecast_high, scale=std)
                - norm.cdf(low - 0.5, loc=forecast_high, scale=std)
            )

    return 0.0


def compute_bracket_probabilities(
    historical_peaks: list[float],
    brackets: list[dict],
    forecast_high: float | None = None,
) -> list[BracketEdge]:
    """Compute probability for each bracket.

    Uses a blend of:
    - WU forecast high (if available): 70% weight, modeled as normal distribution
    - Historical peak distribution: 30% weight (or 100% if no forecast)
    """
    if not brackets:
        return []

    has_history = bool(historical_peaks) and len(historical_peaks) >= 3
    has_forecast = forecast_high is not None
    n = len(historical_peaks) if has_history else 0

    # Weights
    forecast_weight = 0.7 if has_forecast and has_history else (1.0 if has_forecast else 0.0)
    history_weight = 1.0 - forecast_weight

    results = []

    for b in brackets:
        label = b.get("label") or b.get("bracket_label", "")
        market_price = b.get("yes_price") or b.get("current_yes_price") or 0
        low, high = parse_bracket_range(label)

        # Historical probability
        hist_prob = 0.0
        if has_history:
            count = sum(1 for t in historical_peaks if temp_in_bracket(t, low, high))
            hist_prob = count / n

        # Forecast probability
        fc_prob = 0.0
        if has_forecast:
            fc_prob = _forecast_probability(forecast_high, low, high)

        # Blend
        our_prob = forecast_weight * fc_prob + history_weight * hist_prob

        edge = our_prob - market_price

        # BUY only if there's real market liquidity (price >= 1¢)
        if market_price >= 0.01 and edge > 0.05 and our_prob >= 0.05:
            verdict = "BUY"
        elif market_price >= 0.01 and edge < -0.10:
            verdict = "AVOID"
        elif market_price < 0.01 and our_prob >= 0.10:
            verdict = "CHEAP"  # our model says likely but no/thin market
        else:
            verdict = "NEUTRAL"

        results.append(BracketEdge(
            label=label,
            our_probability=round(our_prob, 4),
            market_probability=round(market_price, 4),
            edge=round(edge, 4),
            ev_per_dollar=round(edge, 4),
            verdict=verdict,
        ))

    # Best bet: must have real market liquidity (>= 1¢) to actually buy
    buyable = [b for b in results if b.verdict == "BUY" and b.market_probability >= 0.01]
    if buyable:
        best = max(buyable, key=lambda b: b.edge)
        best.is_best_bet = True

    return results


def generate_guidance(
    city_key: str,
    date: str,
    historical_peaks: list[float],
    brackets: list[dict],
    current_temp: float | None = None,
    current_hour: int | None = None,
    confidence_peak_passed: float | None = None,
    unit: str = "celsius",
    forecast_high: float | None = None,
) -> BracketGuidance:
    """Generate full betting guidance for a city.

    Args:
        city_key: City identifier.
        date: Event date YYYY-MM-DD.
        historical_peaks: List of daily max temps from last 30 days.
        brackets: List of bracket dicts with label and yes_price.
        current_temp: Current observed max temp today (None if not scraped).
        current_hour: Current local hour.
        confidence_peak_passed: Model confidence that peak has occurred.
        unit: Temperature unit for display.
    """
    import numpy as np

    unit_symbol = "°F" if unit == "fahrenheit" else "°C"

    # Historical stats
    if historical_peaks:
        avg_high = float(np.mean(historical_peaks))
        median_high = float(np.median(historical_peaks))
        min_high = float(np.min(historical_peaks))
        max_high = float(np.max(historical_peaks))
    else:
        avg_high = median_high = min_high = max_high = 0.0

    # Compute bracket probabilities
    bracket_edges = compute_bracket_probabilities(historical_peaks, brackets, forecast_high=forecast_high)

    # Find predicted bracket (highest our_probability)
    predicted_bracket = None
    if bracket_edges:
        best_prob = max(bracket_edges, key=lambda b: b.our_probability)
        if best_prob.our_probability > 0:
            predicted_bracket = best_prob.label

    # Find best bet
    best_bet = next((b for b in bracket_edges if b.is_best_bet), None)

    # Build summary
    summary_parts = []

    if forecast_high is not None:
        summary_parts.append(f"WU Forecast high: {forecast_high:.0f}{unit_symbol}")

    if historical_peaks:
        summary_parts.append(
            f"30d avg: {avg_high:.0f}{unit_symbol}, "
            f"range {min_high:.0f}-{max_high:.0f}{unit_symbol}"
        )

    if current_temp is not None and current_hour is not None:
        summary_parts.append(f"Current peak: {current_temp}{unit_symbol} at {current_hour}:00")

    if confidence_peak_passed is not None:
        pct = confidence_peak_passed * 100
        if pct >= 85:
            summary_parts.append(f"Peak likely passed ({pct:.0f}% confidence)")
        elif pct >= 50:
            summary_parts.append(f"Peak may have passed ({pct:.0f}% confidence)")
        else:
            summary_parts.append(f"Peak likely NOT yet reached ({pct:.0f}% confidence)")

    if best_bet:
        profit_pct = best_bet.edge * 100
        buy_price = best_bet.market_probability * 100
        our_pct = best_bet.our_probability * 100
        summary_parts.append(
            f"BEST BET: {best_bet.label} — we give {our_pct:.0f}%, "
            f"market says {buy_price:.0f}¢, edge +{profit_pct:.0f}%"
        )
    elif bracket_edges:
        # Check if market is fairly priced
        max_edge = max(b.edge for b in bracket_edges) if bracket_edges else 0
        if max_edge <= 0.03:
            summary_parts.append("No edge found — market is fairly priced or overpriced vs our model")

    summary = " | ".join(summary_parts)

    return BracketGuidance(
        city_key=city_key,
        date=date,
        current_temp=current_temp,
        current_hour=current_hour,
        historical_avg_high=round(avg_high, 1),
        historical_median_high=round(median_high, 1),
        historical_min_high=round(min_high, 1),
        historical_max_high=round(max_high, 1),
        predicted_bracket=predicted_bracket,
        confidence_peak_passed=confidence_peak_passed,
        brackets=bracket_edges,
        best_bet=best_bet,
        summary=summary,
    )
