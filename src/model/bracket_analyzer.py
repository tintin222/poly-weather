"""Peak-confidence based betting guidance.

Core strategy: detect when the daily peak temperature has passed,
then buy the matching bracket at a discount before the market fully prices it in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.model.features import compute_current_max, compute_temperature_trend, compute_hours_since_peak
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PeakGuidance:
    """Complete betting guidance for a city based on peak-confidence analysis."""
    city_key: str
    date: str

    # Current WU situation
    current_peak: float | None = None
    peak_hour: int | None = None
    current_temp: float | None = None
    latest_hour: int | None = None
    trend: float = 0.0              # positive=rising, negative=falling
    hours_since_peak: int = 0
    observation_count: int = 0

    # Peak confidence
    confidence: float = 0.0         # P(peak has passed) [0-1]
    confidence_reason: str = ""     # human explanation

    # Matching bracket
    matching_bracket: str | None = None
    matching_price: float = 0.0     # market price for matching bracket
    matching_bracket_id: int | None = None
    matching_token_id: str | None = None

    # Best active bracket (highest market price > 2¢)
    best_bracket: str | None = None
    best_bracket_price: float = 0.0

    # Recommendation
    action: str = "WAIT"            # BUY, WAIT, SKIP, NO_DATA
    reason: str = ""
    edge: float = 0.0              # confidence - market_price
    potential_gain_pct: float = 0.0

    # All brackets for display
    brackets: list[dict] = field(default_factory=list)

    # Timestamps
    wu_scraped_at: int | None = None
    market_polled_at: int | None = None


def parse_bracket_range(label: str) -> tuple[float | None, float | None]:
    """Parse a bracket label into (low, high) bounds."""
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
    """Check if a temperature falls within a bracket range (using rounding)."""
    rounded = round(temp)
    if low is None and high is not None:
        return rounded <= round(high)
    if high is None and low is not None:
        return rounded >= round(low)
    if low is not None and high is not None:
        if low == high:
            return rounded == round(low)
        else:
            return round(low) <= rounded <= round(high)
    return False


def find_bracket_for_temp(temp: float, brackets: list[dict]) -> dict | None:
    """Find the bracket that matches a temperature value."""
    for b in brackets:
        label = b.get("label") or b.get("bracket_label", "")
        low, high = parse_bracket_range(label)
        if temp_in_bracket(temp, low, high):
            return b
    return None


def generate_peak_guidance(
    city_key: str,
    date: str,
    brackets: list[dict],
    observations: list | None,
    peak_confidence: float | None,
    peak_confidence_detail: dict | None,
    current_hour: int | None,
    unit: str = "celsius",
    wu_scraped_at: int | None = None,
    market_polled_at: int | None = None,
    min_confidence: float = 0.85,
) -> PeakGuidance:
    """Generate betting guidance based on peak-confidence analysis.

    Core logic:
    1. What's the current peak temperature from WU?
    2. Is the temperature declining?
    3. How confident are we the peak won't go higher?
    4. What bracket does the current peak match?
    5. Is there an edge (confidence > market price)?
    """
    unit_symbol = "°F" if unit == "fahrenheit" else "°C"

    guidance = PeakGuidance(
        city_key=city_key,
        date=date,
        wu_scraped_at=wu_scraped_at,
        market_polled_at=market_polled_at,
        brackets=[
            {
                "label": b.get("label") or b.get("bracket_label", ""),
                "yes_price": b.get("yes_price") or b.get("current_yes_price") or 0,
                "id": b.get("id"),
                "token_id_yes": b.get("token_id_yes") or b.get("clob_token_id_yes"),
            }
            for b in brackets
        ],
    )

    # Compute best active bracket (highest market price > 2¢) — runs unconditionally
    active = [(b["label"], b["yes_price"]) for b in guidance.brackets if b.get("yes_price", 0) > 0.02]
    if active:
        best_label, best_price = max(active, key=lambda x: x[1])
        guidance.best_bracket = best_label
        guidance.best_bracket_price = best_price

    # No observations yet
    if not observations or len(observations) < 2:
        guidance.action = "NO_DATA"
        guidance.reason = "No weather data scraped yet. Waiting for WU scrape..."
        return guidance

    # Current situation from observations
    max_temp, peak_hour = compute_current_max(observations)
    trend = compute_temperature_trend(observations, window=3)
    hours_since = compute_hours_since_peak(observations)

    guidance.current_peak = max_temp
    guidance.peak_hour = peak_hour
    guidance.current_temp = observations[-1]["temperature"]
    guidance.latest_hour = observations[-1]["hour"]
    guidance.trend = round(trend, 1)
    guidance.hours_since_peak = hours_since
    guidance.observation_count = len(observations)

    # Peak confidence
    if peak_confidence is not None:
        guidance.confidence = peak_confidence
        detail = peak_confidence_detail or {}

        # Build explanation
        reasons = []
        base = detail.get("base_confidence", 0)
        mean_h = detail.get("mean_peak_hour", 0)
        hours_past = detail.get("hours_past_mean", 0)

        if base > 0:
            reasons.append(f"Historical: {base*100:.0f}% of days peaked by {current_hour}:00")
        if trend < -0.5:
            reasons.append(f"Temp dropping ({trend:+.1f}{unit_symbol} in last readings)")
        elif trend > 0.5:
            reasons.append(f"Temp still rising ({trend:+.1f}{unit_symbol})")
        if hours_past > 0:
            reasons.append(f"{hours_past:.1f}h past avg peak time ({mean_h:.0f}:00)")
        elif hours_past < -1:
            reasons.append(f"{abs(hours_past):.1f}h before avg peak time ({mean_h:.0f}:00)")

        guidance.confidence_reason = " | ".join(reasons)
    else:
        guidance.action = "WAIT"
        guidance.reason = "Need historical data to calculate peak confidence. Backfilling..."
        return guidance

    # Find matching bracket
    matching = find_bracket_for_temp(max_temp, brackets)
    if matching:
        label = matching.get("label") or matching.get("bracket_label", "")
        price = matching.get("yes_price") or matching.get("current_yes_price") or 0
        guidance.matching_bracket = label
        guidance.matching_price = price
        guidance.matching_bracket_id = matching.get("id")
        guidance.matching_token_id = matching.get("token_id_yes") or matching.get("clob_token_id_yes")
    else:
        guidance.action = "WAIT"
        guidance.reason = f"Current peak {max_temp}{unit_symbol} doesn't match any bracket."
        return guidance

    # Decision logic
    conf = guidance.confidence
    price = guidance.matching_price

    if conf < 0.50:
        guidance.action = "WAIT"
        guidance.reason = (
            f"Too early — only {conf*100:.0f}% confident peak has passed. "
            f"Peak {max_temp}{unit_symbol} at {peak_hour}:00, temp {'still rising' if trend > 0 else 'may still change'}."
        )
    elif conf < min_confidence:
        guidance.action = "WATCH"
        guidance.reason = (
            f"Peak may have passed ({conf*100:.0f}% confident) but below {min_confidence*100:.0f}% threshold. "
            f"Peak {max_temp}{unit_symbol} at {peak_hour}:00. Waiting..."
        )
    elif price >= 0.97:
        guidance.action = "SKIP"
        guidance.reason = (
            f"Peak passed ({conf*100:.0f}% confident) but {guidance.matching_bracket} already at "
            f"{price*100:.0f}¢ — no profit margin left."
        )
    elif price >= conf:
        guidance.action = "SKIP"
        guidance.edge = conf - price
        guidance.reason = (
            f"No edge: {conf*100:.0f}% confident but {guidance.matching_bracket} at {price*100:.0f}¢. "
            f"Need price below {conf*100:.0f}¢ for a profitable bet."
        )
    else:
        # We have an edge!
        guidance.action = "BUY"
        guidance.edge = conf - price
        guidance.potential_gain_pct = round((1.0 / price - 1.0) * 100, 1) if price > 0 else 0
        guidance.reason = (
            f"Peak {max_temp}{unit_symbol} at {peak_hour}:00, declining for {hours_since}h. "
            f"{conf*100:.0f}% confident → buy {guidance.matching_bracket} at {price*100:.0f}¢ "
            f"(+{guidance.edge*100:.0f}% edge, potential +{guidance.potential_gain_pct:.0f}% gain)."
        )

    return guidance
