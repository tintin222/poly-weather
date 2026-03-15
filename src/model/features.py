"""Feature engineering from hourly weather observations."""

from __future__ import annotations

from collections import defaultdict


def compute_current_max(observations: list) -> tuple[float, int]:
    """Return (max_temp, peak_hour) from today's observations so far.

    Args:
        observations: list of Row-like objects with 'temperature' and 'hour' fields.

    Returns:
        Tuple of (maximum temperature seen, hour at which it occurred).

    Raises:
        ValueError: if observations is empty.
    """
    if not observations:
        raise ValueError("No observations provided")

    max_temp = float("-inf")
    peak_hour = 0
    for obs in observations:
        temp = obs["temperature"]
        hour = obs["hour"]
        if temp > max_temp:
            max_temp = temp
            peak_hour = hour
    return max_temp, peak_hour


def compute_temperature_trend(observations: list, window: int = 3) -> float:
    """Compute temperature trend over the last N readings.

    Positive value means temperature is still rising; negative means dropping.

    Args:
        observations: list of Row-like objects with 'temperature' field, ordered by hour.
        window: number of recent readings to consider.

    Returns:
        Temperature change (last - first) over the window. Returns 0.0 if
        fewer than 2 observations in the window.
    """
    if len(observations) < 2:
        return 0.0

    recent = observations[-window:]
    if len(recent) < 2:
        return 0.0

    return recent[-1]["temperature"] - recent[0]["temperature"]


def compute_hours_since_peak(observations: list) -> int:
    """How many hours since the current max was recorded.

    Args:
        observations: list of Row-like objects with 'temperature' and 'hour' fields,
            ordered by hour.

    Returns:
        Difference between the last observation's hour and the peak hour.
        Returns 0 if observations is empty.
    """
    if not observations:
        return 0

    _, peak_hour = compute_current_max(observations)
    last_hour = observations[-1]["hour"]
    return last_hour - peak_hour


def extract_peak_hours_from_history(historical_data: list) -> list[int]:
    """For each day in historical data, find the hour when the max occurred.

    Groups observations by date and finds the peak hour for each day.

    Args:
        historical_data: list of Row-like objects with 'date', 'temperature',
            and 'hour' fields.

    Returns:
        List of peak hours (one per day found in the data).
    """
    # Group observations by date
    daily: dict[str, list] = defaultdict(list)
    for obs in historical_data:
        daily[obs["date"]].append(obs)

    peak_hours = []
    for date_key in sorted(daily.keys()):
        day_obs = daily[date_key]
        _, peak_hour = compute_current_max(day_obs)
        peak_hours.append(peak_hour)

    return peak_hours
