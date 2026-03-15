"""Timezone and time utility functions."""

from __future__ import annotations

from datetime import datetime, date
from zoneinfo import ZoneInfo


def now_in_tz(timezone: str) -> datetime:
    """Get current datetime in a specific timezone."""
    return datetime.now(ZoneInfo(timezone))


def today_in_tz(timezone: str) -> str:
    """Get today's date string (YYYY-MM-DD) in a specific timezone."""
    return now_in_tz(timezone).strftime("%Y-%m-%d")


def current_hour_in_tz(timezone: str) -> int:
    """Get current hour (0-23) in a specific timezone."""
    return now_in_tz(timezone).hour


def parse_date(date_str: str) -> date:
    """Parse a YYYY-MM-DD date string."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def date_range_for_window(target_date: str, window_days: int = 10) -> tuple[int, int]:
    """Return (day_start, day_end) for a +/- window around a target date's day-of-month.

    Clamps to 1-28 to avoid month boundary issues.
    """
    d = parse_date(target_date)
    start = max(1, d.day - window_days)
    end = min(28, d.day + window_days)
    return start, end
