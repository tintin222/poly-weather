"""Analysis of historical peak-hour patterns."""

from __future__ import annotations

import numpy as np


class PeakHourDistribution:
    """Empirical distribution of daily peak temperature hours.

    Fits a distribution from observed peak hours and provides CDF queries
    to answer: "What is the probability the daily peak has already occurred
    by hour H?"
    """

    def __init__(self) -> None:
        self._peak_hours: np.ndarray | None = None
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, peak_hours: list[int]) -> None:
        """Compute empirical CDF and statistics from historical peak hours.

        Args:
            peak_hours: list of hours (0-23) at which historical daily peaks occurred.

        Raises:
            ValueError: if peak_hours is empty.
        """
        if not peak_hours:
            raise ValueError("Cannot fit distribution with empty peak_hours")

        self._peak_hours = np.array(peak_hours, dtype=float)
        self._mean = float(np.mean(self._peak_hours))
        self._std = float(np.std(self._peak_hours, ddof=1)) if len(peak_hours) > 1 else 1.0
        # Ensure std is never zero (would break CDF)
        if self._std < 0.1:
            self._std = 0.1

    def cdf(self, hour: int) -> float:
        """P(peak_hour <= hour) -- probability that the peak has occurred by this hour.

        Uses a smooth CDF based on the empirical data. We use the empirical CDF
        (fraction of historical peak hours <= the given hour) for robustness.

        Args:
            hour: hour of day (0-23).

        Returns:
            Probability in [0, 1].

        Raises:
            RuntimeError: if fit() has not been called.
        """
        if self._peak_hours is None:
            raise RuntimeError("Must call fit() before cdf()")

        # Empirical CDF: fraction of historical peaks at or before this hour
        return float(np.mean(self._peak_hours <= hour))

    @property
    def mean_peak_hour(self) -> float:
        """Average hour of peak temperature."""
        if self._peak_hours is None:
            raise RuntimeError("Must call fit() first")
        return self._mean

    @property
    def std_peak_hour(self) -> float:
        """Standard deviation of peak hours."""
        if self._peak_hours is None:
            raise RuntimeError("Must call fit() first")
        return self._std

    def percentile_range(self, low: int = 10, high: int = 90) -> tuple[int, int]:
        """Typical peak hour range (low-th to high-th percentile).

        Args:
            low: lower percentile (default 10).
            high: upper percentile (default 90).

        Returns:
            Tuple of (low_percentile_hour, high_percentile_hour).

        Raises:
            RuntimeError: if fit() has not been called.
        """
        if self._peak_hours is None:
            raise RuntimeError("Must call fit() first")

        low_val = int(np.percentile(self._peak_hours, low))
        high_val = int(np.percentile(self._peak_hours, high))
        return low_val, high_val
