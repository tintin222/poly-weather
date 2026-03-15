"""Peak temperature prediction model.

Combines historical peak-hour distribution with real-time temperature
observations to estimate the probability that today's maximum temperature
has already occurred.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.db.database import Database
from src.db.repositories import WeatherRepository
from src.model.features import (
    compute_temperature_trend,
    extract_peak_hours_from_history,
)
from src.model.historical_analysis import PeakHourDistribution
from src.utils.logger import get_logger
from src.utils.time_utils import date_range_for_window, parse_date

log = get_logger(__name__)


@dataclass
class PeakConfidence:
    """Result of peak prediction."""

    confidence: float  # probability that daily max has already occurred [0, 1]
    base_confidence: float  # from historical CDF alone
    trend_adjustment: float  # adjustment from temperature trend
    hours_past_mean: float  # how many hours past the mean peak hour
    mean_peak_hour: float  # from the model


class PeakPredictor:
    """Predicts whether the daily peak temperature has already occurred.

    Uses historical peak-hour distributions combined with real-time
    temperature trends to produce a confidence score.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._repo = WeatherRepository(db)
        self._distributions: dict[str, PeakHourDistribution] = {}

    def train(
        self,
        city_key: str,
        target_date: str,
        window_days: int = 10,
        years_back: int = 3,
    ) -> PeakHourDistribution:
        """Build the peak-hour distribution for a city around a target date.

        Loads historical hourly data for the same time-of-year window across
        multiple past years and fits a PeakHourDistribution.

        Args:
            city_key: city identifier (e.g. "Ankara").
            target_date: date string "YYYY-MM-DD" around which to gather data.
            window_days: +/- days around target day-of-month.
            years_back: how many years of history to use.

        Returns:
            The fitted PeakHourDistribution.
        """
        d = parse_date(target_date)
        month = d.month
        day_start, day_end = date_range_for_window(target_date, window_days)

        historical = self._repo.get_historical_hourly(
            city_key, month, day_start, day_end, years_back
        )

        if historical:
            peak_hours = extract_peak_hours_from_history(historical)
            log.info(
                "trained_from_historical",
                city=city_key,
                num_days=len(peak_hours),
                month=month,
                day_range=f"{day_start}-{day_end}",
            )
        else:
            # No historical data -- use a sensible default distribution
            # centered around typical afternoon peak hours (13-15)
            log.warning(
                "no_historical_data",
                city=city_key,
                target_date=target_date,
                msg="Using default peak hour distribution",
            )
            peak_hours = [13, 14, 14, 15, 14, 13, 14, 15, 15, 14]

        dist = PeakHourDistribution()
        dist.fit(peak_hours)
        self._distributions[city_key] = dist

        log.info(
            "peak_distribution_fitted",
            city=city_key,
            mean_peak_hour=round(dist.mean_peak_hour, 1),
            std_peak_hour=round(dist.std_peak_hour, 1),
            percentile_range=dist.percentile_range(),
        )

        return dist

    def predict_confidence(
        self,
        city_key: str,
        current_hour: int,
        current_max_temp: float,
        observations: list,
    ) -> PeakConfidence:
        """Return confidence that the daily peak has already occurred.

        Combines:
        1. Historical CDF: P(peak_hour <= current_hour)
        2. Trend adjustment based on recent temperature direction
        3. Time-past-mean adjustment

        Args:
            city_key: city identifier.
            current_hour: current hour of day (0-23).
            current_max_temp: highest temperature observed so far today.
            observations: today's hourly observations so far (Row-like objects).

        Returns:
            PeakConfidence with the combined confidence score.

        Raises:
            RuntimeError: if train() has not been called for this city.
        """
        if city_key not in self._distributions:
            raise RuntimeError(
                f"No distribution for {city_key}. Call train() first."
            )

        dist = self._distributions[city_key]

        # 1. Base confidence from historical CDF
        base_confidence = dist.cdf(current_hour)

        # 2. Trend adjustment
        trend = compute_temperature_trend(observations, window=3)
        trend_adjustment = 0.0

        confidence = base_confidence

        # If temperature dropped > 1.0 degree in last 2 readings: more confident
        if len(observations) >= 2:
            last_two_trend = observations[-1]["temperature"] - observations[-2]["temperature"]
            if last_two_trend < -1.0:
                trend_adjustment = confidence * 0.1  # will multiply by 1.1
                confidence *= 1.1

        # If temperature still rising (> 0.5 degree increase): less confident
        if trend > 0.5:
            adj = confidence * 0.15
            trend_adjustment -= adj  # negative adjustment
            confidence *= 0.85

        # 3. Time-past-mean adjustment
        hours_past_mean = current_hour - dist.mean_peak_hour
        if current_hour > dist.mean_peak_hour + dist.std_peak_hour:
            confidence *= 1.15

        # Clamp to [0.01, 0.99]
        confidence = max(0.01, min(0.99, confidence))

        return PeakConfidence(
            confidence=confidence,
            base_confidence=base_confidence,
            trend_adjustment=trend_adjustment,
            hours_past_mean=hours_past_mean,
            mean_peak_hour=dist.mean_peak_hour,
        )
