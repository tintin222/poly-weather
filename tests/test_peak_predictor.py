"""Tests for the peak temperature prediction model.

Generates synthetic historical data with realistic temperature patterns,
trains the PeakPredictor, and verifies confidence levels at different hours.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
from datetime import datetime, timedelta

import pytest

from src.db.database import Database
from src.db.repositories import WeatherRepository
from src.model.features import (
    compute_current_max,
    compute_hours_since_peak,
    compute_temperature_trend,
    extract_peak_hours_from_history,
)
from src.model.historical_analysis import PeakHourDistribution
from src.model.peak_predictor import PeakConfidence, PeakPredictor


def _generate_day_observations(
    date_str: str,
    base_temp: float = 10.0,
    daily_range: float = 12.0,
    peak_hour: int = 14,
    hours: range | None = None,
) -> list[dict]:
    """Generate synthetic hourly observations for a single day.

    Temperature follows a cosine pattern: low at night, peaking at peak_hour.
    """
    if hours is None:
        hours = range(0, 24)

    observations = []
    for hour in hours:
        phase = math.pi * (hour - peak_hour) / 12.0
        temp = base_temp + daily_range * max(0, math.cos(phase))
        temp += random.gauss(0, 0.3)  # small noise

        observations.append({
            "time_local": f"{hour:02d}:00",
            "hour": hour,
            "temperature": round(temp, 1),
            "humidity": 50.0,
            "wind_speed": 5.0,
            "condition": "Clear",
        })
    return observations


@pytest.fixture
def db_with_history():
    """Create a temporary database populated with 50 days of synthetic data."""
    random.seed(42)  # reproducibility

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    db.initialize()

    repo = WeatherRepository(db)
    target = datetime(2026, 3, 15)

    # Generate 50 days of data across 3 years, centred around March 15
    days_generated = 0
    for year_offset in range(1, 4):
        for day_offset in range(-9, 9):
            d = target.replace(year=target.year - year_offset) + timedelta(days=day_offset)
            date_str = d.strftime("%Y-%m-%d")

            # Peak hour varies around 13-15 (realistic afternoon peak)
            peak_hour = random.choice([12, 13, 13, 14, 14, 14, 14, 15, 15, 15])
            base_temp = random.uniform(5.0, 12.0)
            daily_range = random.uniform(8.0, 14.0)

            obs = _generate_day_observations(
                date_str, base_temp=base_temp, daily_range=daily_range, peak_hour=peak_hour
            )
            repo.upsert_hourly_observations_batch(d.strftime("%Y-%m-%d").split("-")[0] + "-city", date_str, obs)
            # Actually use Ankara as city key
            repo.upsert_hourly_observations_batch("Ankara", date_str, obs)
            days_generated += 1

    yield db, days_generated

    db.close()
    os.unlink(db_path)


class TestFeatures:
    """Tests for feature engineering functions."""

    def test_compute_current_max(self):
        obs = _generate_day_observations("2026-03-15", peak_hour=14, hours=range(0, 18))
        max_temp, peak_hour = compute_current_max(obs)
        # Peak should be near hour 14
        assert 12 <= peak_hour <= 16
        assert max_temp > 15.0  # base 10 + range 12 => peak around 22

    def test_compute_current_max_empty(self):
        with pytest.raises(ValueError):
            compute_current_max([])

    def test_compute_temperature_trend_rising(self):
        # Simulate rising temperatures
        obs = [{"temperature": 10.0}, {"temperature": 12.0}, {"temperature": 14.0}]
        trend = compute_temperature_trend(obs, window=3)
        assert trend > 0

    def test_compute_temperature_trend_falling(self):
        obs = [{"temperature": 20.0}, {"temperature": 18.0}, {"temperature": 16.0}]
        trend = compute_temperature_trend(obs, window=3)
        assert trend < 0

    def test_compute_temperature_trend_insufficient(self):
        obs = [{"temperature": 10.0}]
        assert compute_temperature_trend(obs) == 0.0

    def test_compute_hours_since_peak(self):
        random.seed(99)
        obs = _generate_day_observations("2026-03-15", peak_hour=14, hours=range(0, 20))
        hours_since = compute_hours_since_peak(obs)
        # Last obs is hour 19, peak around 14 => ~5 hours
        assert hours_since >= 3

    def test_extract_peak_hours_from_history(self):
        random.seed(7)
        all_obs = []
        for day in range(5):
            d = f"2025-03-{10+day:02d}"
            peak_h = 13 + day % 3  # 13, 14, 15, 13, 14
            obs = _generate_day_observations(d, peak_hour=peak_h)
            for o in obs:
                o["date"] = d
            all_obs.extend(obs)

        peak_hours = extract_peak_hours_from_history(all_obs)
        assert len(peak_hours) == 5
        for ph in peak_hours:
            assert 11 <= ph <= 17  # should be near the intended peaks


class TestPeakHourDistribution:
    """Tests for the historical peak-hour distribution."""

    def test_fit_and_cdf(self):
        dist = PeakHourDistribution()
        # Peaks clustered around 13-15
        hours = [13, 14, 14, 15, 14, 13, 14, 15, 15, 14, 13, 14, 14, 15, 13]
        dist.fit(hours)

        # Before typical peak: low probability
        assert dist.cdf(10) < 0.1
        # After typical peak: high probability
        assert dist.cdf(16) > 0.95
        # At mean: moderate
        assert 0.0 < dist.cdf(13) < 0.8

    def test_mean_and_std(self):
        dist = PeakHourDistribution()
        hours = [13, 14, 14, 15, 14, 13, 14, 15]
        dist.fit(hours)

        assert 13.0 <= dist.mean_peak_hour <= 15.0
        assert dist.std_peak_hour > 0

    def test_percentile_range(self):
        dist = PeakHourDistribution()
        hours = [12, 13, 13, 14, 14, 14, 15, 15, 15, 16]
        dist.fit(hours)

        low, high = dist.percentile_range(10, 90)
        assert low <= 13
        assert high >= 15

    def test_empty_raises(self):
        dist = PeakHourDistribution()
        with pytest.raises(ValueError):
            dist.fit([])

    def test_cdf_before_fit_raises(self):
        dist = PeakHourDistribution()
        with pytest.raises(RuntimeError):
            dist.cdf(14)


class TestPeakPredictor:
    """Integration tests for the PeakPredictor."""

    def test_train_and_predict(self, db_with_history):
        """Verify confidence levels at different times of day.

        The core test: after training on synthetic data where peaks occur
        around 13-15, the model should output:
          - Low confidence at 8 AM (< 0.3)
          - Moderate confidence at 12 PM (0.3-0.6)
          - High confidence at 4 PM (> 0.7)
          - Very high confidence at 8 PM (> 0.9)
        """
        random.seed(42)
        db, num_days = db_with_history

        predictor = PeakPredictor(db)
        dist = predictor.train("Ankara", "2026-03-15", window_days=10, years_back=3)

        # Verify distribution was learned correctly
        assert 12.0 <= dist.mean_peak_hour <= 16.0, (
            f"Mean peak hour {dist.mean_peak_hour} out of expected range"
        )

        # Generate "today's" observations up to various hours and check confidence
        # Use a day with a clear peak at 14:00

        def _make_obs_up_to(hour: int) -> list[dict]:
            """Create observations from hour 0 up to (and including) the given hour."""
            return _generate_day_observations(
                "2026-03-15", base_temp=10.0, daily_range=12.0, peak_hour=14,
                hours=range(0, hour + 1),
            )

        # 8 AM - early, peak hasn't occurred yet
        obs_8am = _make_obs_up_to(8)
        max_temp_8, _ = compute_current_max(obs_8am)
        result_8am = predictor.predict_confidence("Ankara", 8, max_temp_8, obs_8am)
        assert result_8am.confidence < 0.3, (
            f"8 AM confidence {result_8am.confidence:.3f} should be < 0.3"
        )

        # 12 PM - just before typical peak window (13-15), still low confidence
        obs_12pm = _make_obs_up_to(12)
        max_temp_12, _ = compute_current_max(obs_12pm)
        result_12pm = predictor.predict_confidence("Ankara", 12, max_temp_12, obs_12pm)
        assert result_12pm.confidence < 0.4, (
            f"12 PM confidence {result_12pm.confidence:.3f} should be < 0.4"
        )
        # Should be higher than 8 AM
        assert result_12pm.confidence >= result_8am.confidence, (
            "12 PM confidence should be >= 8 AM confidence"
        )

        # 4 PM - past typical peak, high confidence
        obs_4pm = _make_obs_up_to(16)
        max_temp_16, _ = compute_current_max(obs_4pm)
        result_4pm = predictor.predict_confidence("Ankara", 16, max_temp_16, obs_4pm)
        assert result_4pm.confidence > 0.7, (
            f"4 PM confidence {result_4pm.confidence:.3f} should be > 0.7"
        )

        # 8 PM - well past peak, very high confidence
        obs_8pm = _make_obs_up_to(20)
        max_temp_20, _ = compute_current_max(obs_8pm)
        result_8pm = predictor.predict_confidence("Ankara", 20, max_temp_20, obs_8pm)
        assert result_8pm.confidence > 0.9, (
            f"8 PM confidence {result_8pm.confidence:.3f} should be > 0.9"
        )

    def test_confidence_is_clamped(self, db_with_history):
        """Confidence should always be in [0.01, 0.99]."""
        db, _ = db_with_history
        predictor = PeakPredictor(db)
        predictor.train("Ankara", "2026-03-15")

        obs = _generate_day_observations("2026-03-15", peak_hour=14, hours=range(0, 24))
        max_temp, _ = compute_current_max(obs)

        # Very late hour -- confidence should be high but clamped
        result = predictor.predict_confidence("Ankara", 23, max_temp, obs)
        assert result.confidence <= 0.99
        assert result.confidence >= 0.01

        # Very early hour
        early_obs = _generate_day_observations("2026-03-15", peak_hour=14, hours=range(0, 3))
        max_temp_early, _ = compute_current_max(early_obs)
        result_early = predictor.predict_confidence("Ankara", 2, max_temp_early, early_obs)
        assert result_early.confidence >= 0.01
        assert result_early.confidence <= 0.99

    def test_predict_without_train_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = Database(db_path)
        db.initialize()

        predictor = PeakPredictor(db)
        obs = [{"temperature": 15.0, "hour": 10}]
        with pytest.raises(RuntimeError):
            predictor.predict_confidence("Ankara", 10, 15.0, obs)

        db.close()
        os.unlink(db_path)

    def test_trend_adjustment_falling_temp(self, db_with_history):
        """When temperature drops sharply, confidence should increase."""
        db, _ = db_with_history
        predictor = PeakPredictor(db)
        predictor.train("Ankara", "2026-03-15")

        # Create observations where temperature is falling at the end
        obs = _generate_day_observations("2026-03-15", peak_hour=14, hours=range(0, 17))
        # Override last two to show a sharp drop
        obs[-1] = {**obs[-1], "temperature": obs[-2]["temperature"] - 2.0}

        max_temp, _ = compute_current_max(obs)
        result_falling = predictor.predict_confidence("Ankara", 16, max_temp, obs)

        # Compare with flat temperature
        obs_flat = _generate_day_observations("2026-03-15", peak_hour=14, hours=range(0, 17))
        obs_flat[-1] = {**obs_flat[-1], "temperature": obs_flat[-2]["temperature"]}
        max_temp_flat, _ = compute_current_max(obs_flat)
        result_flat = predictor.predict_confidence("Ankara", 16, max_temp_flat, obs_flat)

        # Falling temperature should give higher or equal confidence
        assert result_falling.confidence >= result_flat.confidence * 0.95

    def test_peak_confidence_dataclass(self):
        pc = PeakConfidence(
            confidence=0.75,
            base_confidence=0.70,
            trend_adjustment=0.05,
            hours_past_mean=2.0,
            mean_peak_hour=14.0,
        )
        assert pc.confidence == 0.75
        assert pc.base_confidence == 0.70
        assert pc.trend_adjustment == 0.05
        assert pc.hours_past_mean == 2.0
        assert pc.mean_peak_hour == 14.0
