"""Pydantic models for weather observation data."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HourlyObservation(BaseModel):
    """A single hourly weather observation from Weather Underground."""

    time_local: str = Field(description="Local time string, e.g. '12:00 AM'")
    hour: int = Field(ge=0, le=23, description="Hour in 24h format (0-23)")
    temperature: float = Field(description="Temperature (units depend on city)")
    dew_point: float | None = Field(default=None)
    humidity: float | None = Field(default=None)
    wind_speed: float | None = Field(default=None)
    wind_gust: float | None = Field(default=None)
    wind_direction: str | None = Field(default=None)
    pressure: float | None = Field(default=None)
    precip_rate: float | None = Field(default=None)
    condition: str | None = Field(default=None)

    def to_dict(self) -> dict:
        """Convert to dict for database insertion."""
        return self.model_dump()


class DailyPeak(BaseModel):
    """Summary of the peak temperature for a city on a given date."""

    city_key: str
    date: str
    peak_temp: float
    peak_hour: int = Field(ge=0, le=23)
    observation_count: int = Field(ge=0)


class DailyScrapeResult:
    """Result of a single-day WU scrape: hourly observations + summary high temp.

    summary_high_temp comes from the WU Summary Table "High Temp (Actual)" row,
    which aggregates ALL station readings (including ASOS reports not in the
    hourly table). This is the value Polymarket uses for resolution.
    It may be higher than max(observations.temperature).
    """

    __slots__ = ("observations", "summary_high_temp")

    def __init__(
        self,
        observations: list[HourlyObservation],
        summary_high_temp: float | None,
    ) -> None:
        self.observations = observations
        self.summary_high_temp = summary_high_temp
