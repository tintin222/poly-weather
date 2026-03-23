"""Enriched prompt builder — wraps the base _build_prompt() with extra weather sources.

Adds OpenWeather and WeatherAPI data sections without modifying the original prompt.
"""

from __future__ import annotations

from src.strategy.llm_advisor import _build_prompt


def _format_openweather(data: dict, unit_label: str) -> str:
    """Format OpenWeather data into prompt section."""
    if not data or not data.get("current"):
        return ""

    U = unit_label
    cur = data["current"]
    lines = [f"## OpenWeather Current + 3-Hour Forecast (independent source)"]
    lines.append(
        f"  Current: {cur.get('temperature')}{U} | feels_like={cur.get('feels_like')}{U} "
        f"| humidity={cur.get('humidity')}% | {cur.get('condition', '')}"
    )

    fc = data.get("hourly_forecast", [])
    if fc:
        lines.append("  Forecast:")
        for h in fc:
            lines.append(
                f"    {h['hour']:02d}:00 | {h['temperature']}{U} | {h.get('condition', '')}"
            )
    else:
        lines.append("  Forecast: (not available)")

    return "\n".join(lines)


def _format_weatherapi(data: dict, unit_label: str) -> str:
    """Format WeatherAPI data into prompt section."""
    if not data or not data.get("current"):
        return ""

    U = unit_label
    cur = data["current"]
    lines = [f"## WeatherAPI Current + Hourly Forecast (independent source)"]
    lines.append(
        f"  Current: {cur.get('temperature')}{U} | feels_like={cur.get('feels_like')}{U} "
        f"| humidity={cur.get('humidity')}% | {cur.get('condition', '')}"
    )

    fc = data.get("hourly_forecast", [])
    if fc:
        lines.append("  Forecast:")
        for h in fc:
            lines.append(
                f"    {h['hour']:02d}:00 | {h['temperature']}{U} | {h.get('condition', '')}"
            )
    else:
        lines.append("  Forecast: (not available)")

    return "\n".join(lines)


_ENRICHED_EXTRA = (
    "- You have 2 ADDITIONAL weather sources (OpenWeather, WeatherAPI) beyond the standard set. "
    "Use cross-source agreement to increase confidence and cross-source disagreement to decrease confidence."
)


def build_enriched_prompt(
    *,
    city_name: str,
    unit: str,
    today: str,
    current_local_hour: int,
    wu_obs: list[dict],
    wu_forecast: list[dict],
    om_hourly: list[dict],
    metar_obs: list[dict],
    summary_high: float | None,
    brackets: list[dict],
    historical_context: list[dict],
    min_bet_hour: int,
    current_position: dict | None = None,
    model_instructions: str = "",
    openweather_data: dict | None = None,
    weatherapi_data: dict | None = None,
) -> str:
    """Build prompt with all standard data + OpenWeather + WeatherAPI sections."""
    # Combine model instructions with enriched-data instruction
    combined_instructions = model_instructions
    if combined_instructions:
        combined_instructions += "\n" + _ENRICHED_EXTRA
    else:
        combined_instructions = _ENRICHED_EXTRA

    base = _build_prompt(
        city_name=city_name,
        unit=unit,
        today=today,
        current_local_hour=current_local_hour,
        wu_obs=wu_obs,
        wu_forecast=wu_forecast,
        om_hourly=om_hourly,
        metar_obs=metar_obs,
        summary_high=summary_high,
        brackets=brackets,
        historical_context=historical_context,
        min_bet_hour=min_bet_hour,
        current_position=current_position,
        model_instructions=combined_instructions,
    )

    # Build extra weather sections
    unit_label = "°F" if unit == "fahrenheit" else "°C"
    extra_sections = []
    ow = _format_openweather(openweather_data or {}, unit_label)
    if ow:
        extra_sections.append(ow)
    wa = _format_weatherapi(weatherapi_data or {}, unit_label)
    if wa:
        extra_sections.append(wa)

    if not extra_sections:
        return base

    # Insert extra sections before the "---" divider
    extra_block = "\n\n".join(extra_sections)
    return base.replace("\n---\n", f"\n\n{extra_block}\n\n---\n", 1)
