"""Enriched LLM advisors — same models as originals but with extra weather data.

Each analyze function builds an enriched prompt (base + OpenWeather + WeatherAPI)
and calls the respective LLM API. Reuses data classes from llm_advisor.
"""

from __future__ import annotations

import json
from typing import Any

from src.strategy.enriched_prompt import build_enriched_prompt
from src.strategy.llm_advisor import (
    BracketRecommendation,
    LLMAdvice,
    _CLAUDE_INSTRUCTIONS,
)
from src.strategy.openai_advisor import _GPT_INSTRUCTIONS
from src.strategy.gemini_advisor import _GEMINI_INSTRUCTIONS
from src.utils.logger import get_logger

log = get_logger(__name__)

_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_GPT_MODEL = "gpt-5.4-mini-2026-03-17"
_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


def _parse_response(raw_text: str, brackets: list[dict], model: str) -> LLMAdvice:
    """Parse JSON response from any LLM into LLMAdvice."""
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        json_text = json_text.split("```")[1]
        if json_text.startswith("json"):
            json_text = json_text[4:]
    json_text = json_text.strip()

    parsed = json.loads(json_text)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    bracket_lookup = {b["label"]: b for b in brackets}
    recs: list[BracketRecommendation] = []

    for r in parsed.get("recommendations", []):
        label = r.get("bracket", "")
        action = r.get("action", "SKIP")
        conf = float(r.get("confidence", 0.0))
        reasoning = r.get("reasoning", "")

        b = bracket_lookup.get(label)
        if b is None:
            continue

        if action == "BUY_YES":
            token_id = b.get("yes_token_id", "")
            price = float(b.get("yes_price") or 0)
        elif action == "BUY_NO":
            token_id = b.get("no_token_id", "")
            price = float(b.get("no_price") or 0)
        else:
            token_id = ""
            price = 0.0

        recs.append(BracketRecommendation(
            bracket_label=label,
            action=action,
            token_id=token_id,
            price=price,
            confidence=conf,
            reasoning=reasoning,
            bracket_db_id=b.get("db_id", 0),
            event_id=b.get("event_id", ""),
        ))

    return LLMAdvice(
        analysis=parsed.get("analysis", ""),
        peak_temp_estimate=parsed.get("peak_temp_estimate"),
        peak_confidence=float(parsed.get("peak_confidence", 0.0)),
        recommendations=recs,
        model=model,
        raw_json=parsed,
    )


def _build_prompt_kwargs(
    city_name: str, unit: str, today: str, current_local_hour: int,
    wu_obs: list[dict], wu_forecast: list[dict], om_hourly: list[dict],
    metar_obs: list[dict], summary_high: float | None, brackets: list[dict],
    historical_context: list[dict], min_bet_hour: int,
    current_position: dict | None,
    model_instructions: str,
    openweather_data: dict | None,
    weatherapi_data: dict | None,
) -> dict:
    return dict(
        city_name=city_name, unit=unit, today=today,
        current_local_hour=current_local_hour,
        wu_obs=wu_obs, wu_forecast=wu_forecast, om_hourly=om_hourly,
        metar_obs=metar_obs, summary_high=summary_high, brackets=brackets,
        historical_context=historical_context, min_bet_hour=min_bet_hour,
        current_position=current_position,
        model_instructions=model_instructions,
        openweather_data=openweather_data,
        weatherapi_data=weatherapi_data,
    )


async def analyze_claude(
    api_key: str,
    city_name: str, unit: str, today: str, current_local_hour: int,
    wu_obs: list[dict], wu_forecast: list[dict], om_hourly: list[dict],
    metar_obs: list[dict], summary_high: float | None, brackets: list[dict],
    historical_context: list[dict], min_bet_hour: int,
    current_position: dict | None = None,
    openweather_data: dict | None = None,
    weatherapi_data: dict | None = None,
) -> LLMAdvice:
    """Call Claude with enriched weather data."""
    import anthropic

    prompt = build_enriched_prompt(**_build_prompt_kwargs(
        city_name, unit, today, current_local_hour,
        wu_obs, wu_forecast, om_hourly, metar_obs, summary_high, brackets,
        historical_context, min_bet_hour, current_position,
        _CLAUDE_INSTRUCTIONS, openweather_data, weatherapi_data,
    ))

    client = anthropic.AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model=_CLAUDE_MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = message.content[0].text
    log.info("enriched_claude_response", city=city_name, chars=len(raw_text))
    return _parse_response(raw_text, brackets, _CLAUDE_MODEL)


async def analyze_gpt(
    api_key: str,
    city_name: str, unit: str, today: str, current_local_hour: int,
    wu_obs: list[dict], wu_forecast: list[dict], om_hourly: list[dict],
    metar_obs: list[dict], summary_high: float | None, brackets: list[dict],
    historical_context: list[dict], min_bet_hour: int,
    current_position: dict | None = None,
    openweather_data: dict | None = None,
    weatherapi_data: dict | None = None,
) -> LLMAdvice:
    """Call GPT with enriched weather data."""
    import openai

    prompt = build_enriched_prompt(**_build_prompt_kwargs(
        city_name, unit, today, current_local_hour,
        wu_obs, wu_forecast, om_hourly, metar_obs, summary_high, brackets,
        historical_context, min_bet_hour, current_position,
        _GPT_INSTRUCTIONS, openweather_data, weatherapi_data,
    ))

    client = openai.AsyncOpenAI(api_key=api_key)
    message = await client.chat.completions.create(
        model=_GPT_MODEL, max_completion_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = message.choices[0].message.content or ""
    log.info("enriched_gpt_response", city=city_name, chars=len(raw_text))
    return _parse_response(raw_text, brackets, _GPT_MODEL)


async def analyze_gemini(
    api_key: str,
    city_name: str, unit: str, today: str, current_local_hour: int,
    wu_obs: list[dict], wu_forecast: list[dict], om_hourly: list[dict],
    metar_obs: list[dict], summary_high: float | None, brackets: list[dict],
    historical_context: list[dict], min_bet_hour: int,
    current_position: dict | None = None,
    openweather_data: dict | None = None,
    weatherapi_data: dict | None = None,
) -> LLMAdvice:
    """Call Gemini with enriched weather data."""
    from google import genai
    from google.genai import types

    prompt = build_enriched_prompt(**_build_prompt_kwargs(
        city_name, unit, today, current_local_hour,
        wu_obs, wu_forecast, om_hourly, metar_obs, summary_high, brackets,
        historical_context, min_bet_hour, current_position,
        _GEMINI_INSTRUCTIONS, openweather_data, weatherapi_data,
    ))

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=_GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=1500),
    )
    raw_text = response.text or ""
    log.info("enriched_gemini_response", city=city_name, chars=len(raw_text))
    return _parse_response(raw_text, brackets, _GEMINI_MODEL)
