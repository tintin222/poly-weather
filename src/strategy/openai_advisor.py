"""OpenAI GPT betting advisor — parallel to llm_advisor.py but using OpenAI API.

Reuses the same prompt builder and data classes from llm_advisor.
Only the API call differs: openai.AsyncOpenAI instead of anthropic.AsyncAnthropic.
"""

from __future__ import annotations

import json

from src.strategy.llm_advisor import (
    BracketRecommendation,
    LLMAdvice,
    _build_prompt,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_MODEL = "gpt-5.4-mini-2026-03-17"


async def analyze(
    api_key: str,
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
) -> LLMAdvice:
    """Call GPT to analyze all weather data and return bracket recommendations."""
    import openai

    prompt = _build_prompt(
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
    )

    client = openai.AsyncOpenAI(api_key=api_key)
    message = await client.chat.completions.create(
        model=_MODEL,
        max_completion_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.choices[0].message.content or ""
    log.info("gpt_response_received", city=city_name, chars=len(raw_text), preview=raw_text[:120])

    # Parse JSON — strip markdown fences if present
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        json_text = json_text.split("```")[1]
        if json_text.startswith("json"):
            json_text = json_text[4:]
    json_text = json_text.strip()

    parsed = json.loads(json_text)
    # Handle double-encoded JSON (model wraps the object in a string)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from GPT, got {type(parsed).__name__}: {str(parsed)[:200]}")

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
        model=_MODEL,
        raw_json=parsed,
    )
