"""Google Gemini betting advisor — parallel to llm_advisor.py and openai_advisor.py.

Reuses the same prompt builder and data classes from llm_advisor.
Only the API call differs: google.genai.Client instead of anthropic/openai.
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

_MODEL = "gemini-3.1-flash-lite-preview"


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
    peak_history: list[dict],
    avg_peak_hour: float,
) -> LLMAdvice:
    """Call Gemini to analyze all weather data and return bracket recommendations."""
    from google import genai
    from google.genai import types

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
        peak_history=peak_history,
        avg_peak_hour=avg_peak_hour,
    )

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=1500),
    )

    raw_text = response.text or ""
    log.info("gemini_response_received", city=city_name, chars=len(raw_text))

    # Parse JSON — strip markdown fences if present
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        json_text = json_text.split("```")[1]
        if json_text.startswith("json"):
            json_text = json_text[4:]
    json_text = json_text.strip()

    parsed = json.loads(json_text)

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
