"""LLM-powered betting advisor using Claude to analyze multi-source weather data.

Given today's WU observations, WU forecast, Open-Meteo forecast, METAR readings,
the WU summary high, and all available Polymarket bracket prices, Claude recommends
which brackets to BUY YES, which to BUY NO, and which to skip — across all brackets
simultaneously, not just the single most likely winner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BracketRecommendation:
    bracket_label: str
    action: str          # "BUY_YES" | "BUY_NO" | "SKIP"
    token_id: str        # the token to buy (YES or NO depending on action)
    price: float         # current market price of that token
    confidence: float    # 0.0–1.0 LLM confidence
    reasoning: str
    bracket_db_id: int
    event_id: str


@dataclass
class LLMAdvice:
    analysis: str                          # Claude's narrative summary
    peak_temp_estimate: float | None       # Claude's best estimate of today's high
    peak_confidence: float                 # Overall confidence in the estimate
    recommendations: list[BracketRecommendation]
    model: str
    raw_json: dict


def _build_prompt(
    city_name: str,
    unit: str,
    today: str,
    current_local_hour: int,
    wu_obs: list[dict],
    wu_forecast: list[dict],
    om_hourly: list[dict],
    metar_obs: list[dict],
    summary_high: float | None,
    brackets: list[dict],   # includes label, yes_price, no_price, yes_token, no_token
    peak_history: list[dict],
    avg_peak_hour: float,
) -> str:
    unit_label = "°F" if unit == "fahrenheit" else "°C"
    U = unit_label

    # Day completion status
    hours_to_peak = avg_peak_hour - current_local_hour
    if hours_to_peak > 1.5:
        day_status = f"EARLY — {hours_to_peak:.1f} hrs before typical peak ({avg_peak_hour:.1f}:00). WU summary high will likely INCREASE."
        summary_high_note = "⚠ NOT FINAL — day is still early, temperature will likely rise further."
    elif hours_to_peak > -1.0:
        day_status = f"NEAR PEAK — within 1-2 hrs of typical peak ({avg_peak_hour:.1f}:00). Summary high may still tick up slightly."
        summary_high_note = "Approaching final — monitor closely, may rise slightly."
    else:
        day_status = f"LATE — {abs(hours_to_peak):.1f} hrs past typical peak ({avg_peak_hour:.1f}:00). WU summary high is likely near-final."
        summary_high_note = "Likely near-final for the day."

    # Format WU observations
    obs_lines = []
    for o in sorted(wu_obs, key=lambda x: x.get("hour", 0)):
        h = o.get("hour", 0)
        t = o.get("temperature")
        cond = o.get("condition", "")
        hum = o.get("humidity")
        obs_lines.append(
            f"  {h:02d}:00 | {t}{U} | humidity={hum}% | {cond}"
        )

    # Format WU forecast
    fc_lines = []
    for o in sorted(wu_forecast, key=lambda x: x.get("hour", 0)):
        h = o.get("hour", 0)
        t = o.get("temperature")
        cond = o.get("condition", "")
        fc_lines.append(f"  {h:02d}:00 | {t}{U} | {cond}")

    # Format OM forecast
    om_lines = []
    for o in sorted(om_hourly, key=lambda x: x.get("hour", 0)):
        h = o.get("hour", 0)
        t = o.get("temperature")
        is_fc = o.get("is_forecast", False)
        tag = " fc" if is_fc else ""
        om_lines.append(f"  {h:02d}:00 | {t}{U}{tag}")

    # Format METAR
    metar_lines = []
    for o in sorted(metar_obs, key=lambda x: x.get("hour", 0)):
        h = o.get("hour", 0)
        t = o.get("temperature")
        metar_lines.append(f"  {h:02d}:00 | {t}{U}")

    # Format brackets — include market implied probability interpretation
    bracket_lines = []
    for b in brackets:
        label = b["label"]
        yp = b.get("yes_price") or 0
        np_ = b.get("no_price") or 0
        bracket_lines.append(
            f"  {label:15s} | YES={yp:.3f} ({round(yp*100)}% market prob)  NO={np_:.3f}"
        )

    # Format peak history (last 14 days for better seasonal context)
    hist_lines = []
    for r in peak_history[:14]:
        hist_lines.append(
            f"  {r.get('date', '')} | peak={r.get('peak_temp')}{U} | peak_hour={r.get('peak_hour', '?')}:00"
        )

    wu_obs_block = "\n".join(obs_lines) if obs_lines else "  (none yet)"
    wu_fc_block = "\n".join(fc_lines) if fc_lines else "  (none available)"
    om_block = "\n".join(om_lines) if om_lines else "  (none available)"
    metar_block = "\n".join(metar_lines) if metar_lines else "  (none available)"
    bracket_block = "\n".join(bracket_lines) if bracket_lines else "  (none available)"
    hist_block = "\n".join(hist_lines) if hist_lines else "  (no history)"
    summary_str = f"{summary_high}{U} — {summary_high_note}" if summary_high is not None else "not yet available"

    # Compute observed max from wu_obs to surface explicitly — WU summary high often lags
    obs_temps = [o.get("temperature") for o in wu_obs if o.get("temperature") is not None]
    observed_max = max(obs_temps) if obs_temps else None
    if observed_max is not None and summary_high is not None and observed_max > summary_high:
        observed_max_note = f"⚠ OBSERVED MAX {observed_max}{U} already EXCEEDS the summary high ({summary_high}{U}). The summary high is stale — use the observed max as your temperature floor."
    elif observed_max is not None:
        observed_max_note = f"Observed max so far: {observed_max}{U} (consistent with or below summary high)"
    else:
        observed_max_note = "No observations yet today"

    # Identify market leader (highest YES-priced bracket) for late-day warnings
    market_leader_line = ""
    if brackets:
        leader = max(brackets, key=lambda b: float(b.get("yes_price") or 0))
        leader_price = float(leader.get("yes_price") or 0)
        if leader_price >= 0.60:
            market_leader_line = f"\n⚠ MARKET LEADER: '{leader['label']}' is priced at {round(leader_price*100)}% YES. This is a very strong market consensus — override it only if you have concrete observed data (not just forecasts) directly contradicting it."

    return f"""You are a weather-temperature Polymarket bracket trading expert. Analyze multi-source weather data and recommend bracket trades.

## City: {city_name}
Date: {today}  |  Current local hour: {current_local_hour:02d}:00  |  Unit: {unit_label}
Day status: {day_status}

## WU Observed Temperatures (today, actual station readings)
{wu_obs_block}

## Key temperature summary
{observed_max_note}

## WU Summary High (official daily high — Polymarket resolves against this value at day's end)
{summary_str}

## WU Hourly Forecast
{wu_fc_block}

## Open-Meteo Hourly (actual + forecast marked "fc")
{om_block}

## METAR / ASOS Readings (aviation weather, independent source)
{metar_block}

## Recent Peak History (last 14 days — use to judge typical daily range and peak timing)
{hist_block}
City avg peak hour: {avg_peak_hour:.1f}:00

## Polymarket Brackets & Current Prices
{bracket_block}
Note: Market prices reflect the collective probability estimate of all traders. Large divergence between your estimate and market pricing requires strong justification — the market often knows things you don't (e.g. forecast model updates, recent station data).{market_leader_line}

---

Your task:
1. Estimate today's FINAL WU summary high (at day's end, not now).
2. Determine which bracket that falls into.
3. Recommend BUY_YES / BUY_NO / SKIP for EVERY bracket.

Critical reasoning rules:
- If day status is EARLY or NEAR PEAK: the WU summary high is NOT the final value. Forecasts showing higher temps are credible — use them. The current summary high is just a floor.
- If day status is LATE: use the OBSERVED MAX as your temperature floor (not the summary high — it often lags real readings by 30–60 minutes). If the observed max exceeds the summary high, trust the observed max.
- NEVER anchor your peak estimate purely on the current summary high when it's early in the day and forecasts show higher temps coming.
- MARKET RESPECT: The market price is a very strong prior. If a bracket is priced ≥60%, treat it as highly credible. You may only bet YES on a different bracket if you have concrete observed temperatures (actual WU readings or METAR) — not just forecasts — that directly contradict the market.
- In LATE day status: forecasts alone are NOT sufficient evidence to override a strongly-priced market bracket. Prefer SKIP over making contrarian bets based on forecasts.
- BUY_YES only on the bracket you are genuinely confident will win AND the price gives real edge (your confidence substantially exceeds the market price).
- BUY_NO on brackets clearly excluded by evidence (price > 0.03, meaning there's value to capture).
- SKIP when evidence is ambiguous or your edge is small.
- Lower your confidence when: it is early in the day, forecasts disagree, historical peaks vary widely for this city.

Respond ONLY with this JSON:
{{
  "analysis": "3-5 sentence narrative: current situation, what the forecasts say, whether the day is early/late, and why you chose your estimate",
  "peak_temp_estimate": <number or null>,
  "peak_confidence": <0.0-1.0>,
  "recommendations": [
    {{
      "bracket": "<exact bracket label>",
      "action": "BUY_YES" | "BUY_NO" | "SKIP",
      "confidence": <0.0-1.0>,
      "reasoning": "one sentence"
    }}
  ]
}}

Include ALL brackets. Order by conviction (highest first).
"""


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
    """Call Claude to analyze all weather data and return bracket recommendations."""
    import anthropic

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

    client = anthropic.AsyncAnthropic(api_key=api_key)
    message = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text
    log.info("llm_response_received", city=city_name, chars=len(raw_text))

    # Parse JSON — strip markdown fences if present
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        json_text = json_text.split("```")[1]
        if json_text.startswith("json"):
            json_text = json_text[4:]
    json_text = json_text.strip()

    parsed = json.loads(json_text)

    # Map LLM bracket labels back to DB bracket objects + build recommendations
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
        model="claude-haiku-4-5-20251001",
        raw_json=parsed,
    )
