"""
llm_decision.py

Wires persona prompts with fundamentals and price context, and executes trading decisions
via either:
1. Free & fast deterministic Mock/Rule Engine (runs offline, zero cost, instant backtests)
2. Google Gemini API (gemini-2.5-flash / gemini-1.5-flash)
3. OpenAI API (gpt-4o-mini / gpt-4o)
"""

import json
import os
import re
import requests

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt_template(persona_name: str) -> str:
    """Load the system/persona prompt template from prompts/ directory."""
    filename = f"{persona_name}.txt"
    filepath = os.path.join(PROMPTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return f"You are a trading agent following persona: {persona_name}."


def sanitize_json_response(raw_text: str) -> dict:
    """Extract and parse valid JSON from LLM response text."""
    if not raw_text:
        return {"action": "hold", "confidence": 0.5, "reasoning": "Empty LLM response; holding position."}

    # Remove markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex extraction
        action_match = re.search(r'"action"\s*:\s*"?(buy|sell|hold)"?', raw_text, re.IGNORECASE)
        conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', raw_text)
        reas_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw_text)

        action = action_match.group(1).lower() if action_match else "hold"
        conf = float(conf_match.group(1)) if conf_match else 0.5
        reasoning = reas_match.group(1) if reas_match else raw_text[:200]
        data = {"action": action, "confidence": conf, "reasoning": reasoning}

    # Normalize fields
    action = str(data.get("action", "hold")).lower().strip()
    if action not in ["buy", "sell", "hold"]:
        action = "hold"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    reasoning = str(data.get("reasoning", "")).strip() or "Standard persona rule alignment."

    return {
        "action": action,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
    }


def call_mock_engine(persona_name: str, ticker: str, fundamentals_data: dict, price_ctx: dict, dad_analysis: dict = None) -> dict:
    """
    Intelligent, deterministic rule engine that faithfully implements each persona's
    trading logic without requiring external network calls or API credits.
    """
    curr_price = price_ctx.get("current_price", 0.0)
    rsi = price_ctx.get("rsi_14", 50.0)
    ret_5d = price_ctx.get("ret_5d", 0.0)
    ret_20d = price_ctx.get("ret_20d", 0.0)
    sma_20 = price_ctx.get("sma_20", curr_price)
    sma_50 = price_ctx.get("sma_50", curr_price)
    volatility = price_ctx.get("volatility_20d", 18.0)
    vol_ratio = price_ctx.get("volume_ratio", 1.0)
    pct_from_high = price_ctx.get("pct_from_high", 0.0)
    trailing_pe = fundamentals_data.get("trailing_pe") if fundamentals_data else None

    if persona_name == "dad_hybrid":
        # Dad's 14 rules composite evaluation
        if dad_analysis:
            score = dad_analysis.get("overall_score", 70)
            fails = dad_analysis.get("fail_count", 0)
            scores = dad_analysis.get("scores", {})

            # Key rules check
            r1_pass = scores.get("Rule 1 & 2: P/E Screen & Leader Moat", {}).get("status") == "PASS"
            r6_pass = scores.get("Rule 5 & 6: Short-Term Debt vs Working Capital", {}).get("status") == "PASS"
            r13_pass = scores.get("Rule 13: Reserves & Retained Earnings Growth", {}).get("status") == "PASS"

            if fails >= 2 or score < 45:
                return {
                    "action": "sell",
                    "confidence": 0.85,
                    "reasoning": f"Failed {fails} core rules (Score: {score}/100); deteriorating balance sheet discipline violates capital preservation principles.",
                }
            elif score >= 75 and r6_pass and r13_pass:
                return {
                    "action": "buy",
                    "confidence": round(score / 100.0, 2),
                    "reasoning": f"Passed {dad_analysis.get('pass_count', 0)}/10 rules (Score: {score}/100). Low debt relative to working capital, solid margins, and expanding reserves make this an ideal long-term holding.",
                }
            else:
                return {
                    "action": "hold",
                    "confidence": 0.65,
                    "reasoning": f"Quality score {score}/100 meets baseline criteria but lacks deep valuation discount for fresh allocation; maintaining existing position.",
                }
        else:
            return {"action": "hold", "confidence": 0.5, "reasoning": "Fundamental statement data not yet loaded; holding."}

    elif persona_name == "momentum_trader":
        # Pure technical momentum: buy strong uptrends + high volume, sell reversals
        if rsi >= 53 and curr_price > sma_20 and ret_5d > 0.2 and vol_ratio >= 0.9:
            conf = min(0.95, 0.60 + (rsi - 50) / 100 + (0.1 if vol_ratio > 1.2 else 0))
            return {
                "action": "buy",
                "confidence": round(conf, 2),
                "reasoning": f"Strong upside momentum: Price ₹{curr_price:,.1f} > SMA20 ₹{sma_20:,.1f}, RSI at {rsi:.1f}, and volume {vol_ratio:.2f}x average.",
            }
        elif rsi < 44 or (curr_price < sma_20 and ret_5d < -1.5):
            return {
                "action": "sell",
                "confidence": 0.82,
                "reasoning": f"Momentum breaking down: Price dropped below SMA20 with RSI cooling to {rsi:.1f} and 5d return {ret_5d:+.1f}%.",
            }
        else:
            return {
                "action": "hold",
                "confidence": 0.55,
                "reasoning": f"Momentum neutral: RSI {rsi:.1f} in middle band, waiting for decisive breakout confirmation.",
            }

    elif persona_name == "value_investor":
        # Undervalued, quality, low P/E, buy dips
        pe = trailing_pe or 22.0
        if pe <= 24 and pct_from_high < -6.0:
            return {
                "action": "buy",
                "confidence": 0.84,
                "reasoning": f"Attractive margin of safety: P/E is {pe:.1f}x and stock is trading {pct_from_high:+.1f}% below 52w highs.",
            }
        elif pe > 38:
            return {
                "action": "sell",
                "confidence": 0.78,
                "reasoning": f"Valuation stretched: P/E at {pe:.1f}x offers no adequate margin of safety.",
            }
        else:
            return {
                "action": "hold",
                "confidence": 0.60,
                "reasoning": f"Fair valuation at {pe:.1f}x P/E; comfortable holding long-term position.",
            }

    elif persona_name == "risk_averse":
        # Capital preservation, avoid volatility, require high confidence
        if volatility > 26.0 or curr_price < sma_50:
            return {
                "action": "sell",
                "confidence": 0.88,
                "reasoning": f"Elevated annualized volatility ({volatility:.1f}%) and trading below 50-day SMA; prioritizing capital preservation in cash.",
            }
        elif volatility <= 18.0 and curr_price >= sma_20 and 46 <= rsi <= 62:
            return {
                "action": "buy",
                "confidence": 0.82,
                "reasoning": f"Low volatility ({volatility:.1f}%) and stable trend near SMA support satisfy strict risk parameters.",
            }
        else:
            return {
                "action": "hold",
                "confidence": 0.70,
                "reasoning": "Holding cash/position until low-risk, high-probability entry presents itself.",
            }

    elif persona_name == "news_sentiment":
        # Reacts to price shocks and volume spikes as sentiment proxies
        if ret_1d := price_ctx.get("ret_1d", 0.0):
            if ret_1d >= 1.8 or (ret_5d >= 3.5 and vol_ratio > 1.3):
                return {
                    "action": "buy",
                    "confidence": 0.80,
                    "reasoning": f"Surge in positive sentiment: {ret_1d:+.1f}% single-day advance with {vol_ratio:.1f}x volume expansion signaling institutional accumulation.",
                }
            elif ret_1d <= -1.8:
                return {
                    "action": "sell",
                    "confidence": 0.78,
                    "reasoning": f"Negative headline reaction: sharp {ret_1d:+.1f}% decline indicates adverse sentiment flow.",
                }
        return {
            "action": "hold",
            "confidence": 0.50,
            "reasoning": "No headline catalysts or sentiment divergence detected today.",
        }

    return {"action": "hold", "confidence": 0.5, "reasoning": "Holding position."}


def call_gemini_api(prompt_text: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
    """Call Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "")
    return ""


def call_openai_api(prompt_text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI Chat Completions REST API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional trading system returning ONLY valid JSON."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


def get_trading_decision(
    persona_name: str,
    ticker: str,
    date_str: str,
    price_ctx: dict,
    fundamentals_ctx_text: str = "",
    fundamentals_data: dict = None,
    dad_analysis: dict = None,
    provider: str = "mock",
    api_key: str = None,
) -> dict:
    """
    Main entry point for getting a trading decision.
    Routes to Mock Engine or Live API (Gemini/OpenAI) based on configuration.
    """
    if provider == "mock" or not api_key:
        return call_mock_engine(
            persona_name=persona_name,
            ticker=ticker,
            fundamentals_data=fundamentals_data or {},
            price_ctx=price_ctx,
            dad_analysis=dad_analysis,
        )

    # Prepare prompt for LLM
    persona_prompt = load_prompt_template(persona_name)
    user_prompt = f"""{persona_prompt}

EVALUATION REQUEST:
Ticker: {ticker}
Date: {date_str}

{price_ctx.get('context_text', '')}

{fundamentals_ctx_text if fundamentals_ctx_text else 'No fundamental statement context provided.'}

Based strictly on your persona instructions above, evaluate {ticker} and respond ONLY in valid JSON:
{{"action": "buy" | "sell" | "hold", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    try:
        if provider == "gemini":
            raw_output = call_gemini_api(user_prompt, api_key)
        elif provider == "openai":
            raw_output = call_openai_api(user_prompt, api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return sanitize_json_response(raw_output)

    except Exception as e:
        print(f"  [Warning] LLM call failed ({provider}): {e}. Falling back to rule engine.")
        # Graceful fallback to mock engine if network/API limits hit
        decision = call_mock_engine(
            persona_name=persona_name,
            ticker=ticker,
            fundamentals_data=fundamentals_data or {},
            price_ctx=price_ctx,
            dad_analysis=dad_analysis,
        )
        decision["reasoning"] += f" (Note: LLM fallback triggered: {str(e)[:60]})"
        return decision
