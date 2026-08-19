import json
import re
import time

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def parse_duration(text):
    """Parse Groq's rate-limit reset strings, e.g. '547ms', '2.5s', '1m30s', '10h48m0s'."""
    if text is None:
        return 0.0
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", text):
        value = float(value)
        total += value / 1000 if unit == "ms" else value * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def call_groq(prompt_text, api_key, model=DEFAULT_MODEL, retries=8):
    content, _ = call_groq_with_ratelimit(prompt_text, api_key, model, retries)
    return content


def call_groq_with_ratelimit(prompt_text, api_key, model=DEFAULT_MODEL, retries=8):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.2,
    }
    for attempt in range(retries):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            ratelimit = {
                "remaining_tokens": int(resp.headers["x-ratelimit-remaining-tokens"]) if "x-ratelimit-remaining-tokens" in resp.headers else None,
                "reset_tokens_s": parse_duration(resp.headers.get("x-ratelimit-reset-tokens")),
            }
            return content, ratelimit
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else parse_duration(resp.headers.get("x-ratelimit-reset-tokens")) or min(2 ** attempt, 60)
            time.sleep(wait + 0.5)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Groq API failed after {retries} retries (rate limited)")


def parse_json_response(raw_text):
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        obj = json.loads(match.group(0)) if match else json.loads(raw_text)
        return obj, "valid"
    except (json.JSONDecodeError, AttributeError):
        return {}, "invalid"
