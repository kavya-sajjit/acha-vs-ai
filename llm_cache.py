"""
llm_cache.py

Every live LLM call goes through `cached_call`. The prompt, model, raw response,
latency and token usage are appended to logs/llm_calls.jsonl, one JSON object
per line. If the exact same (provider, model, prompt) was asked before, the
saved response is returned and no API call is made.

Why this matters for the experiment:
  * Reproducibility: a backtest re-run gives identical decisions, even though the
    API itself is not deterministic.
  * Cost: re-running charts or fixing a plotting bug costs nothing.
  * Auditability: anyone can read exactly what each model was asked and answered.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone

CACHE_PATH = os.environ.get("ACHA_LLM_CACHE", os.path.join("logs", "llm_calls.jsonl"))

_index = None  # key -> record, loaded lazily from the JSONL file


def cache_key(provider: str, model: str, prompt: str) -> str:
    return hashlib.sha256(f"{provider}\x00{model}\x00{prompt}".encode("utf-8")).hexdigest()


def _load_index(path: str) -> dict:
    index = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    index[rec["key"]] = rec
    return index


def cached_call(provider: str, model: str, prompt: str, call_fn, path: str = None, **meta) -> str:
    """Return the model's raw text response, from cache if possible.

    call_fn: zero-argument function that performs the real API call and returns
             either the response text, or a (text, usage_dict) tuple.
    meta:    extra fields to store with the record (e.g. ticker, date, persona).
    """
    global _index
    path = path or CACHE_PATH
    if _index is None or _index.get("__path__") != path:
        _index = _load_index(path)
        _index["__path__"] = path

    key = cache_key(provider, model, prompt)
    if key in _index:
        return _index[key]["response"]

    t0 = time.time()
    result = call_fn()
    text, usage = result if isinstance(result, tuple) else (result, {})
    record = {
        "key": key,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "response": text,
        "usage": usage,
        "latency_s": round(time.time() - t0, 3),
        "called_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _index[key] = record
    return text
