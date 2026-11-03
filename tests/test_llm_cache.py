import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_cache


def test_second_identical_call_hits_the_cache(tmp_path):
    path = str(tmp_path / "calls.jsonl")
    calls = []

    def fake_api():
        calls.append(1)
        return '{"action": "buy"}', {"total_tokens": 42}

    a = llm_cache.cached_call("gemini", "m", "prompt", fake_api, path=path, ticker="TCS.NS")
    b = llm_cache.cached_call("gemini", "m", "prompt", fake_api, path=path)
    assert a == b == '{"action": "buy"}'
    assert len(calls) == 1

    rec = json.loads(open(path).read().strip())
    assert rec["prompt"] == "prompt" and rec["usage"]["total_tokens"] == 42 and rec["ticker"] == "TCS.NS"


def test_different_prompt_or_model_is_a_new_call(tmp_path):
    path = str(tmp_path / "calls.jsonl")
    n = []
    api = lambda: (n.append(1), "x")[1]
    llm_cache.cached_call("gemini", "m1", "p", api, path=path)
    llm_cache.cached_call("gemini", "m2", "p", api, path=path)
    llm_cache.cached_call("gemini", "m1", "q", api, path=path)
    assert len(n) == 3


def test_cache_survives_a_restart(tmp_path):
    path = str(tmp_path / "calls.jsonl")
    llm_cache.cached_call("openai", "m", "p", lambda: "first", path=path)
    llm_cache._index = None                       # simulate a fresh Python process
    assert llm_cache.cached_call("openai", "m", "p", lambda: "SHOULD NOT RUN", path=path) == "first"
