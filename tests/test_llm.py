import asyncio
import json

import httpx
import pytest

from atlandex_agent.config import Settings
from atlandex_agent.llm import (BudgetExceeded, LLMUnavailable, OpenRouterClient, PaidCallBlocked, Throttle)

FREE, PAID = "a/m:free", "a/m"
KEY = "sk-or-v1-TESTKEY-NOT-REAL"


def ok_body(content="hi", cost=0.0):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost}}


def make(tmp_path, handler, **kw):
    s = Settings(openrouter_api_key=KEY, cache_dir=tmp_path / "c", max_retries=3,
                 model_small=FREE, paid_small=PAID, **kw)
    slept: list[float] = []

    async def fake_sleep(d):
        slept.append(d)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterClient(s, http=http, sleep=fake_sleep), slept


def chat(client, model=FREE, text="hello"):
    return asyncio.run(client.chat(model, [{"role": "user", "content": text}], role="t"))


def test_paid_model_is_refused_without_allow_paid_and_no_request_is_sent(tmp_path):
    sent = []
    client, _ = make(tmp_path, lambda req: sent.append(req) or httpx.Response(200, json=ok_body()))
    with pytest.raises(PaidCallBlocked):
        chat(client, PAID)
    assert sent == []


def test_free_call_records_usage_and_costs_nothing(tmp_path):
    client, _ = make(tmp_path, lambda req: httpx.Response(200, json=ok_body("hello", cost=0.5)))
    r = chat(client)
    assert r.content == "hello" and r.record.prompt_tokens == 10 and r.record.cost_usd == 0.0
    assert client.paid_spend_usd == 0.0


def test_retries_on_429_with_backoff_then_succeeds(tmp_path):
    seq = [httpx.Response(429, json={"error": {"message": "rate"}}),
           httpx.Response(429, json={"error": {"message": "rate"}}), httpx.Response(200, json=ok_body())]
    client, slept = make(tmp_path, lambda req: seq.pop(0))
    r = chat(client)
    assert r.record.attempts == 3 and len(slept) == 2 and slept[1] > slept[0]


def test_free_outage_without_allow_paid_reports_failure_and_never_falls_back(tmp_path):
    urls = []

    def handler(req):
        urls.append(json.loads(req.content)["model"])
        return httpx.Response(503, json={"error": {"message": "down"}})

    client, _ = make(tmp_path, handler)
    with pytest.raises(LLMUnavailable):
        chat(client)
    assert set(urls) == {FREE}


def test_free_outage_with_allow_paid_falls_back_and_logs_it(tmp_path):
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": PAID, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]})
        if json.loads(req.content)["model"] == FREE:
            return httpx.Response(503, json={"error": {"message": "down"}})
        return httpx.Response(200, json=ok_body(cost=0.0001))

    client, _ = make(tmp_path, handler, allow_paid=True, budget_usd=1.0)
    r = chat(client)
    assert r.record.model == PAID and r.record.fallback_from == FREE and r.record.paid
    assert client.paid_spend_usd == pytest.approx(0.0001)


def test_budget_blocks_a_paid_call_before_it_is_sent(tmp_path):
    posts = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": PAID, "pricing": {"prompt": "0.001", "completion": "0.001"}}]})
        posts.append(1)
        return httpx.Response(200, json=ok_body())

    client, _ = make(tmp_path, handler, allow_paid=True, budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        chat(client, PAID)
    assert posts == []


def test_cache_hit_makes_no_request_and_flags_the_record(tmp_path):
    n = []
    client, _ = make(tmp_path, lambda req: n.append(1) or httpx.Response(200, json=ok_body("cached text")))
    chat(client)
    r = chat(client)
    assert len(n) == 1 and r.content == "cached text" and r.record.cached


def test_cache_can_be_disabled(tmp_path):
    n = []
    client, _ = make(tmp_path, lambda req: n.append(1) or httpx.Response(200, json=ok_body()), cache_enabled=False)
    chat(client)
    chat(client)
    assert len(n) == 2


def test_tool_calls_are_parsed(tmp_path):
    body = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "function": {"name": "find_videos", "arguments": '{"term": "rl"}'}}]}}], "usage": {}}
    client, _ = make(tmp_path, lambda req: httpx.Response(200, json=body))
    r = chat(client)
    assert r.tool_calls == [{"id": "x", "name": "find_videos", "args": {"term": "rl"}}]


def test_throttle_spaces_requests_to_the_limit():
    now = [0.0]
    slept = []

    async def sleep(d):
        slept.append(d)
        now[0] += d

    t = Throttle(2, clock=lambda: now[0], sleep=sleep)

    async def go():
        for _ in range(3):
            await t.wait()

    asyncio.run(go())
    assert len(slept) == 1 and slept[0] == pytest.approx(60.05)


def test_the_api_key_never_appears_in_repr_or_errors(tmp_path):
    s = Settings(openrouter_api_key=KEY, atlandex_api_url="https://secret.example/api")
    assert KEY not in repr(s) and "secret.example" not in repr(s)
    client, _ = make(tmp_path, lambda req: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(Exception) as e:
        chat(client)
    assert KEY not in str(e.value)
