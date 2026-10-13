"""OpenRouter chat client: paid-call lock, client-side throttle, retry/backoff,
disk cache, per-call usage and a spend budget.

The API key is only ever placed in the Authorization header of a request. It is
not stored on any record, error message or log line.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Protocol

import httpx

from .config import Settings

log = logging.getLogger("atlandex_agent.llm")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class PaidCallBlocked(RuntimeError):
    """A non-:free model was requested while ALLOW_PAID is false."""


class BudgetExceeded(RuntimeError):
    """A paid call would push cumulative paid spend past BUDGET_USD."""


class LLMUnavailable(RuntimeError):
    """The model could not be reached after all retries."""


class LLMError(RuntimeError):
    """A non-retryable error response from the provider."""


class CallBudgetExceeded(RuntimeError):
    """A run tried to make more LLM calls than Settings.max_llm_calls_per_run allows.

    A safety net, not a control path: the graph asks `CallBudget.left()` before optional calls and degrades
    instead, so this should never be raised by a graph that behaves.
    """


@dataclass
class CallRecord:
    role: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_sec: float = 0.0
    cached: bool = False
    paid: bool = False
    fallback_from: str | None = None
    attempts: int = 1


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]]  # [{"id", "name", "args"}]
    record: CallRecord
    finish_reason: str | None = None  # "stop", "tool_calls", "length" (cut off at max_tokens), ...


class LLM(Protocol):
    """What the graph and the judge need. Tests provide a fake with this shape."""

    model: str
    calls: list[CallRecord]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        role: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


def is_free(model: str) -> bool:
    return model.endswith(":free")


class Throttle:
    """Sliding-window limiter: at most `rpm` requests in any 60 seconds."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.rpm = max(1, rpm)
        self._clock, self._sleep = clock, sleep
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return
                await self._sleep(60.0 - (now - self._stamps[0]) + 0.05)


class OpenRouterClient:
    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.s = settings
        self._http = http or httpx.AsyncClient(timeout=settings.request_timeout_sec)
        self._sleep = sleep
        self._clock = clock
        self.throttle = Throttle(settings.requests_per_minute, clock, sleep)
        self.calls: list[CallRecord] = []
        self.paid_spend_usd = 0.0
        self._prices: dict[str, tuple[float, float]] | None = None

    def bind(self, model: str) -> "BoundLLM":
        return BoundLLM(self, model)

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- guards ---------------------------------------------------------
    def _guard(self, model: str) -> None:
        if not is_free(model) and not self.s.allow_paid:
            raise PaidCallBlocked(
                f"Refusing to call paid model '{model}': ALLOW_PAID is false. "
                "Use a ':free' model or ask the owner to set ALLOW_PAID=true."
            )

    async def _price(self, model: str) -> tuple[float, float]:
        """(input, output) USD per token, from /models. Cached for the process."""
        if self._prices is None:
            r = await self._http.get(
                f"{self.s.base_url}/models",
                headers={"Authorization": f"Bearer {self.s.openrouter_api_key}"},
            )
            r.raise_for_status()
            self._prices = {
                m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
                for m in r.json()["data"]
            }
        if model not in self._prices:
            raise LLMError(f"No price known for '{model}'; refusing to spend blind.")
        return self._prices[model]

    async def _check_budget(self, model: str, messages: list, max_tokens: int) -> None:
        p_in, p_out = await self._price(model)
        prompt_tokens_upper = len(json.dumps(messages)) / 2.5  # deliberately pessimistic
        worst = prompt_tokens_upper * p_in + max_tokens * p_out
        if self.paid_spend_usd + worst > self.s.budget_usd:
            raise BudgetExceeded(
                f"Paid call to '{model}' could cost up to ${worst:.5f}; spent "
                f"${self.paid_spend_usd:.5f} of BUDGET_USD={self.s.budget_usd}."
            )

    async def key_status(self) -> dict[str, int]:
        """The key's free-model request allowance today, from GET /key: {"daily_limit", "used_today", "remaining"}.

        Used by the eval's free-request guard. It is not a chat request, so it is not throttled or cached, and the
        key only goes into the Authorization header.
        """
        r = await self._http.get(f"{self.s.base_url}/key",
                                 headers={"Authorization": f"Bearer {self.s.openrouter_api_key}"})
        r.raise_for_status()
        free = ((_safe_json(r).get("data") or {}).get("free_model_daily_requests")) or {}
        try:
            return {"daily_limit": int(free["limit"]), "used_today": int(free["used"]),
                    "remaining": int(free["remaining"])}
        except (KeyError, TypeError, ValueError):
            raise LLMError("GET /key did not report free_model_daily_requests") from None

    # -- cache ----------------------------------------------------------
    def _cache_path(self, key: str):
        return self.s.cache_dir / f"{key}.json"

    @staticmethod
    def _key(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # -- request --------------------------------------------------------
    async def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """POST with throttle and retry. Returns (json, attempts)."""
        last = "no response"
        for attempt in range(1, self.s.max_retries + 1):
            await self.throttle.wait()
            retry_after = None
            try:
                r = await self._http.post(
                    f"{self.s.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.s.openrouter_api_key}"},
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"transport error: {type(e).__name__}"
            else:
                body = _safe_json(r)
                err = body.get("error") if isinstance(body, dict) else None
                code = r.status_code if r.status_code != 200 else (err or {}).get("code", 200)
                if r.status_code == 200 and not err and body.get("choices"):
                    return body, attempt
                if isinstance(code, int) and code in RETRY_STATUS or (r.status_code == 200 and err and not body.get("choices")):
                    last = f"HTTP {code}: {_err_msg(body)}"
                    retry_after = _retry_after(r)
                else:
                    raise LLMError(f"HTTP {r.status_code}: {_err_msg(body)}")
            delay = retry_after if retry_after is not None else min(2.0 ** attempt, 60.0)
            log.warning("LLM attempt %d/%d failed (%s); retrying in %.0fs", attempt, self.s.max_retries, last, delay)
            await self._sleep(delay)
        raise LLMUnavailable(f"'{payload['model']}' unavailable after {self.s.max_retries} attempts ({last})")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        role: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """`max_tokens` overrides Settings.max_tokens for this call (the judge needs more, agent turns less)."""
        self._guard(model)
        max_tokens = max_tokens or self.s.max_tokens
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.s.temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        key = self._key({k: v for k, v in payload.items() if k != "usage"})
        if self.s.cache_enabled and self._cache_path(key).exists():
            hit = json.loads(self._cache_path(key).read_text(encoding="utf-8"))
            rec = CallRecord(role=role, model=model, cached=True, paid=not is_free(model),
                             prompt_tokens=hit["prompt_tokens"], completion_tokens=hit["completion_tokens"])
            self.calls.append(rec)
            return LLMResponse(hit["content"], hit["tool_calls"], rec, hit.get("finish_reason"))

        fallback_from = None
        t0 = self._clock()
        try:
            if not is_free(model):
                await self._check_budget(model, messages, max_tokens)
            body, attempts = await self._post(payload)
        except LLMUnavailable:
            fb = self.s.paid_fallback_for(model)
            if not (self.s.allow_paid and fb):
                raise  # ALLOW_PAID=false: report the failure, never fall back silently
            log.warning("Free model '%s' unavailable; falling back to paid '%s'", model, fb)
            fallback_from, model = model, fb
            payload["model"] = model
            await self._check_budget(model, messages, max_tokens)
            body, attempts = await self._post(payload)

        msg = body["choices"][0]["message"]
        finish_reason = body["choices"][0].get("finish_reason")
        usage = body.get("usage") or {}
        paid = not is_free(model)
        cost = float(usage.get("cost") or 0.0) if paid else 0.0
        self.paid_spend_usd += cost
        rec = CallRecord(
            role=role, model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=cost, latency_sec=self._clock() - t0, paid=paid,
            fallback_from=fallback_from, attempts=attempts,
        )
        self.calls.append(rec)
        calls = [_parse_tool_call(tc) for tc in (msg.get("tool_calls") or [])]
        content = msg.get("content") or ""
        if self.s.cache_enabled:
            self.s.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(json.dumps({
                "content": content, "tool_calls": calls, "finish_reason": finish_reason,
                "prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens,
            }), encoding="utf-8")
        return LLMResponse(content, calls, rec, finish_reason)


class BoundLLM:
    """An OpenRouterClient pinned to one model; satisfies the LLM protocol."""

    def __init__(self, client: OpenRouterClient, model: str):
        self._client, self.model = client, model

    @property
    def calls(self) -> list[CallRecord]:
        return self._client.calls

    async def chat(self, messages, *, tools=None, role: str = "", max_tokens: int | None = None) -> LLMResponse:
        return await self._client.chat(self.model, messages, tools=tools, role=role, max_tokens=max_tokens)


class CallBudget:
    """Counts the LLM calls of one run and refuses the call after the cap.

    Wraps any LLM (see the protocol above). It counts logical calls, not HTTP attempts: retries after a 429 or
    5xx happen inside the client and are reported separately (CallRecord.attempts). A call that fails still
    counts, because it was still a request.
    """

    def __init__(self, inner: LLM, cap: int):
        self._inner, self.cap, self.used = inner, cap, 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def calls(self) -> list[CallRecord]:
        return self._inner.calls

    def left(self) -> int:
        return self.cap - self.used

    async def chat(self, messages, *, tools=None, role: str = "", max_tokens: int | None = None) -> LLMResponse:
        if self.used >= self.cap:
            raise CallBudgetExceeded(f"{self.cap} LLM calls already made in this run (role '{role}' refused)")
        self.used += 1
        return await self._inner.chat(messages, tools=tools, role=role, max_tokens=max_tokens)


def _safe_json(r: httpx.Response) -> dict[str, Any]:
    try:
        b = r.json()
        return b if isinstance(b, dict) else {}
    except ValueError:
        return {}


def _err_msg(body: dict[str, Any]) -> str:
    e = body.get("error") if body else None
    if isinstance(e, dict):
        return str(e.get("message", e))[:300]
    return str(e)[:300] if e else "no error body"


def _retry_after(r: httpx.Response) -> float | None:
    v = r.headers.get("retry-after")
    try:
        return min(float(v), 120.0) if v else None
    except ValueError:
        return None


def _parse_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function", {})
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except ValueError:
        args = {"_unparsed": raw}
    return {"id": tc.get("id") or fn.get("name", "call"), "name": fn.get("name", ""), "args": args}


def usage_summary(records: list[CallRecord]) -> dict[str, Any]:
    return {
        "llm_calls": len(records),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "cost_usd": round(sum(r.cost_usd for r in records), 6),
        "cached_calls": sum(r.cached for r in records),
        "fallbacks": sum(1 for r in records if r.fallback_from),
    }


def records_to_dicts(records: list[CallRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]
