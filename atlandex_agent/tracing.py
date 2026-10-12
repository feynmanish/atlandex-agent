"""Optional Langfuse tracing.

Tracing is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and the
`langfuse` package is installed: `pip install -e ".[tracing]"`). Otherwise `tracer_for`
returns a no-op tracer and nothing here imports langfuse, so the agent, the eval harness
and the offline tests behave exactly as before.

One agent run is one trace:

    atlandex-agent                  root span: question in, answer + summary out
      LangGraph                     LangChain callback handler (langfuse.langchain)
        router / agent / tools / answer / citation_check   one span per graph node
          llm:<role>                generation: model, messages, output, tokens, cost
          <tool name>               tool span: arguments and result

The callback handler only sees LangChain runnables. The LLM client (httpx to OpenRouter) and
the MCP tools are not runnables, so `TracedLLM` and `TracedToolBox` record those themselves.
They nest under the active node span through OpenTelemetry context.

Keys are read from Settings and passed to the Langfuse client; they are never logged.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .config import Settings

log = logging.getLogger("atlandex_agent.tracing")

MAX_IO_CHARS = 8000  # tool results and prompts are cut here in the trace; the run itself is unaffected


@dataclass
class TraceInfo:
    """Labels attached to a trace so runs can be filtered in Langfuse."""

    run_id: str = ""  # eval results directory name; groups all traces of one eval run (Langfuse session)
    query_id: str = ""  # dataset record id
    model_alias: str = ""  # small | large | explicit id
    extra_tags: list[str] = field(default_factory=list)


def _clip(text: Any, limit: int = MAX_IO_CHARS) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + f"... [{len(text) - limit} more chars not shown]"
    return text


class TracedLLM:
    """Wraps any LLM (see llm.LLM) and records each chat() call as a Langfuse generation."""

    def __init__(self, inner: Any, client: Any):
        self._inner, self._client = inner, client

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def calls(self):
        return self._inner.calls

    async def chat(self, messages, *, tools=None, role: str = "", max_tokens=None):
        with self._client.start_as_current_observation(
            as_type="generation", name=f"llm:{role or 'chat'}", model=self._inner.model,
            input=[{**m, "content": _clip(m.get("content"))} for m in messages],
            metadata={"role": role, "tools_offered": [t["function"]["name"] for t in tools or []]},
        ) as gen:
            try:
                r = await self._inner.chat(messages, tools=tools, role=role, max_tokens=max_tokens)
            except Exception as e:
                gen.update(level="ERROR", status_message=f"{type(e).__name__}: {str(e)[:300]}")
                raise
            rec = r.record
            gen.update(
                model=rec.model,  # differs from the requested one after a paid fallback
                output=r.content if not r.tool_calls else {"content": r.content, "tool_calls": r.tool_calls},
                usage_details={"input": rec.prompt_tokens, "output": rec.completion_tokens,
                               "total": rec.prompt_tokens + rec.completion_tokens},
                cost_details={"total": rec.cost_usd},
                metadata={"role": role, "cached": rec.cached, "paid": rec.paid, "attempts": rec.attempts,
                          "latency_sec": round(rec.latency_sec, 3), "fallback_from": rec.fallback_from},
            )
            return r


class TracedToolBox:
    """Wraps a ToolBox and records each call as a Langfuse tool span."""

    def __init__(self, inner: Any, client: Any):
        self._inner, self._client = inner, client

    def openai_specs(self):
        return self._inner.openai_specs()

    async def call(self, name: str, args: dict[str, Any]) -> str:
        from langchain_core.runnables.config import var_child_runnable_config

        with self._client.start_as_current_observation(as_type="tool", name=name, input=args) as span:
            # MCP tools are LangChain tools and would inherit the graph's callbacks, so the handler would
            # record the call a second time. Hide the inherited config for the duration of the call.
            token = var_child_runnable_config.set(None)
            try:
                out = await self._inner.call(name, args)
            finally:
                var_child_runnable_config.reset(token)
            failed = out.startswith("Error")
            span.update(output=_clip(out), level="ERROR" if failed else None,
                        status_message=out[:200] if failed else None)
            return out


class NullRun:
    """What a run looks like with tracing off: no callbacks, wrappers return their input."""

    trace_id: str | None = None
    callbacks: list[Any] = []
    config: dict[str, Any] = {}

    def llm(self, llm):
        return llm

    def tools(self, tools):
        return tools

    def finish(self, output: dict[str, Any]) -> None:
        pass


class LiveRun(NullRun):
    def __init__(self, tracer: "LangfuseTracer", root: Any, handler: Any, config: dict[str, Any]):
        self._tracer, self._root = tracer, root
        self.trace_id = root.trace_id
        self.callbacks = [handler]
        self.config = config

    def llm(self, llm):
        return TracedLLM(llm, self._tracer.client)

    def tools(self, tools):
        return TracedToolBox(tools, self._tracer.client)

    def finish(self, output: dict[str, Any]) -> None:
        self._root.update(output=output)


class NullTracer:
    enabled = False

    @contextmanager
    def run(self, question: str, model: str, info: TraceInfo | None = None) -> Iterator[NullRun]:
        yield NullRun()

    def score(self, trace_id: str | None, name: str, value: float | bool, comment: str | None = None) -> None:
        pass

    def trace_url(self, trace_id: str | None) -> str | None:
        return None

    def flush(self) -> None:
        pass


class LangfuseTracer(NullTracer):
    enabled = True

    def __init__(self, client: Any, public_key: str):
        from langfuse.langchain import CallbackHandler  # needs the `langchain` package too

        class InlineHandler(CallbackHandler):
            # LangChain runs sync handlers of async runs in an executor with a *copy* of the context,
            # so the span the handler opens for a node would not be the parent of the LLM and tool
            # spans created inside that node. Running inline keeps them nested under their node.
            run_inline = True

        self.client, self._public_key, self._handler_cls = client, public_key, InlineHandler

    @contextmanager
    def run(self, question: str, model: str, info: TraceInfo | None = None) -> Iterator[LiveRun]:
        from langfuse import propagate_attributes

        info = info or TraceInfo()
        tags = [f"model:{model}"]
        meta = {"model": model}
        if info.run_id:
            tags.append(f"run:{info.run_id}")
            meta["run_id"] = info.run_id
        if info.query_id:
            tags.append(f"query:{info.query_id}")
            meta["query_id"] = info.query_id
        if info.model_alias:
            meta["model_alias"] = info.model_alias
        tags += info.extra_tags
        with self.client.start_as_current_observation(
            as_type="span", name="atlandex-agent", input={"question": question}, metadata=meta,
        ) as root, propagate_attributes(
            trace_name="atlandex-agent", session_id=info.run_id or None, tags=tags, metadata=meta,
        ):
            handler = self._handler_cls(public_key=self._public_key)
            run = LiveRun(self, root, handler, {"callbacks": [handler]})
            try:
                yield run
            except Exception as e:
                root.update(level="ERROR", status_message=f"{type(e).__name__}: {str(e)[:300]}")
                raise

    def score(self, trace_id: str | None, name: str, value: float | bool, comment: str | None = None) -> None:
        if not trace_id:
            return
        is_bool = isinstance(value, bool)
        try:
            self.client.create_score(
                trace_id=trace_id, name=name, value=float(value), data_type="BOOLEAN" if is_bool else "NUMERIC",
                comment=comment[:2000] if comment else None, score_id=f"{trace_id}-{name}",
            )
        except Exception as e:  # tracing must never break a run
            log.warning("could not send Langfuse score '%s': %s", name, type(e).__name__)

    def trace_url(self, trace_id: str | None) -> str | None:
        try:
            return self.client.get_trace_url(trace_id=trace_id) if trace_id else None
        except Exception:
            return None

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception as e:
            log.warning("Langfuse flush failed: %s", type(e).__name__)


_NULL = NullTracer()
_cache: dict[str, LangfuseTracer] = {}


def tracer_for(settings: Settings) -> NullTracer:
    """The Langfuse tracer if keys are configured and langfuse is installed, else a no-op tracer."""
    if not (settings.tracing and settings.langfuse_public_key and settings.langfuse_secret_key):
        return _NULL
    cached = _cache.get(settings.langfuse_public_key)
    if cached:
        return cached
    try:
        from langfuse import Langfuse

        client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                          base_url=settings.langfuse_host)
        tracer = LangfuseTracer(client, settings.langfuse_public_key)
    except ImportError:
        log.warning("Langfuse keys are set but tracing is off: install it with pip install -e '.[tracing]'.")
        return _NULL
    except Exception as e:  # bad config must not stop the agent
        log.warning("Langfuse could not start (%s); tracing is off.", type(e).__name__)
        return _NULL
    _cache[settings.langfuse_public_key] = tracer
    return tracer
