"""Tracing is optional: off without keys (no langfuse import, no change in results), and when on,
one run is one trace with router, LLM, tool and citation-check spans. Nothing here uses the network."""
import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from atlandex_agent import tracing
from atlandex_agent.config import Settings
from atlandex_agent.graph import run_agent
from atlandex_agent.tracing import NullTracer, TraceInfo, tracer_for
from eval.run import send_scores
from fakes import FakeLLM, FakeToolBox, VID_A, find_json, search_json, tool_turn

SEG = f"{VID_A}@120"


@pytest.fixture(autouse=True)
def _fresh_tracer_cache():
    tracing._cache.clear()
    yield
    tracing._cache.clear()


def agent_parts():
    llm = FakeLLM({
        "router": ['{"route": "simple", "reason": "one topic"}'],
        "agent": [tool_turn(("find_videos", {"term": "scaling"})),
                  tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
        "answer": [f"Scaling laws predict loss [{SEG}]."],
    })
    tools = FakeToolBox({"find_videos": find_json((VID_A, 60, True)),
                         "search_video": search_json(VID_A, (120, "scaling laws predict loss"))})
    return llm, tools


def base(**kw) -> Settings:
    return Settings(openrouter_api_key="test-key", cache_enabled=False, **kw)


# -- tracing off -------------------------------------------------------------

def test_no_keys_gives_null_tracer():
    assert isinstance(tracer_for(base()), NullTracer)
    assert not tracer_for(base()).enabled


@pytest.mark.parametrize("kw", [
    {"langfuse_public_key": "pk-x"},                                  # secret missing
    {"langfuse_secret_key": "sk-x"},                                  # public missing
    {"langfuse_public_key": "pk-x", "langfuse_secret_key": "sk-x", "tracing": False},  # switched off
])
def test_partial_or_disabled_keys_give_null_tracer(kw):
    assert not tracer_for(base(**kw)).enabled


def clear_langfuse_env(monkeypatch):
    """Unset the vars and make monkeypatch restore 'absent' afterwards (load_dotenv would otherwise leak them)."""
    for v in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL", "LANGFUSE_TRACING"):
        monkeypatch.setenv(v, "")
        monkeypatch.delenv(v)


def test_settings_from_env_without_langfuse_vars_is_off(monkeypatch, tmp_path):
    clear_langfuse_env(monkeypatch)
    s = Settings.from_env(tmp_path / "missing.env")
    assert s.langfuse_public_key == s.langfuse_secret_key == ""
    assert not tracer_for(s).enabled


def test_settings_from_env_reads_langfuse_vars_from_dotenv(monkeypatch, tmp_path):
    clear_langfuse_env(monkeypatch)
    f = tmp_path / ".env"
    f.write_text("LANGFUSE_PUBLIC_KEY=pk-a\nLANGFUSE_SECRET_KEY=sk-b\nLANGFUSE_BASE_URL=https://lf.example\n",
                 encoding="utf-8")
    s = Settings.from_env(f)
    assert (s.langfuse_public_key, s.langfuse_secret_key, s.langfuse_host) == ("pk-a", "sk-b", "https://lf.example")
    assert s.tracing
    monkeypatch.setenv("LANGFUSE_TRACING", "false")
    assert not Settings.from_env(f).tracing  # the off switch beats the keys


def test_keys_set_but_langfuse_not_installed_degrades_to_off(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "langfuse", None)  # makes `import langfuse` raise ImportError
    s = base(langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
    with caplog.at_level("WARNING", logger="atlandex_agent.tracing"):
        t = tracer_for(s)
    assert not t.enabled
    assert "tracing is off" in caplog.text
    assert "sk-x" not in caplog.text and "pk-x" not in caplog.text


def test_run_agent_without_keys_is_unchanged():
    llm, tools = agent_parts()
    res = asyncio.run(run_agent("q?", llm, tools, base()))
    assert res["trace_id"] is None
    assert res["route"] == "simple" and res["steps"] == 2 and res["cited_ids"] == [SEG]
    assert llm.roles() == ["router", "agent", "agent", "agent", "answer"]


def test_off_path_never_imports_langfuse():
    code = (
        "import asyncio, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from fakes import FakeLLM, FakeToolBox, VID_A, find_json, tool_turn\n"
        "from atlandex_agent.config import Settings\n"
        "from atlandex_agent.graph import run_agent\n"
        "llm = FakeLLM({'router': ['{\"route\": \"simple\", \"reason\": \"r\"}'],\n"
        "               'agent': [tool_turn(('find_videos', {'term': 't'})), 'READY'], 'answer': ['x']})\n"
        "tools = FakeToolBox({'find_videos': find_json((VID_A, 60, True))})\n"
        "asyncio.run(run_agent('q?', llm, tools, Settings(openrouter_api_key='k', cache_enabled=False)))\n"
        "bad = [m for m in sys.modules if m == 'langfuse' or m.startswith('langfuse.')]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_settings_repr_hides_langfuse_keys():
    s = base(langfuse_public_key="pk-secret-1", langfuse_secret_key="sk-secret-2")
    assert "secret" not in repr(s)


def test_send_scores_is_a_noop_without_tracing():
    rec = {"trace_id": None, "cited_ids": [SEG], "gold_segments": [{"video": VID_A, "start": 100, "end": 150}]}
    send_scores(NullTracer(), rec, 60)  # must not raise


# -- tracing on (needs the optional packages; skipped where they are absent, e.g. CI) --------

class Recorder:
    def __init__(self):
        self.scores = []

    def create_score(self, **kw):
        self.scores.append(kw)


@pytest.fixture
def live():
    pytest.importorskip("langfuse")
    pytest.importorskip("langchain")
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    pk = f"pk-test-{uuid.uuid4().hex[:8]}"  # Langfuse keeps one client per public key, so make each test's own
    client = Langfuse(public_key=pk, secret_key="sk-test", base_url="http://localhost:9", span_exporter=exporter)
    tracer = tracing.LangfuseTracer(client, pk)
    recorder = Recorder()
    client.create_score = recorder.create_score
    yield tracer, exporter, recorder
    client.shutdown()


def spans_by_name(exporter):
    out = {}
    for s in exporter.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


def test_one_run_is_one_trace_with_router_llm_tool_and_citation_check_spans(live):
    tracer, exporter, _ = live
    llm, tools = agent_parts()
    info = TraceInfo(run_id="20260101-000000", query_id="Q7", model_alias="small")
    res = asyncio.run(run_agent("q?", llm, tools, base(), trace=info, tracer=tracer))
    tracer.flush()
    spans = exporter.get_finished_spans()

    assert len({s.context.trace_id for s in spans}) == 1
    assert res["trace_id"] == format(spans[0].context.trace_id, "032x")
    named = spans_by_name(exporter)
    for name in ("atlandex-agent", "router", "agent", "tools", "answer", "citation_check",
                 "find_videos", "search_video"):
        assert name in named, name
    assert len(named["llm:agent"]) == 3 and len(named["llm:router"]) == 1 and len(named["llm:answer"]) == 1

    # LLM and tool spans nest under their graph node
    by_id = {s.context.span_id: s for s in spans}
    parent = lambda s: by_id[s.parent.span_id].name  # noqa: E731
    assert parent(named["llm:router"][0]) == "router"
    assert parent(named["llm:answer"][0]) == "answer"
    assert parent(named["find_videos"][0]) == "tools"

    gen = named["llm:answer"][0].attributes
    assert gen["langfuse.observation.type"] == "generation"
    assert gen["langfuse.observation.model.name"] == "fake/model:free"
    assert json.loads(gen["langfuse.observation.usage_details"]) == {"input": 100, "output": 10, "total": 110}
    assert json.loads(gen["langfuse.observation.cost_details"]) == {"total": 0.0}

    root = named["atlandex-agent"][0].attributes
    out = json.loads(root["langfuse.observation.output"])
    assert out["route"] == "simple" and out["citation_check"] == {"first_failed": False, "retried": False, "dropped": []}
    assert out["usage"]["prompt_tokens"] == 500 and out["cited_ids"] == [SEG]
    assert root["session.id"] == "20260101-000000"
    assert set(root["langfuse.trace.tags"]) == {"model:fake/model:free", "run:20260101-000000", "query:Q7"}
    assert root["langfuse.trace.metadata.query_id"] == "Q7"
    assert root["langfuse.trace.metadata.model_alias"] == "small"


def test_failed_tool_call_is_marked_as_error(live):
    tracer, exporter, _ = live
    llm, tools = agent_parts()
    tools.outputs["find_videos"] = ["Error calling find_videos: boom"]
    asyncio.run(run_agent("q?", llm, tools, base(), tracer=tracer))
    tracer.flush()
    span = spans_by_name(exporter)["find_videos"][0]
    assert span.attributes["langfuse.observation.level"] == "ERROR"


def test_scores_use_the_trace_id_and_project_definitions(live):
    tracer, _, recorder = live
    rec = {
        "trace_id": "t" * 32, "cited_ids": [SEG], "route": "simple", "expected_route": "multi_step",
        "gold_video_ids": [VID_A], "gold_segments": [{"video": VID_A, "start": 100, "end": 150}],
        "judge": {"sentences": [{"text": "A.", "verdict": "supported", "reason": ""},
                                {"text": "B claim.", "verdict": "unsupported", "reason": "not in passage"}]},
    }
    send_scores(tracer, rec, 60)
    got = {s["name"]: s for s in recorder.scores}
    assert set(got) == {"hallucination_rate", "unjudged_share", "task_success", "video_hit", "router_correct"}
    assert all(s["trace_id"] == "t" * 32 for s in got.values())
    assert got["hallucination_rate"]["value"] == 0.5 and "B claim." in got["hallucination_rate"]["comment"]
    assert got["task_success"]["value"] == 1.0 and got["task_success"]["data_type"] == "BOOLEAN"
    assert got["router_correct"]["value"] == 0.0
    assert got["unjudged_share"]["value"] == 0.0  # both sentences were judged


def test_no_scores_for_a_record_without_a_trace(live):
    tracer, _, recorder = live
    send_scores(tracer, {"trace_id": None, "cited_ids": [], "gold_video_ids": [VID_A]}, 60)
    assert recorder.scores == []


def test_langchain_tools_are_recorded_once(live):
    """Real MCP tools are LangChain tools: the callback handler would record them a second time."""
    from langchain_core.tools import StructuredTool

    from atlandex_agent.mcp_tools import McpToolBox

    async def find_videos(term: str) -> str:
        return find_json((VID_A, 60, True))

    async def search_video(video: str, question: str) -> str:
        return search_json(VID_A, (120, "scaling laws predict loss"))

    tool_list = [StructuredTool.from_function(coroutine=f, name=f.__name__, description=f.__name__)
                 for f in (find_videos, search_video)]
    tracer, exporter, _ = live
    llm, _ = agent_parts()
    asyncio.run(run_agent("q?", llm, McpToolBox(tool_list), base(), tracer=tracer))
    tracer.flush()
    named = spans_by_name(exporter)
    assert len(named["find_videos"]) == 1 and len(named["search_video"]) == 1
