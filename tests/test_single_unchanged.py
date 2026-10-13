"""The single-agent graph must behave exactly as it did before the multi-agent variant was added.

Four scripted scenarios are replayed against the fakes. For each one the test records everything the graph
sent to the LLM (role, messages, token cap, tools offered), every tool call and the visible result, and compares
it with tests/golden/single_<name>.json. The fixtures were generated from `main` before graph.py was refactored,
so any change to a single-agent prompt, cap, call order or result fails here.

Regenerate on purpose with:  UPDATE_GOLDEN=1 pytest tests/test_single_unchanged.py
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from atlandex_agent.config import Settings
from atlandex_agent.graph import run_agent
from fakes import FakeLLM, FakeToolBox, VID_A, VID_B, find_json, search_json, tool_turn

GOLDEN = Path(__file__).parent / "golden"
ROUTE_SIMPLE = '{"route": "simple", "reason": "one topic"}'
ROUTE_MULTI = '{"route": "multi_step", "reason": "compare"}'
SEG_A, SEG_B = f"{VID_A}@120", f"{VID_B}@300"


def settings(**kw) -> Settings:
    return Settings(openrouter_api_key="test-key", cache_enabled=False, **kw)


def toolbox() -> FakeToolBox:
    return FakeToolBox({
        "find_videos": find_json((VID_A, 60, True), (VID_B, 90, True)),
        "search_video": [search_json(VID_A, (120, "scaling laws predict loss")),
                         search_json(VID_B, (300, "data quality matters most"))],
        "locate_quote": "{}",
    })


SCENARIOS = {
    # a linked video is read first, then the agent finishes at once
    "simple_with_link": (
        f"About https://youtu.be/{VID_A}: what do scaling laws predict?",
        {"router": [ROUTE_SIMPLE], "agent": ["READY"], "answer": [f"Scaling laws predict loss [{SEG_A}]."]}),
    # several searches in one batch, then the finish tool
    "multi_step_no_link": (
        "How do A and B differ on scaling versus data?",
        {"router": [ROUTE_MULTI],
         "agent": [tool_turn(("find_videos", {"term": "scaling"})),
                   tool_turn(("search_video", {"video": VID_A, "question": "q"}),
                             ("search_video", {"video": VID_B, "question": "q"})),
                   tool_turn(("finish", {}))],
         "answer": [f"A says scaling laws predict loss [{SEG_A}]. B says data quality matters [{SEG_B}]."]}),
    # first draft cites an unretrieved id: one rewrite, which still has a bad sentence that is dropped
    "citation_retry_then_drop": (
        "What do scaling laws predict?",
        {"router": [ROUTE_SIMPLE],
         "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
         "answer": [f"Scaling laws predict loss [{SEG_A}]. Invented fact [ZZZZZZZZZZZ@5]."]}),
    # an empty first answer is asked again with the larger budget
    "empty_answer_then_retry": (
        "What do scaling laws predict?",
        {"router": [ROUTE_SIMPLE],
         "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
         "answer": ["", f"Scaling laws predict loss [{SEG_A}]."]}),
}


def record(name: str) -> dict:
    question, replies = SCENARIOS[name]
    llm, tb = FakeLLM(replies), toolbox()
    res = asyncio.run(run_agent(question, llm, tb, settings()))
    return {
        "prompts": [[role, msgs] for role, msgs in llm.prompts],
        "max_tokens": llm.max_tokens,
        "tools_offered": llm.tools_offered,
        "tool_calls": [[n, a] for n, a in tb.calls],
        "result": {k: res[k] for k in ("route", "route_reason", "answer", "cited_ids", "retrieved_ids", "steps",
                                       "citation_check", "notes", "agent_turns")},
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_single_agent_transcript_matches_the_golden_fixture(name):
    got = json.loads(json.dumps(record(name)))  # tuples -> lists, as in the fixture
    path = GOLDEN / f"single_{name}.json"
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(json.dumps(got, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    assert got == json.loads(path.read_text(encoding="utf-8"))
