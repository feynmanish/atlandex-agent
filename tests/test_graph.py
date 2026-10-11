import asyncio

from atlandex_agent.config import Settings
from atlandex_agent.graph import NO_COVERAGE, parse_json_object, run_agent
from fakes import FakeLLM, FakeToolBox, VID_A, VID_B, find_json, search_json, tool_turn

ROUTE_SIMPLE = '{"route": "simple", "reason": "one topic"}'
ROUTE_MULTI = '{"route": "multi_step", "reason": "compare"}'
ROUTE_UNCLEAR = '{"route": "unclear", "reason": "no topic"}'
SEG_A = f"{VID_A}@120"
SEG_B = f"{VID_B}@300"


def settings(**kw) -> Settings:
    return Settings(openrouter_api_key="test-key", cache_enabled=False, **kw)


def toolbox() -> FakeToolBox:
    return FakeToolBox({
        "find_videos": find_json((VID_A, 60, True), (VID_B, 90, True)),
        "search_video": [search_json(VID_A, (120, "scaling laws predict loss")),
                         search_json(VID_B, (300, "data quality matters most"))],
        "locate_quote": "{}",
    })


def run(llm, tools, **kw):
    return asyncio.run(run_agent("q?", llm, tools, settings(**kw)))


def test_simple_route_searches_then_answers_with_valid_citation():
    llm = FakeLLM({
        "router": [ROUTE_SIMPLE],
        "agent": [tool_turn(("find_videos", {"term": "scaling"})),
                  tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
        "answer": [f"Scaling laws predict loss [{SEG_A}]."],
    })
    res = run(llm, toolbox())
    assert res["route"] == "simple"
    assert res["steps"] == 2
    assert res["cited_ids"] == [SEG_A]
    assert SEG_A in res["retrieved_ids"]
    assert res["citation_check"]["dropped"] == [] and not res["citation_check"]["retried"]
    assert res["citations"][SEG_A].startswith("[Video A, 2:00]")
    assert res["usage"]["llm_calls"] == len(llm.calls)


def test_multi_step_route_is_recorded_and_allows_several_searches():
    tb = toolbox()
    llm = FakeLLM({
        "router": [ROUTE_MULTI],
        "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"}),
                            ("search_video", {"video": VID_B, "question": "q"})), "READY"],
        "answer": [f"A says scaling [{SEG_A}]. B says data [{SEG_B}]."],
    })
    res = run(llm, tb)
    assert res["route"] == "multi_step" and res["steps"] == 2
    assert res["cited_ids"] == [SEG_A, SEG_B]


def test_unclear_route_asks_one_question_and_never_calls_tools():
    tb = toolbox()
    llm = FakeLLM({"router": [ROUTE_UNCLEAR], "clarify": ["Which topic do you mean?"]})
    res = run(llm, tb)
    assert res["route"] == "unclear" and res["clarification"]
    assert res["answer"] == "Which topic do you mean?"
    assert tb.calls == [] and llm.roles() == ["router", "clarify"]


def test_router_garbage_falls_back_to_multi_step_and_is_noted():
    llm = FakeLLM({"router": ["no idea"], "agent": ["READY"], "answer": ["x"]})
    res = run(llm, toolbox())
    assert res["route"] == "multi_step"
    assert any("router fell back" in n for n in res["notes"])


def test_hard_step_limit_stops_a_looping_agent():
    tb = FakeToolBox({"find_videos": find_json((VID_A, 60, True)),
                      "search_video": search_json(VID_A, (120, "text"))})
    # the model never says READY; each turn asks for a new search
    turns = [tool_turn(("search_video", {"video": VID_A, "question": f"q{i}"})) for i in range(20)]
    llm = FakeLLM({"router": [ROUTE_MULTI], "agent": turns, "answer": [f"Text [{SEG_A}]."]})
    res = run(llm, tb, max_tool_steps=3)
    assert res["steps"] == 3 and len(tb.calls) == 3
    assert llm.roles().count("agent") == 3


def test_step_limit_truncates_a_batch_of_parallel_calls():
    tb = toolbox()
    batch = tool_turn(*[("search_video", {"video": VID_A, "question": f"q{i}"}) for i in range(5)])
    llm = FakeLLM({"router": [ROUTE_MULTI], "agent": [batch], "answer": [f"Text [{SEG_A}]."]})
    res = run(llm, tb, max_tool_steps=2)
    assert res["steps"] == 2


def test_agent_that_skips_tools_is_nudged_once():
    llm = FakeLLM({"router": [ROUTE_SIMPLE],
                   "agent": ["READY", tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
                   "answer": [f"Text [{SEG_A}]."]})
    res = run(llm, toolbox())
    assert res["steps"] == 1 and any("nudged" in n for n in res["notes"])


def test_context_is_bounded_not_a_growing_history():
    tb = FakeToolBox({"search_video": search_json(VID_A, (120, "x" * 900))})
    turns = [tool_turn(("search_video", {"video": VID_A, "question": f"q{i}"})) for i in range(6)] + ["READY"]
    llm = FakeLLM({"router": [ROUTE_MULTI], "agent": turns, "answer": [f"Text [{SEG_A}]."]})
    run(llm, tb, evidence_char_budget=3000)
    sizes = [len(m[1]["content"]) for r, m in llm.prompts if r == "agent"]
    assert all(len(m) == 2 for r, m in llm.prompts if r == "agent")  # system + one user message, always
    assert max(sizes) < 3000 + 1200


def test_no_retrieved_segments_gives_no_coverage_without_an_answer_call():
    tb = FakeToolBox({"find_videos": find_json()})
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [tool_turn(("find_videos", {"term": "zzz"})), "READY"],
                   "answer": ["should never be called"]})
    res = run(llm, tb)
    assert res["answer"] == NO_COVERAGE and "answer" not in llm.roles()


def test_citation_check_retries_once_then_passes():
    llm = FakeLLM({
        "router": [ROUTE_SIMPLE],
        "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
        "answer": ["Made up claim [ZZZZZZZZZZZ@5].", f"Scaling laws predict loss [{SEG_A}]."],
    })
    res = run(llm, toolbox())
    cc = res["citation_check"]
    assert cc["retried"] and cc["first_failed"] and cc["dropped"] == []
    assert res["cited_ids"] == [SEG_A]
    assert llm.roles().count("answer") == 2
    retry_prompt = [m for r, m in llm.prompts if r == "answer"][1][1]["content"]
    assert "ZZZZZZZZZZZ@5" in retry_prompt


def test_citation_check_drops_unsupported_claims_after_failed_retry():
    llm = FakeLLM({
        "router": [ROUTE_SIMPLE],
        "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"],
        "answer": [f"Scaling laws predict loss [{SEG_A}]. Invented fact [ZZZZZZZZZZZ@5]."],
    })
    res = run(llm, toolbox())
    cc = res["citation_check"]
    assert cc["retried"] and len(cc["dropped"]) == 1 and "Invented" in cc["dropped"][0]
    assert "Invented" not in res["answer"] and "Scaling laws" in res["answer"]
    assert llm.roles().count("answer") == 2  # one retry, never a third


def test_parse_json_object_is_tolerant():
    assert parse_json_object('```json\n{"route": "simple"}\n```') == {"route": "simple"}
    assert parse_json_object('<think>hmm {x}</think> ok {"a": 1} bye') == {"a": 1}
    assert parse_json_object("nothing") is None
