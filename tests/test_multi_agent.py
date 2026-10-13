"""The multi-agent graph: supervisor -> search agent -> answer -> citation check -> verify. Offline."""
import asyncio

import pytest

from atlandex_agent import prompts
from atlandex_agent.config import Settings
from atlandex_agent.graph import NO_COVERAGE, run_agent
from fakes import FakeLLM, FakeToolBox, VID_A, VID_B, find_json, json_reply, search_json, tool_turn

ROUTE_SIMPLE = '{"route": "simple", "reason": "one topic"}'
ROUTE_MULTI = '{"route": "multi_step", "reason": "compare"}'
SEG_A, SEG_B = f"{VID_A}@120", f"{VID_B}@300"
SEARCH = json_reply({"action": "search", "task": "what does A say about scaling"})
SEARCH_B = json_reply({"action": "search", "task": "what does B say about data"})
ANSWER = json_reply({"action": "answer", "task": ""})
NONE_FLAGGED = json_reply({"unsupported": []})
GOOD = f"Scaling laws predict loss [{SEG_A}]."


def settings(**kw) -> Settings:
    return Settings(openrouter_api_key="test-key", cache_enabled=False, **kw)


def toolbox() -> FakeToolBox:
    return FakeToolBox({
        "find_videos": find_json((VID_A, 60, True), (VID_B, 90, True)),
        "search_video": [search_json(VID_A, (120, "scaling laws predict loss")),
                         search_json(VID_B, (300, "data quality matters most"))],
        "locate_quote": "{}",
    })


def search_a(i: int = 0) -> dict:
    return tool_turn(("search_video", {"video": VID_A, "question": f"q{i}"}))


def run(llm, tb=None, question="q?", **kw):
    return asyncio.run(run_agent(question, llm, tb or toolbox(), settings(**kw), arch="multi"))


def basic_llm(**over) -> FakeLLM:
    replies = {"router": [ROUTE_SIMPLE], "supervisor": [SEARCH, ANSWER], "agent": [search_a(), "READY"],
               "answer": [GOOD], "verify": [NONE_FLAGGED]}
    replies.update(over)
    return FakeLLM(replies)


def test_supervisor_delegates_a_task_the_search_agent_finishes_then_answer_and_verify():
    llm = basic_llm()
    res = run(llm)
    assert llm.roles() == ["router", "supervisor", "agent", "agent", "supervisor", "answer", "verify"]
    assert res["arch"] == "multi" and res["steps"] == 1 and res["cited_ids"] == [SEG_A]
    assert [t["action"] for t in res["supervisor_turns"]] == ["search", "answer"]
    assert res["supervisor_turns"][0]["task"] == "what does A say about scaling"
    assert res["verify"]["rounds"] == 1 and res["verify"]["dropped_unverified"] == [] and not res["verify"]["failed"]
    assert res["answer"] == GOOD


def test_the_task_reaches_the_search_agent_whose_system_prompt_is_unchanged():
    llm = basic_llm()
    run(llm)
    agent_turns = [m for r, m in llm.prompts if r == "agent"]
    assert agent_turns[0][0]["content"] == prompts.AGENT_SYSTEM.format(max_steps=6)
    assert "Task: what does A say about scaling" in agent_turns[0][1]["content"]
    assert "Steps left for this task: 3." in agent_turns[0][1]["content"]
    assert "Steps left for this task: 2." in agent_turns[1][1]["content"]


def test_the_search_agent_returns_to_the_supervisor_when_its_task_allowance_is_used_up():
    llm = basic_llm(agent=[search_a(i) for i in range(10)])  # never finishes by itself
    res = run(llm)
    assert res["steps"] == 3
    assert llm.roles()[:6] == ["router", "supervisor", "agent", "agent", "agent", "supervisor"]


def test_a_batch_of_calls_is_cut_at_the_task_allowance():
    batch = tool_turn(*[("search_video", {"video": VID_A, "question": f"q{i}"}) for i in range(5)])
    res = run(basic_llm(agent=[batch]))
    assert res["steps"] == 3


def test_the_global_step_limit_holds_across_the_seed_and_several_tasks_and_then_goes_to_the_answer():
    llm = basic_llm(supervisor=[SEARCH, SEARCH_B, ANSWER], agent=[search_a(i) for i in range(10)])
    res = run(llm, question=f"About https://youtu.be/{VID_A}: what do scaling laws predict?")
    assert res["steps"] == 6  # 1 for the linked video, 3 for the first task, 2 left for the second
    roles = llm.roles()
    assert roles.count("supervisor") == 2 and roles.count("agent") == 5
    assert roles[roles.index("answer") - 1] == "agent"  # the last tools node went to the answer, not the supervisor
    agent_msgs = [m[1]["content"] for r, m in llm.prompts if r == "agent"]
    assert "Steps left for this task: 2." in agent_msgs[3]  # second task: only 2 of the 6 steps remain


def test_the_last_supervisor_turn_can_only_answer():
    llm = basic_llm(supervisor=[SEARCH], agent=[search_a(0), "READY", search_a(1), "READY"])
    res = run(llm)
    assert [t["action"] for t in res["supervisor_turns"]] == ["search", "search", "answer"]
    assert llm.roles().count("supervisor") == 3 and llm.roles().count("answer") == 1
    last_prompt = [m for r, m in llm.prompts if r == "supervisor"][2][1]["content"]
    assert "the last turn" in last_prompt
    assert any("last turn" in n for n in res["notes"])


@pytest.mark.parametrize("bad", ["no json at all", json_reply({"action": "dance"}),
                                 json_reply({"action": "search", "task": ""})])
def test_an_unparseable_supervisor_reply_searches_the_question_when_nothing_was_retrieved(bad):
    llm = basic_llm(supervisor=[bad, ANSWER])
    res = run(llm, question="What about scaling?")
    first = res["supervisor_turns"][0]
    assert first["action"] == "search" and first["task"] == "What about scaling?" and first["fallback"]
    assert any("unparseable" in n for n in res["notes"])


def test_an_unparseable_supervisor_reply_goes_to_the_answer_once_something_was_retrieved():
    llm = basic_llm(supervisor=[SEARCH, "garbage"])
    res = run(llm)
    assert [t["action"] for t in res["supervisor_turns"]] == ["search", "answer"]
    assert res["supervisor_turns"][1]["fallback"] and res["answer"] == GOOD


def test_answering_before_any_search_is_turned_into_a_search_of_the_question():
    llm = basic_llm(supervisor=[ANSWER, ANSWER])
    res = run(llm, question="What about scaling?")
    assert res["supervisor_turns"][0]["action"] == "search" and res["supervisor_turns"][0]["task"] == "What about scaling?"
    assert any("before any search" in n for n in res["notes"])


def test_the_unclear_route_goes_only_to_clarify():
    llm = FakeLLM({"router": ['{"route": "unclear", "reason": "vague"}'], "clarify": ["Which topic?"]})
    tb = toolbox()
    res = run(llm, tb)
    assert llm.roles() == ["router", "clarify"] and tb.calls == [] and res["clarification"]


def test_a_linked_video_is_read_before_the_first_supervisor_turn_and_counts_as_a_step():
    tb = toolbox()
    llm = basic_llm(supervisor=[ANSWER])
    res = run(llm, tb, question=f"About https://youtu.be/{VID_A}: what do scaling laws predict?")
    assert tb.calls[0][0] == "search_video" and tb.calls[0][1]["video"] == VID_A
    assert llm.roles()[:2] == ["router", "supervisor"] and res["steps"] == 1
    assert res["supervisor_turns"][0]["action"] == "answer"  # the linked video was enough: no extra search


@pytest.mark.parametrize("model, cap", [("fake/model:free", 600), (Settings().model_large, 1500)])
def test_supervisor_and_search_turns_get_the_agent_cap_and_verify_gets_the_answer_cap(model, cap):
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "supervisor": [SEARCH, ANSWER], "agent": [search_a(), "READY"],
                   "answer": [GOOD], "verify": [NONE_FLAGGED]}, model=model)
    run(llm)
    by_role = dict(zip(llm.roles(), llm.max_tokens))
    assert by_role["supervisor"] == cap and by_role["agent"] == cap
    assert by_role["router"] is None and by_role["answer"] is None
    assert by_role["verify"] == 1500


def test_verify_max_tokens_is_a_setting():
    llm = basic_llm()
    run(llm, verify_max_tokens=4000)
    assert dict(zip(llm.roles(), llm.max_tokens))["verify"] == 4000


def test_no_retrieved_segments_gives_no_coverage_and_the_verifier_is_never_called():
    tb = FakeToolBox({"find_videos": find_json()})
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "supervisor": [SEARCH, ANSWER],
                   "agent": [tool_turn(("find_videos", {"term": "zzz"})), "READY"], "answer": ["never"],
                   "verify": ["never"]})
    res = run(llm, tb)
    assert res["answer"] == NO_COVERAGE and "answer" not in llm.roles() and "verify" not in llm.roles()


def test_single_runs_never_see_the_multi_agent_roles():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [search_a(), "READY"], "answer": [GOOD]})
    res = asyncio.run(run_agent("q?", llm, toolbox(), settings()))
    assert "supervisor" not in llm.roles() and "verify" not in llm.roles() and res["arch"] == "single"


def flag_all() -> str:
    """A verifier reply that flags every sentence an answer is likely to have."""
    return json_reply({"unsupported": [0, 1, 2, 3], "reasons": {}})
