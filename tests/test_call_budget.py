"""The per-run cap on logical LLM calls (Settings.max_llm_calls_per_run, 13), shared by both architectures."""
import asyncio

import pytest

from atlandex_agent.config import Settings
from atlandex_agent.graph import run_agent
from atlandex_agent.llm import CallBudget, CallBudgetExceeded
from eval.run import RUN_ERRORS
from fakes import FakeLLM, VID_A, tool_turn
from test_agent_fixes import ROUTE_SIMPLE, settings, toolbox

SEG = f"{VID_A}@120"


def worst_case_single_llm() -> FakeLLM:
    """The most calls a single run can make: router fails twice (2), the agent skips tools once and then makes six
    one-call turns (7), the first answer is empty and then cites an unretrieved id (2), the rewrite is empty and
    then fine (2)."""
    searches = [tool_turn(("search_video", {"video": VID_A, "question": f"q{i}"})) for i in range(6)]
    return FakeLLM({
        "router": ["no json", "still no json"],
        "agent": ["READY", *searches],
        "answer": ["", "Invented fact [ZZZZZZZZZZZ@5].", "", f"Scaling laws predict loss [{SEG}]."],
    })


def test_the_cap_is_thirteen_and_is_the_most_a_single_run_can_use():
    assert Settings().max_llm_calls_per_run == 13
    llm = worst_case_single_llm()
    res = asyncio.run(run_agent("q?", llm, toolbox(), settings()))
    assert len(llm.calls) == 13 and res["llm_calls_used"] == 13 and res["llm_calls_cap"] == 13
    assert res["steps"] == 6 and res["cited_ids"] == [SEG]
    assert llm.roles().count("answer") == 4


def test_a_run_that_reaches_the_cap_exactly_is_not_refused():
    llm = worst_case_single_llm()
    asyncio.run(run_agent("q?", llm, toolbox(), settings(max_llm_calls_per_run=13)))  # would raise if the cap bound


def test_the_budget_refuses_the_call_after_the_cap_and_counts_what_it_let_through():
    llm = FakeLLM({"x": ["ok"]})
    budget = CallBudget(llm, 2)

    async def go():
        await budget.chat([], role="x")
        await budget.chat([], role="x")
        assert budget.left() == 0 and budget.used == 2
        with pytest.raises(CallBudgetExceeded):
            await budget.chat([], role="x")

    asyncio.run(go())
    assert len(llm.calls) == 2 and budget.model == llm.model and budget.calls is llm.calls


def test_a_run_over_the_cap_is_a_failed_run_for_the_eval_not_a_pass():
    assert CallBudgetExceeded in RUN_ERRORS
    llm = worst_case_single_llm()
    with pytest.raises(CallBudgetExceeded):
        asyncio.run(run_agent("q?", llm, toolbox(), settings(max_llm_calls_per_run=12)))


def test_run_agent_defaults_to_the_single_architecture():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})),
                                                       "READY"],
                   "answer": [f"Scaling laws predict loss [{SEG}]."]})
    res = asyncio.run(run_agent("q?", llm, toolbox(), settings()))
    assert res["arch"] == "single" and res["supervisor_turns"] == [] and res["verify"] == {}
    with pytest.raises(ValueError):
        asyncio.run(run_agent("q?", llm, toolbox(), settings(), arch="swarm"))


# --- the multi-agent graph under the same cap -----------------------------------------------------------------

def test_multi_agent_worst_case_stays_under_the_cap_and_still_ends_with_an_answer():
    """The supervisor always searches, the search agent never finishes and the verifier flags everything."""
    from test_multi_agent import SEARCH, flag_all, run as run_multi

    searches = [tool_turn(("search_video", {"video": VID_A, "question": f"q{i}"})) for i in range(10)]
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "supervisor": [SEARCH], "agent": searches,
                   "answer": [f"Scaling laws predict loss [{SEG}]."], "verify": [flag_all()]})
    res = run_multi(llm)
    assert res["llm_calls_used"] == len(llm.calls) <= 13
    assert res["answer"]  # an answer, not an exception
    assert any("call budget" in n for n in res["notes"])


def test_multi_agent_skips_verification_when_the_budget_is_gone():
    from test_multi_agent import ANSWER, flag_all, run as run_multi

    # cap 7: router 1, supervisor 1, answer with its empty retry 2, verify 1, then the rewrite (also empty
    # first) takes the last two calls, so the second verification round has none left
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "supervisor": [ANSWER], "agent": ["READY"],
                   "answer": ["", f"Scaling laws predict loss [{SEG}].", "", f"Scaling laws predict loss [{SEG}]."],
                   "verify": [flag_all()]})
    res = run_multi(llm, question="About https://youtu.be/AAAAAAAAAAA: scaling?", max_llm_calls_per_run=7)
    assert res["llm_calls_used"] <= 7
    assert res["verify"]["skipped_budget"] is True and res["verify"]["rounds"] == 1
