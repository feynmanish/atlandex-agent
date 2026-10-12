"""Per-model agent token cap, the empty-answer retry, and an empty answer as a failed run. Offline."""
import asyncio

import pytest

from atlandex_agent.config import Settings
from atlandex_agent.graph import EmptyAnswerError, FINISH_TOOL, run_agent
from eval.run import RUN_ERRORS
from atlandex_agent.llm import LLMError
from fakes import FakeLLM, VID_A, tool_turn
from test_agent_fixes import ANSWER, ROUTE_SIMPLE, settings, toolbox

SEG = f"{VID_A}@120"
SEARCH = tool_turn(("search_video", {"video": VID_A, "question": "q"}))


def run(llm, **kw):
    return asyncio.run(run_agent("About scaling?", llm, toolbox(), settings(**kw)))


# --- config ---------------------------------------------------------------------------------------------

def test_cap_is_1500_for_the_large_model_and_600_for_the_small_one():
    s = Settings()
    assert s.agent_cap_for(s.model_large) == 1500 and s.agent_cap_for(s.paid_large) == 1500
    assert s.agent_cap_for(s.model_small) == 600 and s.agent_cap_for(s.paid_small) == 600
    assert s.agent_cap_for("some/other-model:free") == 600  # an explicit model id gets the small cap


def test_the_caps_are_settings_not_constants_and_an_override_wins():
    s = Settings(agent_max_tokens_large=2000, agent_max_tokens_small=300)
    assert s.agent_cap_for(s.model_large) == 2000 and s.agent_cap_for(s.model_small) == 300
    assert Settings(agent_max_tokens=250).agent_cap_for(Settings().model_large) == 250


def test_caps_can_be_set_from_the_environment(monkeypatch, tmp_path):
    for v in ("AGENT_MAX_TOKENS", "AGENT_MAX_TOKENS_LARGE", "AGENT_MAX_TOKENS_SMALL", "ANSWER_RETRY_MAX_TOKENS"):
        monkeypatch.setenv(v, "")
        monkeypatch.delenv(v)
    s = Settings.from_env(tmp_path / "missing.env")
    assert (s.agent_max_tokens_large, s.agent_max_tokens_small, s.agent_max_tokens, s.answer_retry_max_tokens) == (
        1500, 600, None, 4000)
    monkeypatch.setenv("AGENT_MAX_TOKENS_LARGE", "1200")
    monkeypatch.setenv("AGENT_MAX_TOKENS_SMALL", "500")
    monkeypatch.setenv("ANSWER_RETRY_MAX_TOKENS", "5000")
    s = Settings.from_env(tmp_path / "missing.env")
    assert (s.agent_max_tokens_large, s.agent_max_tokens_small, s.answer_retry_max_tokens) == (1200, 500, 5000)
    monkeypatch.setenv("AGENT_MAX_TOKENS", "700")
    assert Settings.from_env(tmp_path / "missing.env").agent_max_tokens == 700


# --- the cap actually used ----------------------------------------------------------------------------------

def agent_caps(llm):
    return [cap for (role, _), cap in zip(llm.prompts, llm.max_tokens) if role == "agent"]


def test_agent_turns_of_the_large_model_get_1500_and_of_the_small_model_600():
    s = Settings()
    for model, cap in ((s.model_large, 1500), (s.model_small, 600)):
        llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER], "agent": [SEARCH, tool_turn((FINISH_TOOL, {}))]},
                      model=model)
        run(llm)
        assert agent_caps(llm) == [cap, cap]


def test_every_agent_turn_is_recorded_with_its_cap_and_what_it_did():
    s = Settings()
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [SEARCH, {"text": "I have enough, so here is the answer " * 20, "finish_reason": "stop"}]},
                  model=s.model_large)
    turns = run(llm)["agent_turns"]
    assert [t["tool_calls"] for t in turns] == [["search_video"], []]
    assert turns[0]["finish"] is False and turns[0]["text_chars"] == 0 and turns[0]["cap"] == 1500
    assert turns[1]["finish"] is False and turns[1]["text_chars"] > 500  # wrote prose instead of calling finish
    llm2 = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER], "agent": [SEARCH, tool_turn((FINISH_TOOL, {}))]})
    assert run(llm2)["agent_turns"][1]["finish"] is True


# --- empty answers ----------------------------------------------------------------------------------------------

def answer_caps(llm):
    return [cap for (role, _), cap in zip(llm.prompts, llm.max_tokens) if role == "answer"]


@pytest.mark.parametrize("first", ["", "  \n "])
def test_an_empty_answer_is_retried_once_with_a_larger_budget(first):
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [SEARCH, "READY"], "answer": [first, ANSWER]})
    res = run(llm)
    assert res["answer"] == ANSWER and res["cited_ids"] == [SEG]
    assert answer_caps(llm) == [None, 4000]  # default budget first, then answer_retry_max_tokens
    assert any("answer was empty" in n and "still empty" not in n for n in res["notes"])


def test_the_retry_budget_is_a_setting():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [SEARCH, "READY"], "answer": ["", ANSWER]})
    run(llm, answer_retry_max_tokens=2500)
    assert answer_caps(llm) == [None, 2500]


def test_a_nonempty_answer_is_never_retried_for_emptiness():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [SEARCH, "READY"], "answer": [ANSWER]})
    run(llm)
    assert llm.roles().count("answer") == 1 and answer_caps(llm) == [None]


def test_an_answer_that_is_still_empty_fails_the_run_and_is_not_asked_a_third_time():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [SEARCH, "READY"], "answer": ["", ""]})
    with pytest.raises(EmptyAnswerError):
        run(llm)
    assert llm.roles().count("answer") == 2


def test_an_empty_answer_after_the_citation_retry_also_fails_the_run():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [SEARCH, "READY"],
                   "answer": [f"Scaling laws predict loss [{SEG}]. Thus bigger models always win.", "", ""]})
    with pytest.raises(EmptyAnswerError):
        run(llm)
    assert llm.roles().count("answer") == 3  # first answer, the citation retry (empty), the bigger-budget retry (empty)


def test_an_empty_answer_is_a_failed_run_for_the_eval_and_never_a_pass():
    assert EmptyAnswerError in RUN_ERRORS  # eval.run records it as an error, so --resume re-runs it
    assert not issubclass(EmptyAnswerError, LLMError)
