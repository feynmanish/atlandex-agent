"""Fix 1 (a linked video is read first) and fix 2 (a `finish` tool and a token cap for agent turns). Offline."""
import asyncio

import pytest

from atlandex_agent.config import Settings
from atlandex_agent.graph import FINISH_TOOL, run_agent
from atlandex_agent.urls import strip_urls, video_ids_in
from fakes import FakeLLM, FakeToolBox, VID_A, VID_B, find_json, search_json, tool_turn

ROUTE_SIMPLE = '{"route": "simple", "reason": "one video"}'
ROUTE_UNCLEAR = '{"route": "unclear", "reason": "no topic"}'
ANSWER = f"Scaling laws predict loss [{VID_A}@120]."

# the first question of the existing eval set, as it is in eval/dataset.jsonl
EX_QUESTION = ('About the video "Vladimir Vapnik: Deep Learning and the Essence of Intelligence | AI Podcast Clips" '
               "(https://www.youtube.com/watch?v=Burge5lagdM): How does attention in language models relate to "
               "finding good predicates?")


def settings(**kw) -> Settings:
    return Settings(openrouter_api_key="test-key", cache_enabled=False, **kw)


def toolbox() -> FakeToolBox:
    return FakeToolBox({
        "find_videos": find_json((VID_A, 60, True), (VID_B, 90, True)),
        "search_video": [search_json(VID_A, (120, "scaling laws predict loss")),
                         search_json(VID_B, (300, "data quality matters most"))],
        "locate_quote": "{}",
    })


def run(question, llm, tools, **kw):
    return asyncio.run(run_agent(question, llm, tools, settings(**kw)))


# --- fix 1: URL -> video ID ------------------------------------------------------------------------------

@pytest.mark.parametrize("text, ids", [
    ("see https://www.youtube.com/watch?v=Burge5lagdM please", ["Burge5lagdM"]),
    ("https://youtube.com/watch?feature=share&v=Burge5lagdM&t=30s", ["Burge5lagdM"]),
    ("https://youtu.be/Burge5lagdM?si=abc", ["Burge5lagdM"]),
    ("https://www.youtube.com/shorts/Burge5lagdM", ["Burge5lagdM"]),
    ("https://m.youtube.com/live/Burge5lagdM", ["Burge5lagdM"]),
    ("(https://www.youtube.com/watch?v=Burge5lagdM) and https://youtu.be/Ib75XEaS-kE, again "
     "https://youtu.be/Burge5lagdM", ["Burge5lagdM", "Ib75XEaS-kE"]),
    ("no link here, just Burge5lagdM and https://example.com/watch?v=Burge5lagdM", []),
    ("", []),
])
def test_video_ids_in(text, ids):
    assert video_ids_in(text) == ids


def test_strip_urls_removes_the_link_and_the_brackets_it_leaves():
    assert strip_urls(EX_QUESTION) == (
        'About the video "Vladimir Vapnik: Deep Learning and the Essence of Intelligence | AI Podcast Clips": '
        "How does attention in language models relate to finding good predicates?")
    assert strip_urls("what is this? https://youtu.be/Burge5lagdM") == "what is this?"


def test_a_linked_video_is_read_before_the_agent_loop_and_counts_as_a_step():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [tool_turn((FINISH_TOOL, {}))], "answer": [ANSWER]})
    tools = toolbox()
    res = run(EX_QUESTION, llm, tools)
    assert tools.calls == [("search_video", {"video": "Burge5lagdM", "question": strip_urls(EX_QUESTION)})]
    assert res["steps"] == 1 and res["tool_calls"][0]["name"] == "search_video"
    assert llm.roles() == ["router", "agent", "answer"]  # no find_videos detour, no nudge
    assert "Step 1: search_video" in llm.prompts[1][1][1]["content"]  # the agent's first turn already sees it
    assert any("Burge5lagdM" in n for n in res["notes"])


def test_every_linked_video_up_to_three_is_read_first():
    q = "Compare " + " and ".join(f"https://youtu.be/{v}" for v in ("AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC",
                                                                      "DDDDDDDDDDD"))
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": [tool_turn((FINISH_TOOL, {}))], "answer": [ANSWER]})
    tools = toolbox()
    run(q, llm, tools)
    assert [c[1]["video"] for c in tools.calls] == ["AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"]


def test_the_seed_counts_against_the_step_limit():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "agent": ["READY"], "answer": [ANSWER]})
    res = run(EX_QUESTION, llm, toolbox(), max_tool_steps=1)
    assert res["steps"] == 1 and llm.roles() == ["router", "answer"]  # limit reached: straight to the answer


def test_a_linked_video_that_cannot_be_searched_falls_back_to_the_normal_loop():
    tools = FakeToolBox({
        "search_video": ["Error: this video has no searchable transcript.", search_json(VID_A, (120, "scaling laws"))],
        "find_videos": find_json((VID_A, 60, True)),
    })
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn(("find_videos", {"term": "scaling"})),
                             tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"]})
    res = run(EX_QUESTION, llm, tools)
    assert [t["name"] for t in res["tool_calls"]] == ["search_video", "find_videos", "search_video"]
    assert res["tool_calls"][0]["error"] is True and res["cited_ids"] == [f"{VID_A}@120"]


def test_a_question_without_a_link_starts_with_the_agent_as_before():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"]})
    tools = toolbox()
    res = run("What is said about scaling laws?", llm, tools)
    assert tools.calls == [("search_video", {"video": VID_A, "question": "q"})]
    assert not any("named in the question" in n for n in res["notes"])


def test_an_unclear_question_with_a_link_still_asks_instead_of_searching():
    llm = FakeLLM({"router": [ROUTE_UNCLEAR], "clarify": ["Which part do you mean?"]})
    tools = toolbox()
    res = run(EX_QUESTION, llm, tools)
    assert res["clarification"] and tools.calls == []


# --- fix 2: finish tool and token cap -----------------------------------------------------------------------

def test_finish_is_offered_next_to_the_real_tools_and_ends_the_loop_without_a_step():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})),
                             tool_turn((FINISH_TOOL, {}))]})
    tools = toolbox()
    res = run("About scaling?", llm, tools)
    agent_turns = [names for (role, _), names in zip(llm.prompts, llm.tools_offered) if role == "agent"]
    assert all(FINISH_TOOL in names and "search_video" in names for names in agent_turns)
    assert FINISH_TOOL not in [c[0] for c in tools.calls]  # never sent to the MCP server
    assert res["steps"] == 1 and llm.roles().count("agent") == 2


def test_finish_next_to_a_real_call_runs_the_real_call_and_ignores_finish():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn((FINISH_TOOL, {}), ("search_video", {"video": VID_A, "question": "q"})),
                             tool_turn((FINISH_TOOL, {}))]})
    tools = toolbox()
    res = run("About scaling?", llm, tools)
    assert [c[0] for c in tools.calls] == ["search_video"] and res["steps"] == 1


def test_finish_before_any_search_is_nudged_once_like_a_bare_reply():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn((FINISH_TOOL, {})), tool_turn(("search_video", {"video": VID_A, "question": "q"})),
                             tool_turn((FINISH_TOOL, {}))]})
    res = run("About scaling?", llm, toolbox())
    assert any("nudged" in n for n in res["notes"]) and res["steps"] == 1


def test_a_bare_ready_reply_still_ends_the_loop():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"]})
    assert run("About scaling?", llm, toolbox())["steps"] == 1


def test_agent_turns_get_the_600_token_cap_and_other_roles_do_not():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})), "READY"]})
    run("About scaling?", llm, toolbox())
    caps = {role: cap for (role, _), cap in zip(llm.prompts, llm.max_tokens)}
    assert caps["agent"] == 600 and caps["router"] is None and caps["answer"] is None
    llm2 = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER], "agent": ["READY"]})
    run(EX_QUESTION, llm2, toolbox(), agent_max_tokens=250)
    assert [c for (r, _), c in zip(llm2.prompts, llm2.max_tokens) if r == "agent"] == [250]


def test_a_turn_cut_off_at_the_cap_without_a_tool_call_counts_as_done_and_is_noted():
    llm = FakeLLM({"router": [ROUTE_SIMPLE], "answer": [ANSWER],
                   "agent": [tool_turn(("search_video", {"video": VID_A, "question": "q"})),
                             {"text": "I have enough evidence to write that", "finish_reason": "length"}]})
    res = run("About scaling?", llm, toolbox())
    assert llm.roles().count("agent") == 2 and llm.roles()[-1] == "answer"
    assert any("token cap" in n for n in res["notes"])
