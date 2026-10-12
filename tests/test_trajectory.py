"""eval.trajectory follows the same queries through several runs and counts capped turns and drafts."""
from eval.trajectory import agent_turns, to_markdown, trajectory, turn_stats

CAPS = {"large": 600, "small": 600}


def rec(id_="q1", alias="large", steps=3, lat=50.0, tin=4000, tout=800, verdicts=("supported", "unsupported"),
        agent_tokens=(120, 150), turns=None, cited=("AAAAAAAAAAA@100",), **kw):
    r = {"id": id_, "model_alias": alias, "model": "m", "steps": steps, "latency_sec": lat, "needs_review": False,
         "usage": {"prompt_tokens": tin, "completion_tokens": tout, "cost_usd": 0.0},
         "judge": {"sentences": [{"verdict": v} for v in verdicts]}, "citation_check": {}, "cited_ids": list(cited),
         "gold_video_ids": ["AAAAAAAAAAA"], "gold_segments": [{"video": "AAAAAAAAAAA", "start": 90, "end": 120}],
         "llm_calls": [{"role": "agent", "completion_tokens": t} for t in agent_tokens] + [{"role": "answer", "completion_tokens": 50}]}
    if turns is not None:
        r["agent_turns"] = turns
    r.update(kw)
    return r


def turn(tool_calls=(), tokens=100, chars=0, reason="tool_calls", cap=1500):
    return {"tool_calls": list(tool_calls), "finish": "finish" in tool_calls, "text_chars": chars,
            "completion_tokens": tokens, "finish_reason": reason, "cap": cap}


def test_turns_of_older_runs_are_rebuilt_from_llm_calls_with_the_cap_in_force():
    ts = agent_turns(rec(agent_tokens=(100, 600)), CAPS)
    assert [t["cap"] for t in ts] == [600, 600] and ts[1]["text_chars"] is None


def test_cap_hits_are_turns_that_used_their_whole_cap():
    old = rec(agent_tokens=(100, 600, 599))
    s = turn_stats([old], CAPS)
    assert s["agent_turns"] == 3 and s["cap_hits"] == 1 and s["runs_with_cap_hit"] == 1
    assert s["drafts"] == 0 and s["exact_turns"] == 0  # not recorded for older runs


def test_drafts_need_no_tool_call_and_long_visible_text_and_cut_offs_are_counted_apart():
    r = rec(turns=[turn(["search_video"]), turn([], tokens=700, chars=2500, reason="stop"),
                   turn([], tokens=1500, chars=0, reason="length"), turn(["finish"], tokens=40),
                   turn([], tokens=5, chars=5, reason="stop")])
    s = turn_stats([r], CAPS)
    assert s["exact_turns"] == 5 and s["drafts"] == 1 and s["cut_off_without_tool_call"] == 1
    assert s["cap_hits"] == 1  # only the 1500-token turn used its whole 1500 cap


def test_ending_turn_proxy_works_for_every_run():
    s = turn_stats([rec(agent_tokens=(100, 713)), rec(agent_tokens=(100, 80))], CAPS)
    assert s["ending_turns"] == 2 and s["ending_turns_over_300_tokens"] == 1


def test_only_query_runs_completed_in_every_run_are_followed():
    runs = {"a": [rec("q1"), rec("q2")], "b": [rec("q1", steps=2), rec("q2", error="boom")], "c": [rec("q1", steps=1), rec("q2")]}
    t = trajectory(runs, {l: CAPS for l in runs})
    assert [r["id"] for r in t["rows"]] == ["q1"] and t["summary"]["large"]["paired"] == 1
    assert [t["summary"]["large"][l]["steps_mean"] for l in "abc"] == [3, 2, 1]


def test_hallucination_uses_judged_sentences_only_and_hits_are_reported():
    runs = {"a": [rec(verdicts=("unjudged", "unsupported", "supported", "supported"), cited=("BBBBBBBBBBB@5",))],
            "b": [rec(verdicts=("supported", "unsupported"))]}
    s = trajectory(runs, {l: CAPS for l in runs})["summary"]["large"]
    assert s["a"]["hallucination_rate"] == 1 / 3 and s["a"]["unjudged_sentences"] == 1
    assert s["a"]["video_hit"] == 0 and s["a"]["segment_hit"] == 0
    assert s["b"]["hallucination_rate"] == 0.5 and s["b"]["video_hit"] == 1 and s["b"]["segment_hit"] == 1


def test_markdown_has_a_row_per_run_and_a_per_query_arrow_chain():
    runs = {"base": [rec("q1")], "v1": [rec("q1", steps=2)], "v2": [rec("q1", steps=1)]}
    md = to_markdown(trajectory(runs, {l: CAPS for l in runs}), {l: l + "-dir" for l in runs})
    assert md.count("| large | 1 |") == 3 and "| q1 | large | 3 -> 2 -> 1 |" in md
    assert "drafts" in md and "n/a" in md  # drafts are n/a for runs without recorded turns
