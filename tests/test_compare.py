"""eval.compare pairs the same query-runs of two runs and never mixes an unjudged sentence into a rate."""
import json

import pytest

from eval.compare import compare, completed, load, to_markdown


def rec(id_="q1", alias="small", steps=4, lat=60.0, tin=5000, tout=900, verdicts=("supported", "unsupported"), **kw):
    r = {"id": id_, "model_alias": alias, "model": "m", "steps": steps, "latency_sec": lat,
         "usage": {"prompt_tokens": tin, "completion_tokens": tout, "cost_usd": 0.0, "llm_calls": 5},
         "judge": {"sentences": [{"verdict": v} for v in verdicts]}, "citation_check": {}, "needs_review": False}
    r.update(kw)
    return r


def test_only_query_runs_completed_in_both_runs_are_paired():
    base = [rec("q1"), rec("q2"), rec("q3", error="LLMUnavailable")]
    fixed = [rec("q1", steps=2), rec("q3"), rec("q4")]
    c = compare(base, fixed)
    assert [r["id"] for r in c["rows"]] == ["q1"]
    s = c["summary"]["all"]
    assert s["paired"] == 1 and s["baseline_completed"] == 2 and s["fixed_completed"] == 3
    assert s["before"]["steps_mean"] == 4 and s["after"]["steps_mean"] == 2


def test_hallucination_uses_judged_sentences_only_and_unjudged_are_counted_apart():
    base = [rec(verdicts=("supported", "unjudged", "unjudged", "unsupported"))]
    fixed = [rec(verdicts=("supported", "supported", "unsupported", "unsupported"))]
    s = compare(base, fixed)["summary"]["all"]
    assert s["before"]["hallucination_rate"] == 0.5 and s["before"]["unjudged_sentences"] == 2
    assert s["after"]["hallucination_rate"] == 0.5 and s["after"]["unjudged_sentences"] == 0
    assert compare([rec(verdicts=("unjudged",))], [rec()])["summary"]["all"]["before"]["hallucination_rate"] is None


def test_needs_review_runs_are_left_out():
    assert completed([rec(needs_review=True), rec("q2")]).keys() == {("q2", "small")}


def test_uncited_counts_come_from_the_citation_check():
    fixed = [rec(citation_check={"uncited_first": ["a", "b", "c"], "dropped_uncited": ["c"]})]
    s = compare([rec()], fixed)["summary"]["all"]["after"]
    assert s["uncited_flagged"] == 3 and s["uncited_dropped"] == 1


def test_baseline_verdicts_come_from_the_rejudge_sidecar_when_present(tmp_path):
    base = rec(verdicts=("unjudged", "unjudged"))
    (tmp_path / "records.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
    new = {"model": "j", "sentences": [{"verdict": "supported"}, {"verdict": "unsupported"}], "usage": {}}
    (tmp_path / "judge_rejudged.jsonl").write_text(json.dumps({"id": "q1", "model": "m", "judge": new}) + "\n",
                                                   encoding="utf-8")
    assert [s["verdict"] for s in load(tmp_path, use_sidecar=True)[0]["judge"]["sentences"]] == ["supported", "unsupported"]
    assert load(tmp_path)[0]["judge"]["sentences"][0]["verdict"] == "unjudged"  # the records file itself is untouched


def test_markdown_has_a_summary_and_one_row_per_paired_query():
    md = to_markdown(compare([rec("q1"), rec("q2")], [rec("q1", steps=2), rec("q2", steps=1)]), "old", "new")
    assert "## Summary" in md and "| q1 | small | 4 -> 2 |" in md and "| q2 | small | 4 -> 1 |" in md
