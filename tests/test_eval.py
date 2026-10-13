import asyncio
import csv
import json

import pytest

from atlandex_agent.citations import drop_unsupported, unsupported_ids
from atlandex_agent.segments import extract_citations, parse_tool_output, segment_id, split_sentences
from eval import agreement, sample_for_labels
from eval.judge import JUDGE_MAX_TOKENS, answer_sentences, build_judge_input, judge_answer, parse_verdicts
from eval.metrics import (aggregate, hallucination_rate, judge_coverage, percentile, router_correct,
                          segment_hit, summarize, video_hit)
from eval.rejudge import coverage_report, has_unjudged, merge_rejudged, report_table
from eval.run import portable_args, send_scores, summary_markdown
from fakes import FakeLLM, VID_A, search_json

A = f"{VID_A}@120"


# --- segments and citations ------------------------------------------------
def test_segment_id_and_unknown_start():
    assert segment_id(VID_A, 120) == A
    assert segment_id(VID_A, None, 2) == f"{VID_A}@na2"


def test_split_sentences_keeps_trailing_citations_with_their_sentence():
    s = split_sentences(f"First claim. [{A}] Second claim [{A}]. Third one.")
    assert len(s) == 3 and s[0].endswith(f"[{A}]") and s[2] == "Third one."


def test_split_sentences_ignores_decimals():
    assert len(split_sentences("It grew 2.5 times in one year.")) == 1


def test_extract_and_check_citations():
    ans = f"One [{A}]. Two [ZZZZZZZZZZZ@9]."
    assert extract_citations(ans) == [A, "ZZZZZZZZZZZ@9"]
    assert unsupported_ids(ans, [A]) == ["ZZZZZZZZZZZ@9"]
    assert unsupported_ids(ans, [A, "ZZZZZZZZZZZ@9"]) == []


def test_normalize_citations_handles_loose_styles():
    from atlandex_agent.segments import normalize_citations
    other = f"{VID_A[:-1]}B@7"
    assert normalize_citations(f"x【{A}】 y ({A}) z [{A}, {other}]") == f"x[{A}] y [{A}] z [{A}] [{other}]"
    assert normalize_citations("a [not an id] b") == "a [not an id] b"
    assert normalize_citations("**[xmkSf5IS‑zw@1879]**") == "**[xmkSf5IS-zw@1879]**"  # non-breaking hyphen


def test_drop_unsupported_keeps_supported_sentences_and_lines():
    ans = f"- Good [{A}].\n- Bad [ZZZZZZZZZZZ@9].\n- Also good [{A}]."
    cleaned, dropped = drop_unsupported(ans, [A])
    assert cleaned == f"- Good [{A}].\n- Also good [{A}]." and len(dropped) == 1


def test_parse_tool_output_passes_errors_through():
    segs, text = parse_tool_output("search_video", "Error executing tool: no searchable transcript")
    assert segs == [] and "no searchable transcript" in text


def test_parse_search_video_builds_segments():
    segs, text = parse_tool_output("search_video", search_json(VID_A, (120, "hello world")))
    assert [s.id for s in segs] == [A] and f"[{A}]" in text and segs[0].kind == "passage"


# --- metrics ----------------------------------------------------------------
def test_video_hit_needs_a_cited_segment_of_a_gold_video():
    assert video_hit([A, "x@1"], [VID_A])
    assert video_hit([f"{VID_A}@na1"], [VID_A])  # an untimed passage still proves the right video
    assert not video_hit(["BBBBBBBBBBB@5"], [VID_A])
    assert not video_hit([], [VID_A])


SPAN = {"video": VID_A, "start": 100, "end": 150}


def test_segment_hit_uses_the_span_widened_by_the_tolerance():
    assert segment_hit([f"{VID_A}@120"], [SPAN])
    assert segment_hit([f"{VID_A}@40"], [SPAN], tolerance_sec=60) and not segment_hit([f"{VID_A}@39"], [SPAN], 60)
    assert segment_hit([f"{VID_A}@210"], [SPAN], 60) and not segment_hit([f"{VID_A}@211"], [SPAN], 60)
    assert not segment_hit([f"{VID_A}@40"], [SPAN], tolerance_sec=0)


def test_segment_hit_rejects_other_videos_and_untimed_citations():
    assert not segment_hit(["BBBBBBBBBBB@120"], [SPAN])
    assert not segment_hit([f"{VID_A}@na1"], [SPAN], 60)  # no start time can never be a segment hit
    assert not segment_hit([], [SPAN])

def test_router_correct():
    assert router_correct("simple", "simple") and not router_correct("simple", "multi_step")
    assert not router_correct(None, "simple")


def test_hallucination_rate_ignores_unjudged():
    assert hallucination_rate(["supported", "unsupported", "unsupported", "unjudged"]) == pytest.approx(2 / 3)
    assert hallucination_rate([]) is None


def test_percentiles():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 50) == 3.0 and percentile(xs, 95) == pytest.approx(4.8)
    assert percentile([], 50) is None


def rec(model="m", source="existing", review=False, cited=(A,), route="simple", exp="simple",
        verdicts=("supported",), lat=2.0, steps=2, error=None):
    return {"id": "q", "model": model, "source": source, "needs_review": review, "cited_ids": list(cited),
            "gold_video_ids": [VID_A], "gold_segments": [{"video": VID_A, "start": 100, "end": 150}],
            "route": route, "expected_route": exp, "steps": steps,
            "latency_sec": lat, "error": error, "citation_check": {},
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.0},
            "judge": {"sentences": [{"verdict": v} for v in verdicts], "usage": {"cost_usd": 0.0}}}


def test_summarize_computes_each_headline_metric():
    s = summarize([rec(), rec(cited=("x@1",), route="multi_step", verdicts=("unsupported", "supported")),
                   rec(error="boom")])
    assert s["n"] == 3 and s["errors"] == 1
    assert s["completed"] == 2
    assert s["video_hit_rate"] == pytest.approx(1 / 2)  # the outage-failed run is not scored as a miss
    assert s["segment_hit_rate"] == pytest.approx(1 / 2) and s["segment_hit_n"] == 2
    assert "task_success" not in s
    assert s["router_accuracy"] == pytest.approx(1 / 2)
    assert s["hallucination_rate"] == pytest.approx(1 / 3)
    assert s["steps_mean"] == 2 and s["latency_p50"] == 2.0


def test_judge_coverage_counts_unjudged_answers_and_sentences():
    c = judge_coverage([["supported", "unsupported"], ["unjudged", "unjudged", "unjudged"],
                        ["supported", "unjudged"]])
    assert c["judge_answers"] == 3 and c["answers_fully_unjudged"] == 1 and c["answers_partly_unjudged"] == 1
    assert c["unjudged_answer_share"] == pytest.approx(2 / 3)
    assert c["judge_sentences"] == 7 and c["unjudged_sentences"] == 4
    assert c["unjudged_sentence_share"] == pytest.approx(4 / 7)
    # the upper bound counts every unjudged sentence as unsupported: (1 unsupported + 4 unjudged) / 7
    assert c["hallucination_rate_upper_bound"] == pytest.approx(5 / 7)


def test_judge_coverage_with_nothing_sent_to_the_judge_is_none_not_zero():
    c = judge_coverage([])
    assert c["unjudged_sentence_share"] is None and c["hallucination_rate_upper_bound"] is None


def test_fully_unjudged_answers_are_never_counted_as_clean():
    s = summarize([rec(verdicts=("unjudged", "unjudged"))])
    assert s["hallucination_rate"] is None  # not 0%
    assert s["hallucination_rate_upper_bound"] == 1.0
    assert s["unjudged_sentence_share"] == 1.0 and s["answers_fully_unjudged"] == 1 and s["judged_sentences"] == 0


def test_summarize_reports_partial_coverage_next_to_the_rate():
    s = summarize([rec(verdicts=("supported", "unsupported", "unjudged", "unjudged"))])
    assert s["hallucination_rate"] == pytest.approx(1 / 2)  # judged sentences only
    assert s["hallucination_rate_upper_bound"] == pytest.approx(3 / 4)
    assert s["unjudged_sentence_share"] == pytest.approx(1 / 2) and s["answers_partly_unjudged"] == 1


class _Args:
    gold_tolerance_sec = 60
    model = "small"
    subset = None


def test_summary_command_has_no_absolute_paths():
    import argparse
    import tempfile
    from pathlib import Path

    from atlandex_agent.config import ROOT

    outside = Path(tempfile.gettempdir()) / "x.jsonl"  # absolute on every OS, and not inside the repo
    a = argparse.Namespace(dataset=str(ROOT / "eval" / "dataset.jsonl"), resume=str(ROOT / "eval" / "results" / "r1"),
                           model="both", subset=None, other=str(outside))
    out = portable_args(a)
    assert out["dataset"] == "eval/dataset.jsonl" and out["resume"] == "eval/results/r1"
    assert out["other"] == "x.jsonl"  # outside the repo: only the file name is kept
    assert out["model"] == "both" and out["subset"] is None
    assert str(ROOT) not in repr(out)


def test_summary_markdown_states_the_unjudged_share_and_warns_above_ten_percent(tmp_path):
    weak = summary_markdown(aggregate([rec(verdicts=("supported", "unjudged"))]), tmp_path, _Args())
    assert "50% of 2 sent., 1 of 1 ans." in weak
    assert "<= 50% if unjudged were unsupported" in weak
    assert "Judge coverage warning" in weak
    full = summary_markdown(aggregate([rec(verdicts=("supported",) * 20)]), tmp_path, _Args())
    assert "0% of 20 sent., 0 of 1 ans." in full and "Judge coverage warning" not in full


class _Scores:
    def __init__(self):
        self.sent = {}

    def score(self, trace_id, name, value, comment=None):
        self.sent[name] = (value, comment)


def test_langfuse_scores_never_give_an_unjudged_answer_a_clean_hallucination_score():
    r = rec(verdicts=("unjudged", "unjudged"))
    r["trace_id"] = "t" * 32
    t = _Scores()
    send_scores(t, r, 60)
    assert "hallucination_rate" not in t.sent  # absent, not 0.0
    assert t.sent["unjudged_share"][0] == 1.0


def test_langfuse_hallucination_score_mentions_unjudged_sentences_it_left_out():
    r = rec(verdicts=("supported", "unsupported", "unjudged"))
    r["judge"]["sentences"][1].update(text="Bad claim.", reason="not in passage")
    r["trace_id"] = "t" * 32
    t = _Scores()
    send_scores(t, r, 60)
    value, comment = t.sent["hallucination_rate"]
    assert value == pytest.approx(1 / 2) and "1 of 3 sentences unjudged" in comment
    assert t.sent["unjudged_share"][0] == pytest.approx(1 / 3)


def test_summarize_counts_uncited_sentences_flagged_and_dropped():
    a = dict(rec(), answer="One claim here. Another claim there.",
             citation_check={"uncited_first": ["x", "y", "z"], "dropped_uncited": ["z"]})
    s = summarize([a, rec()])
    assert s["uncited_flagged"] == 3 and s["uncited_dropped"] == 1


def test_needs_review_records_never_enter_the_headline():
    agg = aggregate([rec(), rec(source="new_candidate", review=True)])
    assert list(agg["headline"]) == ["m|existing"]
    assert agg["unverified_not_in_headline"]["m"]["n"] == 1


# --- judge -------------------------------------------------------------------
def test_judge_input_shows_cited_text_and_flags_unretrieved():
    segs = {A: {"text": "scaling laws predict loss"}}
    out = build_judge_input([f"Claim one [{A}].", "Claim two [ZZZZZZZZZZZ@9]."], segs)
    assert "scaling laws predict loss" in out and "was not retrieved" in out and f"[{A}]" in out.split("Cited")[1]


def test_judge_returns_per_sentence_verdicts_in_one_call():
    llm = FakeLLM({"judge": ['{"verdicts": [{"i": 0, "verdict": "supported", "reason": "ok"},'
                             '{"i": 1, "verdict": "unsupported", "reason": "not in text"}]}']})
    res = asyncio.run(judge_answer(llm, f"First [{A}]. Second [{A}].", {A: {"text": "t"}}))
    assert [s["verdict"] for s in res["sentences"]] == ["supported", "unsupported"]
    assert len(llm.calls) == 1


def test_judge_marks_missing_verdicts_unjudged_after_one_retry():
    llm = FakeLLM({"judge": ["garbage"]})
    res = asyncio.run(judge_answer(llm, f"First [{A}].", {A: {"text": "t"}}))
    assert res["sentences"][0]["verdict"] == "unjudged" and len(llm.calls) == 2


def v(i, verdict="supported", reason="ok"):
    return '{"i": %d, "verdict": "%s", "reason": "%s"}' % (i, verdict, reason)


FOUR = f"One [{A}]. Two [{A}]. Three [{A}]. Four [{A}]."


def test_judge_asks_with_a_larger_token_budget():
    llm = FakeLLM({"judge": ['{"verdicts": [' + v(0) + ']}']})
    asyncio.run(judge_answer(llm, f"First [{A}].", {A: {"text": "t"}}))
    assert llm.max_tokens == [JUDGE_MAX_TOKENS] and JUDGE_MAX_TOKENS >= 4000


def test_parse_verdicts_keeps_complete_verdicts_of_a_cut_off_reply():
    cut = '{"verdicts": [' + v(0) + ', ' + v(1, "unsupported", 'says \\"x\\" which is not in the text') + ', {"i": 2, "verdi'
    got = parse_verdicts(cut)
    assert sorted(got) == [0, 1] and got[1]["verdict"] == "unsupported" and "not in the text" in got[1]["reason"]
    assert parse_verdicts("") == {} and parse_verdicts("no json at all") == {}


def test_judge_keeps_what_a_cut_off_reply_delivered_and_re_asks_only_for_the_rest():
    llm = FakeLLM({"judge": ['{"verdicts": [' + v(0) + ', ' + v(1) + ', {"i": 2, "verd',
                             '{"verdicts": [' + v(0, "unsupported") + ']}']})
    res = asyncio.run(judge_answer(llm, f"One [{A}]. Two [{A}]. Three [{A}].", {A: {"text": "t"}}))
    assert [s["verdict"] for s in res["sentences"]] == ["supported", "supported", "unsupported"]
    assert len(llm.calls) == 2
    retry_prompt = llm.prompts[1][1][1]["content"]
    assert "Three" in retry_prompt and "One" not in retry_prompt and "Two" not in retry_prompt
    assert llm.max_tokens == [JUDGE_MAX_TOKENS, JUDGE_MAX_TOKENS]


def test_judge_retries_an_empty_reply_in_batches_of_three_and_maps_indices_back():
    llm = FakeLLM({"judge": ["",  # reasoning ate the budget: empty reply
                             '{"verdicts": [' + v(0) + ', ' + v(1) + ', ' + v(2, "unsupported") + ']}',
                             '{"verdicts": [' + v(0, "unsupported") + ']}']})
    res = asyncio.run(judge_answer(llm, FOUR, {A: {"text": "t"}}))
    assert [s["verdict"] for s in res["sentences"]] == ["supported", "supported", "unsupported", "unsupported"]
    assert len(llm.calls) == 3  # all four, then [0,1,2], then [3]
    assert "Four" in llm.prompts[2][1][1]["content"] and "Three" not in llm.prompts[2][1][1]["content"]


def test_judge_never_turns_a_failed_retry_into_a_verdict():
    llm = FakeLLM({"judge": ["", "", ""]})
    res = asyncio.run(judge_answer(llm, FOUR, {A: {"text": "t"}}))
    assert [s["verdict"] for s in res["sentences"]] == ["unjudged"] * 4
    assert len(llm.calls) == 3  # one full request, then two retry batches


def test_rejudge_picks_only_answers_with_unjudged_sentences():
    assert has_unjudged(rec(verdicts=("supported", "unjudged")))
    assert not has_unjudged(rec(verdicts=("supported", "unsupported")))
    assert not has_unjudged({"judge": None})


def test_merge_rejudged_replaces_only_rejudged_answers_and_keeps_old_judgement_on_failure():
    a, b, c = (dict(rec(verdicts=("unjudged", "unjudged")), id=i) for i in "abc")
    new = {"model": "j", "sentences": [{"verdict": "supported"}, {"verdict": "unsupported"}], "usage": {}}
    merged = merge_rejudged([a, b, c], [{"id": "a", "model": "m", "judge": new},
                                         {"id": "b", "model": "m", "judge": None, "error": "LLMUnavailable"}])
    assert merged[0]["judge"] is new
    assert merged[1]["judge"] is b["judge"] and merged[2]["judge"] is c["judge"]  # still unjudged, not clean
    assert a["judge"]["sentences"][0]["verdict"] == "unjudged"  # the input records are not modified


def test_coverage_report_before_and_after_a_rejudge():
    old = [dict(rec(verdicts=("supported", "unsupported")), id="a"), dict(rec(verdicts=("unjudged", "unjudged")), id="b")]
    new = {"model": "j", "sentences": [{"verdict": "unsupported"}, {"verdict": "supported"}], "usage": {}}
    before = coverage_report(old)["m|existing"]
    after = coverage_report(merge_rejudged(old, [{"id": "b", "model": "m", "judge": new}]))["m|existing"]
    assert before["unjudged_sentences"] == 2 and before["unjudged_sentence_share"] == 0.5
    assert before["hallucination_rate"] == 0.5 and before["hallucination_rate_upper_bound"] == 0.75
    assert after["unjudged_sentences"] == 0 and after["hallucination_rate"] == 0.5
    assert "| before |" in report_table({"m|existing": before}, {"m|existing": after})


def test_judge_skips_the_no_coverage_reply():
    from atlandex_agent.graph import NO_COVERAGE
    assert asyncio.run(judge_answer(FakeLLM({"judge": ["x"]}), NO_COVERAGE, {})) is None


def test_answer_sentences_split_over_lines():
    assert len(answer_sentences("- One.\n- Two.")) == 2


# --- agreement ---------------------------------------------------------------
def test_kappa_perfect_chance_and_degenerate():
    assert agreement.cohens_kappa(["s", "u", "s", "u"], ["s", "u", "s", "u"]) == 1.0
    assert agreement.cohens_kappa(["s", "s", "u", "u"], ["s", "u", "s", "u"]) == pytest.approx(0.0)
    assert agreement.cohens_kappa(["s", "s"], ["s", "s"]) == 1.0


def test_kappa_known_value():
    human = ["s"] * 20 + ["u"] * 5 + ["s"] * 10 + ["u"] * 15
    judge = ["s"] * 20 + ["s"] * 5 + ["u"] * 10 + ["u"] * 15
    # po = 35/50 = .7; pe = (.5*.6)+(.5*.4)... computed by hand below
    po = 35 / 50
    pe = (25 / 50) * (30 / 50) + (25 / 50) * (20 / 50)
    assert agreement.cohens_kappa(human, judge) == pytest.approx((po - pe) / (1 - pe))
    r = agreement.compute(human, judge)
    assert r["agreement"] == pytest.approx(0.7) and r["n"] == 50


def test_agreement_reports_not_yet_labelled(tmp_path):
    (tmp_path / "to_label.csv").write_text("row_id,question,answer_sentence,cited_segment_text,reference_label\n0,q,s,t,\n")
    (tmp_path / "judge_verdicts.csv").write_text("row_id,judge_verdict\n0,supported\n")
    human, judge = agreement.load_pairs(tmp_path)
    assert agreement.compute(human, judge) == {"status": "not yet labelled", "n": 0}


def test_agreement_reads_labels_from_csv(tmp_path):
    (tmp_path / "to_label.csv").write_text(
        "row_id,question,answer_sentence,cited_segment_text,reference_label\n0,q,s,t,supported\n1,q,s,t,U\n2,q,s,t,\n")
    (tmp_path / "judge_verdicts.csv").write_text(
        "row_id,judge_verdict\n0,supported\n1,supported\n2,unsupported\n")
    human, judge = agreement.load_pairs(tmp_path)
    assert human == ["supported", "unsupported"] and agreement.compute(human, judge)["agreement"] == 0.5


def test_sample_for_labels_hides_the_verdict_and_skips_unreviewed(tmp_path, monkeypatch):
    run_dir = tmp_path / "20260101-000000"
    run_dir.mkdir()
    base = {"model": "m", "question": "q", "segments": {A: {"text": "seg text"}}}
    rows = [dict(base, id="a", needs_review=False, judge={"sentences": [
                {"text": f"S1 [{A}].", "cited_ids": [A], "verdict": "supported", "reason": "r"},
                {"text": "S2.", "cited_ids": [], "verdict": "unjudged", "reason": ""}]}),
            dict(base, id="b", needs_review=True, judge={"sentences": [
                {"text": "S3.", "cited_ids": [], "verdict": "supported", "reason": ""}]})]
    (run_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(sample_for_labels, "LABELS_DIR", tmp_path / "labels")
    monkeypatch.setattr("sys.argv", ["x", "--n", "5", "--results", str(run_dir)])
    sample_for_labels.main()
    text = (tmp_path / "labels" / "to_label.csv").read_text()
    assert "S1" in text and "S3" not in text and "S2" not in text
    assert "supported" not in text.replace("reference_label", "")  # verdict lives in the other file
    assert "judge_verdict" in (tmp_path / "labels" / "judge_verdicts.csv").read_text()


def test_resume_merge_keeps_finished_records_when_only_a_subset_is_rerun():
    from eval.run import merge_records
    ds = [{'id': 'a'}, {'id': 'b'}]
    kept = {('a', 'small', 'single'): {'id': 'a', 'model_alias': 'small', 'x': 1},
            ('b', 'large', 'single'): {'id': 'b', 'model_alias': 'large'}}
    new = [{'id': 'b', 'model_alias': 'small', 'x': 2}, {'id': 'a', 'model_alias': 'small', 'x': 9}]
    out = merge_records(kept, new, ds)
    assert [(r['id'], r['model_alias']) for r in out] == [('a', 'small'), ('b', 'small'), ('b', 'large')]
    assert out[0]['x'] == 9  # a re-run record replaces the old one
