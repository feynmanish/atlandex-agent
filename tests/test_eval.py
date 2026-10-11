import asyncio
import csv
import json

import pytest

from atlandex_agent.citations import drop_unsupported, unsupported_ids
from atlandex_agent.segments import extract_citations, parse_tool_output, segment_id, split_sentences
from eval import agreement, sample_for_labels
from eval.judge import answer_sentences, build_judge_input, judge_answer
from eval.metrics import (aggregate, hallucination_rate, percentile, router_correct, segment_matches,
                          summarize, task_success)
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
def test_task_success_needs_one_gold_segment():
    assert task_success([A, "x@1"], [A, "y@2"])
    assert not task_success(["x@1"], [A])
    assert not task_success([], [A])


def test_gold_tolerance_matches_same_video_only():
    assert not segment_matches(f"{VID_A}@130", A)
    assert segment_matches(f"{VID_A}@130", A, tolerance_sec=30)
    assert not segment_matches(f"BBBBBBBBBBB@130", A, tolerance_sec=30)


def test_bare_video_id_gold_matches_any_segment_of_that_video():
    assert segment_matches(f"{VID_A}@999", VID_A) and segment_matches(f"{VID_A}@na1", VID_A)
    assert not segment_matches("BBBBBBBBBBB@5", VID_A)
    assert not segment_matches(f"{VID_A}@na1", A, tolerance_sec=60)  # timed gold needs a timed citation


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


def rec(model="m", source="existing", review=False, cited=(A,), gold=(A,), route="simple", exp="simple",
        verdicts=("supported",), lat=2.0, steps=2, error=None):
    return {"id": "q", "model": model, "source": source, "needs_review": review, "cited_ids": list(cited),
            "gold_segment_ids": list(gold), "route": route, "expected_route": exp, "steps": steps,
            "latency_sec": lat, "error": error, "citation_check": {},
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.0},
            "judge": {"sentences": [{"verdict": v} for v in verdicts], "usage": {"cost_usd": 0.0}}}


def test_summarize_computes_each_headline_metric():
    s = summarize([rec(), rec(cited=("x@1",), route="multi_step", verdicts=("unsupported", "supported")),
                   rec(error="boom")])
    assert s["n"] == 3 and s["errors"] == 1
    assert s["task_success"] == pytest.approx(1 / 3)  # the errored record counts as a failure
    assert s["router_accuracy"] == pytest.approx(1 / 2)
    assert s["hallucination_rate"] == pytest.approx(1 / 3)
    assert s["steps_mean"] == 2 and s["latency_p50"] == 2.0


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
    (tmp_path / "to_label.csv").write_text("row_id,question,answer_sentence,cited_segment_text,human_label\n0,q,s,t,\n")
    (tmp_path / "judge_verdicts.csv").write_text("row_id,judge_verdict\n0,supported\n")
    human, judge = agreement.load_pairs(tmp_path)
    assert agreement.compute(human, judge) == {"status": "not yet labelled", "n": 0}


def test_agreement_reads_labels_from_csv(tmp_path):
    (tmp_path / "to_label.csv").write_text(
        "row_id,question,answer_sentence,cited_segment_text,human_label\n0,q,s,t,supported\n1,q,s,t,U\n2,q,s,t,\n")
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
    assert "supported" not in text.replace("human_label", "")  # verdict lives in the other file
    assert "judge_verdict" in (tmp_path / "labels" / "judge_verdicts.csv").read_text()
