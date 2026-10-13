"""The eval harness with two architectures: keys, order, summaries, the paired comparison and the free-request guard."""
import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import httpx
import pytest

from atlandex_agent.config import ROOT, Settings
from atlandex_agent.llm import LLMError, OpenRouterClient
from atlandex_agent.tracing import NullTracer
from eval.compare import compare, completed, load_selected, split_spec, to_markdown
from eval.metrics import (aggregate, aggregate_by_arch, bootstrap_diff, discordant, record_arch, sign_test_p,
                          summarize)
from eval.rejudge import merge_rejudged
from eval.run import (FreeRequestsLow, build_summary, merge_records, record_key, run_config, run_order, run_queries,
                      summary_markdown)
from fakes import FakeLLM, FakeToolBox, VID_A, VID_B, find_json, json_reply, search_json, tool_turn
from test_multi_agent import ANSWER, NONE_FLAGGED

A = f"{VID_A}@120"


def rec(id_="q1", alias="small", arch=None, verdicts=("supported", "unsupported"), route="simple", **kw):
    r = {"id": id_, "model_alias": alias, "model": "m", "source": "existing", "needs_review": False,
         "steps": 2, "latency_sec": 30.0, "route": route, "answer": f"One [{A}]. Two [{A}].",
         "cited_ids": [A], "gold_video_ids": [VID_A], "gold_segments": [{"video": VID_A, "start": 100, "end": 150}],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost_usd": 0.0, "llm_calls": 4},
         "llm_calls": [{"role": "agent", "attempts": 2, "cached": False}, {"role": "answer", "attempts": 1,
                                                                            "cached": True}],
         "judge": {"sentences": [{"verdict": v} for v in verdicts]}, "citation_check": {}}
    if arch:
        r["arch"] = arch
    r.update(kw)
    return r


# --- keys and merging -------------------------------------------------------------------------------------------

def test_a_record_without_arch_is_a_single_record_everywhere():
    assert record_arch(rec()) == "single" and record_key(rec()) == ("q1", "small", "single")
    assert record_key(rec(arch="multi")) == ("q1", "small", "multi")


def test_rerunning_only_the_errored_multi_records_keeps_every_finished_single_record():
    dataset = [{"id": "a"}, {"id": "b"}]
    finished = [rec("a", arch="single"), rec("a", arch="multi"), rec("b", arch="single")]
    kept = {record_key(r): r for r in finished}  # b/multi had errored, so --resume did not keep it
    out = merge_records(kept, [rec("b", arch="multi", note="redone")], dataset)
    assert [record_key(r) for r in out] == [("a", "small", "single"), ("a", "small", "multi"),
                                            ("b", "small", "single"), ("b", "small", "multi")]
    assert out[3]["note"] == "redone" and out[2] is finished[2]


def test_run_order_runs_both_arms_back_to_back_and_alternates_who_goes_first():
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    order = run_order(rows, ["large"], ["single", "multi"])
    assert [(r["id"], arch) for r, _, arch in order] == [
        ("a", "single"), ("a", "multi"), ("b", "multi"), ("b", "single"), ("c", "single"), ("c", "multi")]
    two_models = run_order(rows[:1], ["small", "large"], ["single", "multi"])
    assert [(m, a) for _, m, a in two_models] == [("small", "single"), ("small", "multi"),
                                                  ("large", "multi"), ("large", "single")]
    assert [a for _, _, a in run_order(rows, ["large"], ["multi"])] == ["multi"] * 3


# --- metrics ------------------------------------------------------------------------------------------------------

def test_aggregate_of_records_without_arch_is_unchanged():
    records = [rec("q1"), rec("q2", verdicts=("supported",) * 3)]
    s = aggregate(records)["headline"]["m|existing"]
    assert s["completed"] == 2 and s["segment_hit_rate"] == 1.0 and s["video_hit_rate"] == 1.0
    assert s["hallucination_rate"] == pytest.approx(1 / 5) and s["judged_sentences"] == 5
    assert s["steps_mean"] == 2 and s["latency_p50"] == 30 and s["prompt_tokens_mean"] == 1000
    assert aggregate_by_arch(records)["single"]["headline"] == aggregate(records)["headline"]


def test_fixed_v2_headline_is_recomputed_exactly_from_its_records():
    d = ROOT / "eval" / "results" / "20261012-fixed-v2"
    if not (d / "records.jsonl").exists():  # raw records are not committed (see .gitignore)
        pytest.skip("fixed v2 records.jsonl is local only")
    records = [json.loads(l) for l in (d / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in records:
        r.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        r.setdefault("steps", 0)
        r.setdefault("latency_sec", 0.0)
    committed = json.loads((d / "summary.json").read_text(encoding="utf-8"))["headline"]
    new = aggregate(records)["headline"]
    assert {k: {m: v for m, v in s.items() if m in committed[k]} for k, s in new.items()} == committed


def test_summarize_reports_calls_attempts_and_the_deletion_guards():
    from atlandex_agent.graph import NO_COVERAGE
    multi = rec(arch="multi", verify={"dropped_unverified": ["x", "y"], "failed": True, "skipped_budget": False},
                supervisor_turns=[{"fallback": True}, {"fallback": False}])
    empty = rec("q2", arch="multi", answer=NO_COVERAGE, verify={"skipped_budget": True})
    s = summarize([multi, empty])
    assert s["llm_calls_mean"] == 4 and s["llm_calls_max"] == 4
    assert s["http_attempts_total"] == 4  # the cached call made no request: (2 + 0) per record
    assert s["final_sentences_mean"] == 1.5 and s["no_coverage_answers"] == 1
    assert (s["verify_dropped"], s["verify_failed"], s["verify_skipped_budget"], s["supervisor_fallbacks"]) == (2, 1, 1, 1)
    assert s["budget_forced_runs"] == 0
    forced = rec("q3", arch="multi", notes=["call budget: forced answer"])
    assert summarize([multi, forced])["budget_forced_runs"] == 1


def test_sign_test_and_discordant_pairs():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(0.0625)
    assert sign_test_p(0, 5) == pytest.approx(0.0625)
    assert sign_test_p(3, 3) == 1.0 and sign_test_p(1, 0) == 1.0
    assert sign_test_p(10, 1) == pytest.approx(2 * 12 / 2048)
    assert discordant([(True, True), (True, False), (False, True), (False, False), (True, False)]) == (2, 1)


def test_bootstrap_diff_resamples_queries_and_is_deterministic_for_a_seed():
    a = [(1, 4), (0, 3), (2, 5), (1, 2), (0, 4)]
    b = [(0, 4), (0, 3), (1, 5), (0, 2), (0, 4)]
    first, again = bootstrap_diff(a, b, n=500, seed=3), bootstrap_diff(a, b, n=500, seed=3)
    assert first == again and first != bootstrap_diff(a, b, n=500, seed=4)
    assert first["rate_a"] == pytest.approx(4 / 18) and first["rate_b"] == pytest.approx(1 / 18)
    assert first["diff"] == pytest.approx(-3 / 18) and first["lo"] <= first["diff"] <= first["hi"]
    assert bootstrap_diff([(0, 3)] * 4, [(0, 3)] * 4)["lo"] == 0 == bootstrap_diff([(0, 3)] * 4, [(0, 3)] * 4)["hi"]
    assert bootstrap_diff([], [])["diff"] is None


# --- summary.json ---------------------------------------------------------------------------------------------------

def args_ns(**kw):
    base = dict(model="both", arch="both", subset=None, gold_tolerance_sec=60, dataset=str(ROOT / "eval" / "dataset.jsonl"),
                resume=None, min_free_requests=60)
    base.update(kw)
    return argparse.Namespace(**base)


def test_summary_json_has_by_arch_and_a_run_config_without_absolute_paths(tmp_path):
    s = Settings(openrouter_api_key="k", cache_enabled=False)
    records = [rec("q1", arch="single"), rec("q1", arch="multi", verdicts=("supported",))]
    summary, agg, by_arch = build_summary(args_ns(), s, ["small"], ["single", "multi"], records, 0.0, None)
    assert list(by_arch) == ["single", "multi"] and set(summary["by_arch"]) == {"single", "multi"}
    assert summary["headline"] == by_arch["single"]["headline"]
    cfg = summary["run_config"]
    assert set(cfg["prompt_sha256"]) == {"ROUTER_SYSTEM", "CLARIFY_SYSTEM", "AGENT_SYSTEM", "ANSWER_SYSTEM",
                                         "SUPERVISOR_SYSTEM", "VERIFY_SYSTEM", "judge_prompt.md"}
    assert all(len(h) == 64 for h in cfg["prompt_sha256"].values())
    assert cfg["caps"] == {"router_clarify": 1500, "agent_turn_and_supervisor_large": 1500,
                           "agent_turn_and_supervisor_small": 600, "answer": 1500, "answer_empty_retry": 4000,
                           "verify": 1500, "judge": 4000}
    assert (cfg["max_tool_steps"], cfg["max_llm_calls_per_run"], cfg["search_steps_per_task"],
            cfg["max_supervisor_turns"], cfg["evidence_char_budget"], cfg["cache_enabled"]) == (6, 13, 3, 3, 12000, False)
    assert cfg["models"] == {"small": s.model_small} and cfg["judge_model"] == s.model_large
    assert "git_commit" in cfg and cfg["git_dirty"] in (True, False, None)
    text = json.dumps(summary)
    assert str(ROOT) not in text and str(Path(tempfile.gettempdir())) not in text
    md = summary_markdown(agg, tmp_path, args_ns(), by_arch)
    assert "## Headline, single architecture" in md and "## Headline, multi architecture" in md
    assert "LLM calls mean / max" in md


def test_a_single_architecture_run_has_no_by_arch_and_the_old_top_level():
    records = [rec("q1"), rec("q2")]
    summary, agg, by_arch = build_summary(args_ns(arch="single"), Settings(), ["small"], ["single"], records, 0.0, None)
    assert "by_arch" not in summary and summary["headline"] == aggregate(records)["headline"]


def test_a_resumed_run_keeps_its_first_config_and_says_what_changed():
    s = Settings(openrouter_api_key="k")
    first = run_config(s, ["small"], ["single"])
    s.search_steps_per_task = 2
    summary, _, _ = build_summary(args_ns(), s, ["small"], ["single"], [rec()], 0.0, None, previous_config=first)
    assert summary["run_config"]["search_steps_per_task"] == 3
    assert "search_steps_per_task" in summary["run_config_changed_on_resume"]


# --- compare / trajectory selectors ------------------------------------------------------------------------------

def write_run(tmp_path, records):
    d = tmp_path / "R"
    d.mkdir()
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return d


def test_compare_pairs_single_and_multi_arms_within_one_folder(tmp_path):
    d = write_run(tmp_path, [
        rec("q1", arch="single"), rec("q1", arch="multi", steps=3, route="multi_step"),
        rec("q2", arch="single"), rec("q2", arch="multi", error="LLMUnavailable"),  # q2 completed in one arm only
        rec("q3", arch="single", cited_ids=["BBBBBBBBBBB@5"]), rec("q3", arch="multi")])
    single, n1 = load_selected(f"{d}:single")
    multi, n2 = load_selected(f"{d}:multi")
    assert (n1, n2) == ("R:single", "R:multi") and {record_arch(r) for r in single} == {"single"}
    c = compare(single, multi)
    assert [r["id"] for r in c["rows"]] == ["q1", "q3"]
    s = c["summary"]["all"]
    assert s["paired"] == 2 and s["baseline_completed"] == 3 and s["fixed_completed"] == 2
    assert s["route_disagreements"] == ["q1"]
    assert s["segment_hit_pairs"] == {"before_only": 0, "after_only": 1, "sign_test_p": 1.0}
    assert s["before"]["segment_hit_rate"] == 0.5 and s["after"]["segment_hit_rate"] == 1.0
    assert s["before"]["llm_calls_mean"] == 4 and s["after"]["final_sentences_mean"] == 2
    assert "small|existing" in c["summary"]
    md = to_markdown(c, n1, n2)
    assert "## Quality and guards" in md and "route disagreements" in md and "(q1)" in md


def test_a_folder_with_both_architectures_needs_a_selector(tmp_path):
    d = write_run(tmp_path, [rec("q1", arch="single"), rec("q1", arch="multi")])
    with pytest.raises(SystemExit):
        load_selected(str(d))
    (tmp_path / "old").mkdir()
    old = write_run(tmp_path / "old", [rec("q1")])
    assert len(load_selected(str(old))[0]) == 1  # a folder from before the variant is all single
    assert len(load_selected(f"{old}:single")[0]) == 1 and load_selected(f"{old}:multi")[0] == []


def test_split_spec_does_not_mistake_a_windows_drive_letter_for_a_selector():
    assert split_spec("C:\\runs\\x") == (Path("C:\\runs\\x"), None)
    assert split_spec("C:\\runs\\x:multi") == (Path("C:\\runs\\x"), "multi")
    assert split_spec("eval/results/r:single") == (Path("eval/results/r"), "single")
    assert split_spec("eval/results/r") == (Path("eval/results/r"), None)
    assert completed([rec("q1", arch="multi")]).keys() == {("q1", "small")}


# --- rejudge sidecar --------------------------------------------------------------------------------------------------

def test_rejudge_sidecar_keeps_the_two_architectures_apart_and_old_sidecars_still_merge():
    new_s = {"model": "j", "sentences": [{"verdict": "supported"}], "usage": {}}
    new_m = {"model": "j", "sentences": [{"verdict": "unsupported"}], "usage": {}}
    records = [rec("q1", arch="single"), rec("q1", arch="multi"), rec("q2")]  # q2 has no arch
    merged = merge_rejudged(records, [{"id": "q1", "model": "m", "arch": "multi", "judge": new_m},
                                      {"id": "q2", "model": "m", "judge": new_s}])  # an old sidecar row: no arch
    assert merged[0]["judge"] is records[0]["judge"] and merged[1]["judge"] is new_m and merged[2]["judge"] is new_s


# --- the free-request guard ---------------------------------------------------------------------------------------------

def key_client(handler):
    s = Settings(openrouter_api_key="sk-or-v1-TEST", cache_dir=Path(tempfile.gettempdir()) / "unused")
    return OpenRouterClient(s, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_key_status_reads_the_free_model_allowance_and_sends_only_a_get_to_key():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, req.headers["authorization"]))
        return httpx.Response(200, json={"data": {"free_model_daily_requests": {"used": 40, "limit": 1000, "remaining": 960}}})

    st = asyncio.run(key_client(handler).key_status())
    assert st == {"daily_limit": 1000, "used_today": 40, "remaining": 960}
    assert seen == [("GET", "/api/v1/key", "Bearer sk-or-v1-TEST")]
    with pytest.raises(LLMError):
        asyncio.run(key_client(lambda r: httpx.Response(200, json={"data": {}})).key_status())


class GuardClient:
    """Stands in for OpenRouterClient in run_queries: one FakeLLM serves every role and key_status is scripted."""

    def __init__(self, llm, remaining):
        self.llm, self.remaining, self.checks = llm, list(remaining), 0
        self.paid_spend_usd = 0.0

    def bind(self, model):
        return self.llm

    async def key_status(self):
        self.checks += 1
        left = self.remaining.pop(0) if len(self.remaining) > 1 else self.remaining[0]
        if isinstance(left, Exception):
            raise left
        return {"daily_limit": 1000, "used_today": 1000 - left, "remaining": left}


def both_arms_llm() -> FakeLLM:
    return FakeLLM({
        "router": ['{"route": "simple", "reason": "r"}'], "supervisor": [ANSWER], "agent": ["READY"],
        "answer": [f"Scaling laws predict loss [{A}]."], "verify": [NONE_FLAGGED],
        "judge": [json_reply({"verdicts": [{"i": 0, "verdict": "supported", "reason": "ok"}]})]})


def run_loop(client, rows, tmp_path, minimum=60):
    tools = FakeToolBox({"find_videos": find_json((VID_A, 60, True)),
                         "search_video": search_json(VID_A, (120, "scaling laws predict loss"))})
    args = args_ns(min_free_requests=minimum, run_id="r1")
    settings = Settings(openrouter_api_key="k", cache_enabled=False)
    return asyncio.run(run_queries(args, settings, client, tools, NullTracer(), rows, ["large"], ["single", "multi"],
                                   {}, tmp_path / "records.jsonl"))


# each question links the video, so every run reads it first and one scripted reply per role serves all runs
ROWS = [{"id": "q1", "question": f"About https://youtu.be/{VID_A}: what about scaling?", "source": "existing"},
        {"id": "q2", "question": f"About https://youtu.be/{VID_A}: and data?", "source": "existing"}]


def test_both_arms_run_back_to_back_in_alternating_order_through_one_judge(tmp_path):
    llm = both_arms_llm()
    records, stop = run_loop(GuardClient(llm, [500]), ROWS, tmp_path)
    assert stop is None and [(r["id"], r["arch"]) for r in records] == [
        ("q1", "single"), ("q1", "multi"), ("q2", "multi"), ("q2", "single")]
    assert all(r["judge"]["sentences"][0]["verdict"] == "supported" for r in records)
    assert all(r["llm_calls_used"] <= 13 for r in records)
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_the_guard_stops_before_the_next_query_with_both_arms_of_the_last_one_complete(tmp_path):
    llm = both_arms_llm()
    client = GuardClient(llm, [500, 59])  # plenty before q1, below the 60 minimum before q2
    records, stop = run_loop(client, ROWS, tmp_path)
    assert isinstance(stop, FreeRequestsLow) and "59" in str(stop) and "q2" in str(stop)
    assert [(r["id"], r["arch"]) for r in records] == [("q1", "single"), ("q1", "multi")]
    assert client.checks == 2  # once per (query, model), not once per arm


def test_the_guard_stops_before_any_chat_request_when_the_allowance_is_already_low(tmp_path):
    llm = both_arms_llm()
    records, stop = run_loop(GuardClient(llm, [10]), ROWS, tmp_path)
    assert isinstance(stop, FreeRequestsLow) and records == [] and llm.prompts == []


def test_a_guard_that_cannot_read_the_allowance_does_not_stop_the_run(tmp_path):
    llm = both_arms_llm()
    records, stop = run_loop(GuardClient(llm, [LLMError("no body")]), ROWS[:1], tmp_path)
    assert stop is None and len(records) == 2
    records, stop = run_loop(GuardClient(both_arms_llm(), [0]), ROWS[:1], tmp_path, minimum=0)  # guard switched off
    assert stop is None and len(records) == 2
