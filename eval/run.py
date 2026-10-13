"""python -m eval.run --model small|large|both --arch single|multi|both --subset N

Runs the agent over eval/dataset.jsonl, judges each answer with the large model,
and writes eval/results/<timestamp>/{records.jsonl,summary.json,summary.md}.

With --arch both, the single and the multi-agent graph run back to back for each (query, model), alternating which
goes first, through one shared throttle and one judge, so the two arms see the same provider load.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from atlandex_agent import prompts
from atlandex_agent.config import ROOT, Settings
from atlandex_agent.graph import EmptyAnswerError, run_agent
from atlandex_agent.llm import (BudgetExceeded, CallBudgetExceeded, LLMError, LLMUnavailable, OpenRouterClient,
                                PaidCallBlocked)
from atlandex_agent.mcp_tools import open_toolbox
from atlandex_agent.tracing import TraceInfo, tracer_for

from .judge import JUDGE_MAX_TOKENS, JUDGE_PROMPT, judge_answer
from .metrics import (UNJUDGED_WARN_SHARE, aggregate, aggregate_by_arch, hallucination_rate, record_arch,
                      router_correct, segment_hit, video_hit)

# A run that ends with one of these is recorded as failed (not as a miss) and is re-run by --resume.
RUN_ERRORS = (LLMUnavailable, LLMError, EmptyAnswerError, CallBudgetExceeded)


class FreeRequestsLow(RuntimeError):
    """The key has fewer free-model requests left today than --min-free-requests: the run stops cleanly."""


DEFAULT_DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_MIN_FREE_REQUESTS = 60  # a query pair needs up to ~30 requests; the margin also covers 429 retries


def portable_args(args: argparse.Namespace) -> dict[str, Any]:
    """The command-line arguments for summary.json, with paths inside the repo made repo-relative.

    Absolute paths would leak the local user name and folder layout into files that are committed.
    """
    out = dict(vars(args))
    for key, value in out.items():
        if isinstance(value, str) and Path(value).is_absolute():
            try:
                out[key] = Path(value).resolve().relative_to(ROOT).as_posix()
            except ValueError:
                out[key] = Path(value).name
    return out


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(dataset: list[dict], subset: int | None, ids: list[str] | None, include_review: bool) -> list[dict]:
    rows = [r for r in dataset if include_review or not r.get("needs_review")]
    if ids:
        rows = [r for r in dataset if r["id"] in ids]
    return rows[:subset] if subset else rows


def run_order(rows: list[dict], models: list[str], arches: list[str]) -> list[tuple[dict, str, str]]:
    """Every (row, model, arch) to run, in order. The arms of one (query, model) run back to back, and which arm
    goes first alternates with the index of the (query, model) pair: even index first arch first, odd index last
    arch first. Free-tier load and backend state then hit both arms equally."""
    out: list[tuple[dict, str, str]] = []
    k = 0
    for row in rows:
        for which in models:
            out += [(row, which, a) for a in (arches if k % 2 == 0 else list(reversed(arches)))]
            k += 1
    return out


def fmt(x: float | None, pct: bool = False, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def _headline_row(key: str, s: dict) -> str:
    m, src = key.split("|")
    seg = f"{fmt(s['segment_hit_rate'], True)} (n={s['segment_hit_n']})"
    return (f"| {m} | {src} | {s['completed']}/{s['n']} | {fmt(s['video_hit_rate'], True)} | {seg} | "
            f"{fmt(s['router_accuracy'], True)} | "
            f"{fmt(s['hallucination_rate'], True)} ({s['judged_sentences']} sent.; "
            f"<= {fmt(s['hallucination_rate_upper_bound'], True)} if unjudged were unsupported) | "
            f"{fmt(s['unjudged_sentence_share'], True)} of {s['judge_sentences']} sent., "
            f"{s['answers_fully_unjudged'] + s['answers_partly_unjudged']} of {s['judge_answers']} ans. | "
            f"{fmt(s['steps_mean'], nd=1)} | "
            f"{fmt(s['latency_p50'], nd=1)} / {fmt(s['latency_p95'], nd=1)} | "
            f"{fmt(s['prompt_tokens_mean'], nd=0)} / {fmt(s['completion_tokens_mean'], nd=0)} | "
            f"{s['cost_usd_total']:.4f} |")


def _cost_rows(headline: dict[str, dict]) -> list[str]:
    lines = ["| model | source | LLM calls mean / max | HTTP attempts | final sentences / answer | no-coverage answers | "
             "verify dropped / failed / skipped | supervisor fallbacks | budget-forced runs |", "|" + "---|" * 9]
    for key, s in headline.items():
        m, src = key.split("|")
        top = s["llm_calls_max"] if s["llm_calls_max"] is not None else "n/a"
        lines.append(f"| {m} | {src} | {fmt(s['llm_calls_mean'], nd=1)} / {top} | "
                     f"{s['http_attempts_total']} | {fmt(s['final_sentences_mean'], nd=1)} | {s['no_coverage_answers']} | "
                     f"{s['verify_dropped']} / {s['verify_failed']} / {s['verify_skipped_budget']} | "
                     f"{s['supervisor_fallbacks']} | {s['budget_forced_runs']} |")
    return lines


def summary_markdown(agg: dict[str, Any], results_dir: Path, args: argparse.Namespace,
                     by_arch: dict[str, dict[str, Any]] | None = None) -> str:
    """The summary page. With `by_arch` (more than one architecture in the run) there is one headline table per
    architecture, each followed by a table of calls and the deletion guards; `agg` is then not shown."""
    tol = args.gold_tolerance_sec
    head = (f"| model | source | runs ok/total | video hit | segment hit (±{tol} s) | router acc. | "
            "hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |")
    sep = "|" + "---|" * 12
    arch_flag = f" --arch {args.arch}" if getattr(args, "arch", "single") != "single" else ""
    lines = [f"# Eval run {results_dir.name}", "",
             f"Command: `python -m eval.run --model {args.model}{arch_flag}"
             f"{' --subset ' + str(args.subset) if args.subset else ''}`", "",
             "*Video hit* = the answer cites a segment of a gold video. *Segment hit* = it cites a segment whose start "
             f"is inside a gold span (±{tol} s). They are different measures; neither is called task success. "
             "Rates are over completed runs; runs that failed because the free model stayed unavailable are "
             "shown in *runs ok/total*, not counted as misses.", ""]

    def warn(headline: dict[str, dict], prefix: str = "") -> list[str]:
        weak = [k for k, s in headline.items() if (s["unjudged_sentence_share"] or 0) > UNJUDGED_WARN_SHARE]
        if not weak:
            return []
        return ["", f"**Judge coverage warning{prefix}:** for {', '.join(weak)} more than {UNJUDGED_WARN_SHARE:.0%} of "
                    "sentences got no verdict from the judge. The hallucination rate covers judged sentences only; "
                    "unjudged ones are not counted as clean, and the upper bound counts them all as unsupported."]

    if not by_arch:
        lines += ["## Headline (needs_review records excluded)", "", head, sep]
        lines += [_headline_row(k, s) for k, s in agg["headline"].items()] or ["| (no verified records in this run) |"]
        lines += warn(agg["headline"])
        lines += ["", "## Unverified (needs_review: true), NOT part of the headline", "", head, sep]
        lines += [_headline_row(f"{m}|new_candidate", s) for m, s in agg["unverified_not_in_headline"].items()] \
            or ["| none |"]
        return "\n".join(lines) + "\n"

    for arch, a in by_arch.items():
        lines += [f"## Headline, {arch} architecture (needs_review records excluded)", "", head, sep]
        lines += [_headline_row(k, s) for k, s in a["headline"].items()] or ["| (no verified records in this run) |"]
        lines += warn(a["headline"], f" ({arch})")
        lines += ["", f"Calls and guards, {arch}: LLM calls are the agent's own (the judge is not included); HTTP "
                      "attempts include retries after 429/5xx.", ""] + _cost_rows(a["headline"]) + [""]
    for arch, a in by_arch.items():
        lines += [f"## Unverified (needs_review: true), {arch}, NOT part of the headline", "", head, sep]
        lines += [_headline_row(f"{m}|new_candidate", s) for m, s in a["unverified_not_in_headline"].items()] \
            or ["| none |"]
        lines.append("")
    return "\n".join(lines) + "\n"


def send_scores(tracer, rec: dict[str, Any], tolerance_sec: int) -> None:
    """Attach the eval results to the run's Langfuse trace (no-op when tracing is off).

    - hallucination_rate: share of judged answer sentences the judge marked unsupported (0 = clean);
      the comment lists the unsupported sentences. Not sent when no sentence was judged: an answer the
      judge gave no verdict for is never scored as clean.
    - unjudged_share: share of the answer's sentences the judge gave no verdict for (0 = fully judged).
    - task_success: the answer cites a gold segment (start within the tolerance), the project's definition.
      Only sent for records that have gold segments.
    - video_hit: the answer cites a segment of a gold video. Kept separate from task_success.
    - router_correct: the router picked the expected route.
    """
    tid = rec.get("trace_id")
    if not tid:
        return
    cited = rec.get("cited_ids", [])
    sentences = (rec.get("judge") or {}).get("sentences", [])
    rate = hallucination_rate(s["verdict"] for s in sentences)
    unjudged = sum(s["verdict"] not in ("supported", "unsupported") for s in sentences)
    if sentences:
        tracer.score(tid, "unjudged_share", unjudged / len(sentences),
                     f"{unjudged} of {len(sentences)} sentences unjudged")
    if rate is not None:
        bad = [f"- {s['text'][:200]} ({s['reason'][:200]})" for s in sentences if s["verdict"] == "unsupported"]
        note = f"\n({unjudged} of {len(sentences)} sentences unjudged, not counted)" if unjudged else ""
        tracer.score(tid, "hallucination_rate", rate,
                     ("\n".join(bad) or "all judged sentences supported") + note)
    if rec.get("gold_segments"):
        tracer.score(tid, "task_success", segment_hit(cited, rec["gold_segments"], tolerance_sec),
                     f"cites a gold segment within +/-{tolerance_sec}s")
    if rec.get("gold_video_ids"):
        tracer.score(tid, "video_hit", video_hit(cited, rec["gold_video_ids"]))
    if rec.get("expected_route"):
        tracer.score(tid, "router_correct", router_correct(rec.get("route"), rec["expected_route"]),
                     f"route={rec.get('route')} expected={rec['expected_route']}")


def record_key(r: dict[str, Any]) -> tuple[str, str, str]:
    """(query id, model alias, architecture): one query-run. A record without `arch` is a single-agent record."""
    return r["id"], r["model_alias"], record_arch(r)


def merge_records(kept: dict[tuple[str, str, str], dict[str, Any]], new: list[dict[str, Any]],
                  dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Finished records plus this pass's records (which win), in dataset order, small before large, single first."""
    order = {r["id"]: i for i, r in enumerate(dataset)}
    merged = {**kept, **{record_key(r): r for r in new}}
    return sorted(merged.values(),
                  key=lambda r: (order.get(r["id"], 10**9), r["model_alias"] != "small", record_arch(r) != "single"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_config(settings: Settings, models: list[str], arches: list[str]) -> dict[str, Any]:
    """Everything that shapes an arm's behaviour, so two runs can be compared knowing what differed: the git
    commit (and whether tracked files were modified), model ids, every token cap, the step/call limits, the cache
    flag and a SHA-256 of each prompt. Hashes, not texts: the prompts are in the commit. No absolute paths."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "archs": arches,
        "models": {a: settings.model_for(a) for a in models},
        "judge_model": settings.model_large,
        "temperature": settings.temperature,
        "allow_paid": settings.allow_paid,
        "caps": {
            "router_clarify": settings.max_tokens,
            "agent_turn_and_supervisor_large": settings.agent_max_tokens or settings.agent_max_tokens_large,
            "agent_turn_and_supervisor_small": settings.agent_max_tokens or settings.agent_max_tokens_small,
            "answer": settings.max_tokens,
            "answer_empty_retry": settings.answer_retry_max_tokens,
            "verify": settings.verify_max_tokens or settings.max_tokens,
            "judge": JUDGE_MAX_TOKENS,
        },
        "max_tool_steps": settings.max_tool_steps,
        "max_llm_calls_per_run": settings.max_llm_calls_per_run,
        "search_steps_per_task": settings.search_steps_per_task,
        "max_supervisor_turns": settings.max_supervisor_turns,
        "evidence_char_budget": settings.evidence_char_budget,
        "cache_enabled": settings.cache_enabled,
        "prompt_sha256": {**{name: _sha256(getattr(prompts, name)) for name in (
            "ROUTER_SYSTEM", "CLARIFY_SYSTEM", "AGENT_SYSTEM", "ANSWER_SYSTEM", "SUPERVISOR_SYSTEM", "VERIFY_SYSTEM")},
            "judge_prompt.md": _sha256(JUDGE_PROMPT)},
    }


async def free_requests_stop(client, minimum: int, before: str) -> FreeRequestsLow | None:
    """A FreeRequestsLow if the key has fewer than `minimum` free requests left today, else None. A check that
    cannot be made (network, unexpected reply) does not stop the run: it is reported and the run goes on."""
    if minimum <= 0:
        return None
    try:
        status = await client.key_status()
    except (httpx.HTTPError, LLMError) as e:
        print(f"[guard] could not read the free-request allowance ({type(e).__name__}); continuing", flush=True)
        return None
    if status["remaining"] < minimum:
        return FreeRequestsLow(f"{status['remaining']} of {status['daily_limit']} free requests left today, below "
                               f"--min-free-requests {minimum}; stopped before {before}")
    return None


async def run_queries(args: argparse.Namespace, settings: Settings, client, tools, tracer, rows: list[dict],
                      models: list[str], arches: list[str], kept: dict, records_path: Path
                      ) -> tuple[list[dict[str, Any]], Exception | None]:
    """The agent and judge loop. Returns (records of this pass, the exception that stopped the run early, if any).

    The free-request guard is checked once before the first arm of each (query, model): a stop therefore cuts both
    arms at the same point."""
    judge_llm = client.bind(settings.model_large)
    records: list[dict[str, Any]] = []
    stop: Exception | None = None
    last_group: tuple[str, str] | None = None
    for row, which, arch in run_order(rows, models, arches):
        if (row["id"], which, arch) in kept:
            continue
        if (row["id"], which) != last_group:
            last_group = (row["id"], which)
            stop = await free_requests_stop(client, args.min_free_requests, f"{row['id']} {which}")
            if stop:
                break
        llm = client.bind(settings.model_for(which))
        rec: dict[str, Any] = {k: row.get(k) for k in
                               ("id", "question", "gold_video_ids", "gold_segments", "expected_route",
                                "source", "needs_review", "verified_by")}
        rec["model_alias"] = which
        rec["arch"] = arch
        try:
            info = TraceInfo(run_id=args.run_id, query_id=row["id"], model_alias=which)
            res = await run_agent(row["question"], llm, tools, settings, trace=info, tracer=tracer, arch=arch)
            rec.update(res)
            if not res["clarification"]:
                rec["judge"] = await judge_answer(judge_llm, res["answer"], res["segments"])
            send_scores(tracer, rec, args.gold_tolerance_sec)
        except (PaidCallBlocked, BudgetExceeded) as e:
            stop = e
            rec.update(model=llm.model, error=f"{type(e).__name__}: {e}")
        except RUN_ERRORS as e:
            rec.update(model=llm.model, error=f"{type(e).__name__}: {e}")
        records.append(rec)
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{rec['id']}] {which}/{arch}: " + (rec.get("error") or
              f"route={rec['route']} steps={rec['steps']} cited={len(rec['cited_ids'])} "
              f"calls={rec['llm_calls_used']}"), flush=True)
        if stop:
            break
    return records, stop


def build_summary(args: argparse.Namespace, settings: Settings, models: list[str], arches: list[str],
                  records: list[dict[str, Any]], paid_spend_usd: float, stop: Exception | None,
                  previous_config: dict[str, Any] | None = None
                  ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """(summary.json content, the aggregate shown at the top level, the aggregate of each architecture).

    With one architecture the top level is exactly what it always was. With two, the top level is the first
    (single) one for readers of old summaries, `by_arch` holds both, and summary.md shows one table per arch."""
    by_arch = aggregate_by_arch(records, args.gold_tolerance_sec)
    agg = next(iter(by_arch.values())) if by_arch else aggregate(records, args.gold_tolerance_sec)
    summary = {"command": portable_args(args), "models": {a: settings.model_for(a) for a in models},
               "judge_model": settings.model_large, "gold_tolerance_sec": args.gold_tolerance_sec,
               "paid_spend_usd": round(paid_spend_usd, 6), "n_records": len(records),
               "stopped_early": str(stop) if stop else None, **agg,
               "run_config": run_config(settings, models, arches)}
    if len(by_arch) > 1:
        summary["by_arch"] = by_arch
    if previous_config:  # a resumed pass keeps the first pass's config and says what differs now
        now = summary["run_config"]
        changed = sorted(k for k in set(now) | set(previous_config) if now.get(k) != previous_config.get(k)
                         and k not in ("git_dirty",))
        summary["run_config"] = previous_config
        if changed:
            summary["run_config_changed_on_resume"] = changed
    return summary, agg, by_arch


async def amain(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.no_cache:
        settings.cache_enabled = False
    if args.no_trace:
        settings.tracing = False
    tracer = tracer_for(settings)
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2
    dataset = load_dataset(Path(args.dataset))
    rows = select(dataset, args.subset, args.ids.split(",") if args.ids else None, args.include_review)
    models = ["small", "large"] if args.model == "both" else [args.model]
    arches = ["single", "multi"] if args.arch == "both" else [args.arch]

    kept: dict[tuple[str, str, str], dict[str, Any]] = {}
    if args.resume:  # keep finished records (with their original latency), redo only the errored ones
        results_dir = Path(args.resume)
        for line in (results_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r.get("error"):
                kept[record_key(r)] = r
        (results_dir / "records.jsonl.bak").write_text(  # a resume must never be able to lose finished records
            (results_dir / "records.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        (results_dir / "records.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept.values()), encoding="utf-8")
    else:
        results_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir.mkdir(parents=True)
    args.run_id = results_dir.name
    records_path = results_dir / "records.jsonl"
    client = OpenRouterClient(settings)
    async with open_toolbox(settings) as tools:
        records, stop = await run_queries(args, settings, client, tools, tracer, rows, models, arches, kept,
                                          records_path)
    await client.aclose()
    tracer.flush()

    if args.resume:  # finished records stay, whatever subset or model this pass covered
        records = merge_records(kept, records, dataset)
    records_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    # records with a hard error carry no usage; aggregate() skips them via the error field.
    for r in records:
        r.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        r.setdefault("steps", 0)
        r.setdefault("latency_sec", 0.0)
        r["model"] = r.get("model") or settings.model_for(r["model_alias"])
    previous = None
    if args.resume and (results_dir / "summary.json").exists():
        previous = json.loads((results_dir / "summary.json").read_text(encoding="utf-8")).get("run_config")
    summary, agg, by_arch = build_summary(args, settings, models, arches, records, client.paid_spend_usd, stop,
                                          previous)
    if summary.get("run_config_changed_on_resume"):
        print(f"WARNING: the configuration changed since the first pass: {summary['run_config_changed_on_resume']}")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "summary.md").write_text(
        summary_markdown(agg, results_dir, args, by_arch if len(by_arch) > 1 else None), encoding="utf-8")
    print(f"\nResults: {results_dir}\nPaid spend this run: ${client.paid_spend_usd:.4f}")
    if stop:
        print(f"Stopped early: {stop}")
    return 1 if stop else 0


def main() -> None:
    p = argparse.ArgumentParser(prog="eval.run", description=__doc__)
    p.add_argument("--model", default="both", choices=["small", "large", "both"])
    p.add_argument("--arch", default="single", choices=["single", "multi", "both"],
                   help="which graph to run; 'both' runs the two back to back per query, alternating the order")
    p.add_argument("--subset", type=int, default=None, help="run only the first N selected queries")
    p.add_argument("--ids", default=None, help="comma-separated record ids (overrides subset selection)")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--include-review", action="store_true",
                   help="also run needs_review records (they are still excluded from headline numbers)")
    p.add_argument("--gold-tolerance-sec", type=int, default=60,
                   help="a cited segment counts as gold if same video and start within N seconds "
                        "(bare video-ID gold entries match any segment of that video)")
    p.add_argument("--min-free-requests", type=int, default=DEFAULT_MIN_FREE_REQUESTS,
                   help="stop cleanly before a query when the key has fewer than N free-model requests left "
                        "today (0 turns the guard off)")
    p.add_argument("--resume", default=None, metavar="RESULTS_DIR",
                   help="re-run only the errored query-runs of an earlier results directory and update it in place")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-trace", action="store_true", help="do not send traces/scores to Langfuse even if keys are set")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
