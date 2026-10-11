"""python -m eval.run --model small|large|both --subset N

Runs the agent over eval/dataset.jsonl, judges each answer with the large model,
and writes eval/results/<timestamp>/{records.jsonl,summary.json,summary.md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from atlandex_agent.config import ROOT, Settings
from atlandex_agent.graph import run_agent
from atlandex_agent.llm import (BudgetExceeded, LLMError, LLMUnavailable, OpenRouterClient,
                                PaidCallBlocked)
from atlandex_agent.mcp_tools import open_toolbox

from .judge import judge_answer
from .metrics import aggregate

DEFAULT_DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(dataset: list[dict], subset: int | None, ids: list[str] | None, include_review: bool) -> list[dict]:
    rows = [r for r in dataset if include_review or not r.get("needs_review")]
    if ids:
        rows = [r for r in dataset if r["id"] in ids]
    return rows[:subset] if subset else rows


def fmt(x: float | None, pct: bool = False, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def summary_markdown(agg: dict[str, Any], results_dir: Path, args: argparse.Namespace) -> str:
    tol = args.gold_tolerance_sec
    head = (f"| model | source | n | video hit | segment hit (±{tol} s) | router acc. | hallucination | steps | "
            "latency p50/p95 (s) | tokens in/out | cost $ |")
    sep = "|" + "---|" * 11
    lines = [f"# Eval run {results_dir.name}", "",
             f"Command: `python -m eval.run --model {args.model}"
             f"{' --subset ' + str(args.subset) if args.subset else ''}`", "",
             "*Video hit* = the answer cites a segment of a gold video. *Segment hit* = it cites a segment whose start "
             f"is inside a gold span (±{tol} s). They are different measures; neither is called task success.", "",
             "## Headline (needs_review records excluded)", "", head, sep]

    def row(key: str, s: dict) -> str:
        m, src = key.split("|")
        seg = f"{fmt(s['segment_hit_rate'], True)} (n={s['segment_hit_n']})"
        return (f"| {m} | {src} | {s['n']} | {fmt(s['video_hit_rate'], True)} | {seg} | "
                f"{fmt(s['router_accuracy'], True)} | "
                f"{fmt(s['hallucination_rate'], True)} ({s['judged_sentences']} sent.) | {fmt(s['steps_mean'], nd=1)} | "
                f"{fmt(s['latency_p50'], nd=1)} / {fmt(s['latency_p95'], nd=1)} | "
                f"{fmt(s['prompt_tokens_mean'], nd=0)} / {fmt(s['completion_tokens_mean'], nd=0)} | "
                f"{s['cost_usd_total']:.4f} |")

    lines += [row(k, s) for k, s in agg["headline"].items()] or ["| (no verified records in this run) |"]
    lines += ["", "## Unverified (needs_review: true), NOT part of the headline", "", head, sep]
    lines += [row(f"{m}|new_candidate", s) for m, s in agg["unverified_not_in_headline"].items()] or ["| none |"]
    return "\n".join(lines) + "\n"


async def amain(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.no_cache:
        settings.cache_enabled = False
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2
    dataset = load_dataset(Path(args.dataset))
    rows = select(dataset, args.subset, args.ids.split(",") if args.ids else None, args.include_review)
    models = ["small", "large"] if args.model == "both" else [args.model]

    results_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir.mkdir(parents=True)
    records_path = results_dir / "records.jsonl"
    client = OpenRouterClient(settings)
    judge_llm = client.bind(settings.model_large)
    records: list[dict[str, Any]] = []
    stop: Exception | None = None

    async with open_toolbox(settings) as tools:
        for row in rows:
            for which in models:
                llm = client.bind(settings.model_for(which))
                rec: dict[str, Any] = {k: row.get(k) for k in
                                       ("id", "question", "gold_video_ids", "gold_segments", "expected_route",
                                        "source", "needs_review", "verified_by")}
                rec["model_alias"] = which
                try:
                    res = await run_agent(row["question"], llm, tools, settings)
                    rec.update(res)
                    if not res["clarification"]:
                        rec["judge"] = await judge_answer(judge_llm, res["answer"], res["segments"])
                except (PaidCallBlocked, BudgetExceeded) as e:
                    stop = e
                    rec.update(model=llm.model, error=f"{type(e).__name__}: {e}")
                except (LLMUnavailable, LLMError) as e:
                    rec.update(model=llm.model, error=f"{type(e).__name__}: {e}")
                records.append(rec)
                with records_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{rec['id']}] {which}: " + (rec.get("error") or
                      f"route={rec['route']} steps={rec['steps']} cited={len(rec['cited_ids'])}"), flush=True)
                if stop:
                    break
            if stop:
                break
    await client.aclose()

    # records with a hard error carry no usage; aggregate() skips them via the error field.
    for r in records:
        r.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        r.setdefault("steps", 0)
        r.setdefault("latency_sec", 0.0)
        r["model"] = r.get("model") or settings.model_for(r["model_alias"])
    agg = aggregate(records, args.gold_tolerance_sec)
    summary = {"command": vars(args), "models": {a: settings.model_for(a) for a in models},
               "judge_model": settings.model_large, "gold_tolerance_sec": args.gold_tolerance_sec,
               "paid_spend_usd": round(client.paid_spend_usd, 6), "n_records": len(records),
               "stopped_early": str(stop) if stop else None, **agg}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "summary.md").write_text(summary_markdown(agg, results_dir, args), encoding="utf-8")
    print(f"\nResults: {results_dir}\nPaid spend this run: ${client.paid_spend_usd:.4f}")
    return 1 if stop else 0


def main() -> None:
    p = argparse.ArgumentParser(prog="eval.run", description=__doc__)
    p.add_argument("--model", default="both", choices=["small", "large", "both"])
    p.add_argument("--subset", type=int, default=None, help="run only the first N selected queries")
    p.add_argument("--ids", default=None, help="comma-separated record ids (overrides subset selection)")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--include-review", action="store_true",
                   help="also run needs_review records (they are still excluded from headline numbers)")
    p.add_argument("--gold-tolerance-sec", type=int, default=60,
                   help="a cited segment counts as gold if same video and start within N seconds "
                        "(bare video-ID gold entries match any segment of that video)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
