"""Re-judge answers of a finished run, without re-running the agent.

  python -m eval.rejudge --results eval/results/<run> --judge-model <openrouter id> --name <short>
      Re-judges every answer with a different model (an alternative judge); writes judge_<name>.jsonl.

  python -m eval.rejudge --results eval/results/<run> --only-unjudged
      Re-judges only the answers that have sentences the judge gave no verdict for, with the main
      judge (the large model) and the current judge code (bigger token budget, batched retry).
      Writes judge_rejudged.jsonl and rejudge_report.json (coverage and hallucination rate before
      and after). records.jsonl, summary.json and summary.md are NOT changed: the original run stays
      as it was, and the sidecar says what the re-judge found.

Only :free models unless ALLOW_PAID=true.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any

from atlandex_agent.config import Settings
from atlandex_agent.llm import LLMError, LLMUnavailable, OpenRouterClient

from .agreement import LABELS_DIR
from .judge import judge_answer
from .metrics import aggregate, record_arch

SIDECAR = "judge_rejudged.jsonl"
REPORT = "rejudge_report.json"


def has_unjudged(rec: dict[str, Any]) -> bool:
    sentences = (rec.get("judge") or {}).get("sentences", [])
    return any(s["verdict"] not in ("supported", "unsupported") for s in sentences)


def merge_rejudged(records: list[dict[str, Any]], sidecar: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `records` where each re-judged answer carries its new judge result.

    A sidecar row without a result (the judge was unavailable) leaves the old judgement in place,
    so a failed re-judge can never turn an unjudged sentence into a judged one.
    """
    new = {(row["id"], row["model"], row.get("arch") or "single"): row["judge"] for row in sidecar if row.get("judge")}
    key = lambda r: (r["id"], r["model"], record_arch(r))  # noqa: E731  (a sidecar row without arch is a single one)
    return [{**r, "judge": new[key(r)]} if key(r) in new else r for r in records]


def coverage_report(records: list[dict[str, Any]], tolerance_sec: int = 60) -> dict[str, Any]:
    """Judge coverage and hallucination rate per agent model and source, headline records only."""
    recs = [{**r, "usage": r.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
             "steps": r.get("steps", 0), "latency_sec": r.get("latency_sec", 0.0)} for r in records]
    keep = ("judge_answers", "answers_fully_unjudged", "answers_partly_unjudged", "judge_sentences",
            "unjudged_sentences", "unjudged_sentence_share", "unjudged_answer_share", "judged_sentences",
            "hallucination_rate", "hallucination_rate_upper_bound")
    return {key: {k: s[k] for k in keep} for key, s in aggregate(recs, tolerance_sec)["headline"].items()}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def report_table(before: dict[str, Any], after: dict[str, Any]) -> str:
    lines = ["| model | source | | answers judged | answers with unjudged | sentences | unjudged sentences | "
             "hallucination (judged only) | upper bound |", "|" + "---|" * 9]
    for key in before:
        m, src = key.split("|")
        for label, rep in (("before", before), ("after", after)):
            s = rep[key]
            lines.append(f"| {m} | {src} | {label} | {s['judge_answers']} | "
                         f"{s['answers_fully_unjudged'] + s['answers_partly_unjudged']} | {s['judge_sentences']} | "
                         f"{s['unjudged_sentences']} ({_pct(s['unjudged_sentence_share'])}) | "
                         f"{_pct(s['hallucination_rate'])} ({s['judged_sentences']} sent.) | "
                         f"{_pct(s['hallucination_rate_upper_bound'])} |")
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = OpenRouterClient(settings)
    judge_model = args.judge_model or settings.model_large
    llm = client.bind(judge_model)
    results = Path(args.results)
    records = [json.loads(line) for line in (results / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    out = results / (SIDECAR if args.only_unjudged else f"judge_{args.name}.jsonl")
    done = 0
    only = None
    if args.labelled_only:  # just the answers behind the rows in eval/labels (saves free-tier requests)
        with (LABELS_DIR / "judge_verdicts.csv").open(encoding="utf-8", newline="") as vf:
            only = {(row["record_id"], row["model"], row.get("arch") or "single") for row in csv.DictReader(vf)}
    sidecar: list[dict[str, Any]] = []
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            if r.get("error") or r.get("clarification"):
                continue
            if args.arch and record_arch(r) != args.arch:
                continue
            if only is not None and (r["id"], r["model"], record_arch(r)) not in only:
                continue
            if args.only_unjudged and not has_unjudged(r):
                continue
            try:
                j = await judge_answer(llm, r["answer"], r["segments"])
            except (LLMUnavailable, LLMError) as e:
                row = {"id": r["id"], "model": r["model"], "arch": record_arch(r), "judge": None,
                       "error": f"{type(e).__name__}: {e}"}
                f.write(json.dumps(row) + "\n")
                sidecar.append(row)
                continue
            row = {"id": r["id"], "model": r["model"], "arch": record_arch(r), "judge": j}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            sidecar.append(row)
            done += 1
            print(f"[{r['id']}] {r['model']}: {'skipped' if j is None else len(j['sentences'])} sentences", flush=True)
    await client.aclose()
    print(f"wrote {out} ({done} answers); paid spend ${client.paid_spend_usd:.4f}")
    if args.only_unjudged:
        tol = json.loads((results / "summary.json").read_text(encoding="utf-8")).get("gold_tolerance_sec", 60)
        scope = [r for r in records if not args.arch or record_arch(r) == args.arch]
        before = coverage_report(scope, tol)
        after = coverage_report(merge_rejudged(scope, sidecar), tol)
        (results / REPORT).write_text(json.dumps({"judge_model": judge_model, "before": before, "after": after},
                                                 indent=2), encoding="utf-8")
        print(report_table(before, after))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--judge-model", default=None, help="default: the large model (the main judge)")
    p.add_argument("--name", default=None)
    p.add_argument("--labelled-only", action="store_true",
                   help="only re-judge the answers that contain rows of eval/labels/judge_verdicts.csv")
    p.add_argument("--arch", choices=["single", "multi"], default=None,
                   help="only re-judge the records of this architecture (default: all of them)")
    p.add_argument("--only-unjudged", action="store_true",
                   help="re-judge only answers that have unjudged sentences; writes a sidecar and a report")
    args = p.parse_args()
    if not args.only_unjudged and not (args.judge_model and args.name):
        p.error("--judge-model and --name are required unless --only-unjudged is given")
    if args.only_unjudged and not args.arch:
        archs = {record_arch(json.loads(l)) for l in
                 (Path(args.results) / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        if len(archs) > 1:
            p.error("this folder holds several architectures: add --arch single or --arch multi to --only-unjudged")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
