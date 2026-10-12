"""python -m eval.compare --baseline eval/results/<old> --fixed eval/results/<new>

Compares two finished runs of the same dataset, query by query: steps, latency, tokens, hallucination rate,
unjudged and uncited sentences. Writes comparison.md and comparison.json into the fixed run's folder.

Only query-runs that completed in BOTH runs are paired; the summary rows also say how many were paired.
If the baseline folder has judge_rejudged.jsonl (from `eval.rejudge --only-unjudged`), the baseline's
verdicts are taken from it, so the hallucination rate of both runs is measured with the same judge code.
Records stay untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from .metrics import hallucination_rate, percentile
from .rejudge import SIDECAR, merge_rejudged


def load(path: Path, use_sidecar: bool = False) -> list[dict[str, Any]]:
    records = [json.loads(l) for l in (path / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    side = path / SIDECAR
    if use_sidecar and side.exists():
        records = merge_rejudged(records, [json.loads(l) for l in side.read_text(encoding="utf-8").splitlines() if l.strip()])
    return records


def completed(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["id"], r["model_alias"]): r for r in records
            if not r.get("error") and not r.get("needs_review") and "usage" in r}


def verdicts(r: dict[str, Any]) -> list[str]:
    return [s["verdict"] for s in (r.get("judge") or {}).get("sentences", [])]


def row_stats(r: dict[str, Any]) -> dict[str, Any]:
    v = verdicts(r)
    cc = r.get("citation_check") or {}
    return {
        "steps": r["steps"], "latency": r["latency_sec"],
        "tokens_in": r["usage"]["prompt_tokens"], "tokens_out": r["usage"]["completion_tokens"],
        "llm_calls": r["usage"]["llm_calls"] if "llm_calls" in r["usage"] else None,
        "hallucination": hallucination_rate(v),
        "judged": sum(x in ("supported", "unsupported") for x in v), "unjudged": sum(x == "unjudged" for x in v),
        "uncited_dropped": len(cc.get("dropped_uncited", [])), "uncited_flagged": len(cc.get("uncited_first", [])),
    }


def compare(baseline: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> dict[str, Any]:
    b, f = completed(baseline), completed(fixed)
    keys = sorted(set(b) & set(f))
    rows = [{"id": k[0], "model_alias": k[1], "before": row_stats(b[k]), "after": row_stats(f[k])} for k in keys]
    summary: dict[str, Any] = {}
    for alias in sorted({k[1] for k in keys}) + ["all"]:
        sel = [k for k in keys if alias == "all" or k[1] == alias]
        out: dict[str, Any] = {"paired": len(sel), "baseline_completed": sum(1 for k in b if alias in ("all", k[1])),
                               "fixed_completed": sum(1 for k in f if alias in ("all", k[1]))}
        for label, src in (("before", b), ("after", f)):
            recs = [src[k] for k in sel]
            st = [row_stats(r) for r in recs]
            lat = [s["latency"] for s in st]
            out[label] = {
                "steps_mean": mean(s["steps"] for s in st) if st else None,
                "latency_median": median(lat) if lat else None, "latency_p95": percentile(lat, 95),
                "tokens_in_mean": mean(s["tokens_in"] for s in st) if st else None,
                "tokens_out_mean": mean(s["tokens_out"] for s in st) if st else None,
                "hallucination_rate": hallucination_rate(x for r in recs for x in verdicts(r)),
                "judged_sentences": sum(s["judged"] for s in st), "unjudged_sentences": sum(s["unjudged"] for s in st),
                "uncited_flagged": sum(s["uncited_flagged"] for s in st),
                "uncited_dropped": sum(s["uncited_dropped"] for s in st),
            }
        summary[alias] = out
    return {"rows": rows, "summary": summary}


def _f(x: float | None, nd: int = 1, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def to_markdown(c: dict[str, Any], baseline_name: str, fixed_name: str) -> str:
    lines = [f"# Before / after: `{baseline_name}` (baseline) vs `{fixed_name}` (fixed version)", "",
             "Paired query-runs only (completed in both). Latency is wall-clock seconds per query-run; tokens are the "
             "agent's own calls (the judge is not included). Hallucination = judged sentences marked unsupported; "
             "the baseline uses the re-judged verdicts when available, so both sides come from the same judge code.", "",
             "## Summary", "",
             "| agent model | paired | | steps | latency p50 / p95 (s) | tokens in / out | hallucination (judged sent.) | "
             "unjudged sent. | uncited flagged / dropped |", "|" + "---|" * 9]
    for alias, s in c["summary"].items():
        for label in ("before", "after"):
            x = s[label]
            lines.append(f"| {alias} | {s['paired']} of {s['baseline_completed']}/{s['fixed_completed']} | {label} | "
                         f"{_f(x['steps_mean'])} | {_f(x['latency_median'])} / {_f(x['latency_p95'])} | "
                         f"{_f(x['tokens_in_mean'], 0)} / {_f(x['tokens_out_mean'], 0)} | "
                         f"{_f(x['hallucination_rate'], pct=True)} ({x['judged_sentences']}) | {x['unjudged_sentences']} | "
                         f"{x['uncited_flagged']} / {x['uncited_dropped']} |")
    lines += ["", "(*paired* = query-runs completed in both runs, then completed in baseline / completed in fixed run.)", "",
              "## Per query", "",
              "| query | model | steps | latency (s) | tokens in | tokens out | hallucination (judged) | unjudged | uncited dropped |",
              "|" + "---|" * 9]
    for r in c["rows"]:
        a, z = r["before"], r["after"]
        lines.append(f"| {r['id']} | {r['model_alias']} | {a['steps']} -> {z['steps']} | {a['latency']:.0f} -> {z['latency']:.0f} | "
                     f"{a['tokens_in']} -> {z['tokens_in']} | {a['tokens_out']} -> {z['tokens_out']} | "
                     f"{_f(a['hallucination'], pct=True)} -> {_f(z['hallucination'], pct=True)} | "
                     f"{a['unjudged']} -> {z['unjudged']} | {z['uncited_dropped']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True)
    p.add_argument("--fixed", required=True)
    a = p.parse_args()
    base, fixed = Path(a.baseline), Path(a.fixed)
    c = compare(load(base, use_sidecar=True), load(fixed))
    (fixed / "comparison.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    (fixed / "comparison.md").write_text(to_markdown(c, base.name, fixed.name), encoding="utf-8")
    print(to_markdown(c, base.name, fixed.name).split("## Per query")[0])
    print(f"wrote {fixed / 'comparison.md'}")


if __name__ == "__main__":
    main()
