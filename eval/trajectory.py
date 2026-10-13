"""python -m eval.trajectory --run baseline=<dir> --run v1=<dir> --run v2=<dir> --caps baseline=1500:1500 --caps v1=600:600 --out <dir>

A run is DIR or DIR:single / DIR:multi (one architecture of a folder that holds both), as in eval.compare.

Follows the same queries through several runs (baseline -> fixed v1 -> fixed v2): steps, latency, tokens,
hallucination rate, hit rates, and what the agent turns did. Only query-runs completed in EVERY run are used,
per agent model. --caps label=LARGE:SMALL gives the agent-turn token caps in force for runs whose records do
not carry them (older runs, default 1500:600); newer records carry their own.

The first run's hallucination verdicts come from judge_rejudged.jsonl when present (same judge code as the
later runs). Writes trajectory.md and trajectory.json into --out. Records stay untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from .compare import completed, load_selected, row_stats, verdicts
from .metrics import hallucination_rate, percentile, segment_hit, video_hit

DRAFT_CHARS = 200  # visible text above this, with no tool call, is a draft answer instead of `finish`/READY
ENDING_TURN_TOKENS = 300  # proxy for runs that did not record turns: the loop-ending turn used more than this


def agent_turns(r: dict[str, Any], caps: dict[str, int]) -> list[dict[str, Any]]:
    """The agent turns of one run: recorded ones when present, else rebuilt from llm_calls (no text info)."""
    if r.get("agent_turns"):
        return r["agent_turns"]
    return [{"completion_tokens": c["completion_tokens"], "cap": caps[r["model_alias"]], "text_chars": None,
             "tool_calls": None, "finish": None, "finish_reason": None}
            for c in r["llm_calls"] if c["role"] == "agent"]


def turn_stats(records: list[dict[str, Any]], caps: dict[str, int]) -> dict[str, Any]:
    per_run = [agent_turns(r, caps) for r in records]
    turns = [t for run in per_run for t in run]
    exact = [t for t in turns if t["text_chars"] is not None]
    endings = [run[-1] for run in per_run if run]
    return {
        "agent_turns": len(turns),
        "cap_hits": sum(t["completion_tokens"] >= t["cap"] for t in turns),
        "runs_with_cap_hit": sum(any(t["completion_tokens"] >= t["cap"] for t in run) for run in per_run),
        "exact_turns": len(exact),
        "drafts": sum(1 for t in exact if not t["tool_calls"] and t["text_chars"] > DRAFT_CHARS),
        "cut_off_without_tool_call": sum(1 for t in exact if not t["tool_calls"] and t["finish_reason"] == "length"),
        "ending_turns": len(endings),
        "ending_turns_over_300_tokens": sum(t["completion_tokens"] > ENDING_TURN_TOKENS for t in endings),
    }


def hits(records: list[dict[str, Any]]) -> dict[str, Any]:
    vid = [video_hit(r["cited_ids"], r["gold_video_ids"]) for r in records if r.get("gold_video_ids")]
    seg = [segment_hit(r["cited_ids"], r["gold_segments"], 60) for r in records if r.get("gold_segments")]
    return {"video_hit": mean(vid) if vid else None, "segment_hit": mean(seg) if seg else None}


def trajectory(runs: dict[str, list[dict[str, Any]]], caps: dict[str, dict[str, int]]) -> dict[str, Any]:
    done = {label: completed(recs) for label, recs in runs.items()}
    labels = list(runs)
    out: dict[str, Any] = {"labels": labels, "summary": {}, "rows": []}
    for alias in sorted({k[1] for d in done.values() for k in d}):
        keys = sorted(set.intersection(*[{k for k in d if k[1] == alias} for d in done.values()]))
        block: dict[str, Any] = {"paired": len(keys), "completed": {l: sum(1 for k in done[l] if k[1] == alias) for l in labels}}
        for label in labels:
            recs = [done[label][k] for k in keys]
            st = [row_stats(r) for r in recs]
            lat = [s["latency"] for s in st]
            block[label] = {
                "steps_mean": mean(s["steps"] for s in st) if st else None,
                "latency_p50": median(lat) if lat else None, "latency_p95": percentile(lat, 95),
                "tokens_in_mean": mean(s["tokens_in"] for s in st) if st else None,
                "tokens_out_mean": mean(s["tokens_out"] for s in st) if st else None,
                "hallucination_rate": hallucination_rate(x for r in recs for x in verdicts(r)),
                "judged_sentences": sum(s["judged"] for s in st), "unjudged_sentences": sum(s["unjudged"] for s in st),
                **hits(recs), **turn_stats(recs, caps[label]),
            }
        out["summary"][alias] = block
        for k in keys:
            out["rows"].append({"id": k[0], "model_alias": alias,
                                **{label: {**row_stats(done[label][k]), **hits([done[label][k]])} for label in labels}})
    return out


def _f(x: float | None, nd: int = 1, pct: bool = False) -> str:
    return "n/a" if x is None else (f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}")


def to_markdown(t: dict[str, Any], names: dict[str, str]) -> str:
    labels = t["labels"]
    lines = [f"# Trajectory: {' -> '.join(labels)}", "",
             "Query-runs completed in every run, per agent model. Latency is wall-clock seconds per query-run; tokens are "
             "the agent's own calls (the judge is excluded); hallucination counts judged sentences only (the first run uses "
             "its re-judged verdicts when available). `cap hits` = agent turns that used their whole token cap; "
             "`drafts` = turns with no tool call and more than 200 characters of visible text (recorded from v2 on, "
             "`n/a` before); `ending turn > 300 tok` is a proxy available for every run (it also counts hidden reasoning).",
             "", "## Summary", "",
             "| agent model | paired | run | steps | latency p50 / p95 (s) | tokens in / out | hallucination (judged sent.) | "
             "unjudged | segment hit | video hit | agent turns | cap hits (runs) | drafts | ending turn > 300 tok |",
             "|" + "---|" * 14]
    for alias, b in t["summary"].items():
        for label in labels:
            x = b[label]
            drafts = f"{x['drafts']} of {x['exact_turns']}" if x["exact_turns"] else "n/a"
            lines.append(
                f"| {alias} | {b['paired']} | {label} | {_f(x['steps_mean'])} | {_f(x['latency_p50'])} / {_f(x['latency_p95'])} | "
                f"{_f(x['tokens_in_mean'], 0)} / {_f(x['tokens_out_mean'], 0)} | {_f(x['hallucination_rate'], pct=True)} "
                f"({x['judged_sentences']}) | {x['unjudged_sentences']} | {_f(x['segment_hit'], pct=True)} | "
                f"{_f(x['video_hit'], pct=True)} | {x['agent_turns']} | {x['cap_hits']} ({x['runs_with_cap_hit']}) | {drafts} | "
                f"{x['ending_turns_over_300_tokens']} of {x['ending_turns']} |")
    lines += ["", "(`paired` = query-runs completed in all runs; completed per run: "
              + "; ".join(f"{a}: " + ", ".join(f"{l} {n}" for l, n in b["completed"].items()) for a, b in t["summary"].items())
              + ")", "", "## Per query", "",
              "| query | model | steps | latency (s) | tokens in | hallucination (judged) | segment hit |", "|" + "---|" * 7]
    for r in t["rows"]:
        def cell(key: str, fmt) -> str:
            return " -> ".join(fmt(r[l][key]) for l in labels)
        lines.append(f"| {r['id']} | {r['model_alias']} | {cell('steps', str)} | {cell('latency', lambda v: f'{v:.0f}')} | "
                     f"{cell('tokens_in', str)} | {cell('hallucination', lambda v: _f(v, pct=True))} | "
                     f"{cell('segment_hit', lambda v: '-' if v is None else ('yes' if v else 'no'))} |")
    lines += ["", "Runs: " + "; ".join(f"`{l}` = {names[l]}" for l in labels)]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR[:arch]")
    p.add_argument("--caps", action="append", default=[], metavar="LABEL=LARGE:SMALL",
                   help="agent-turn token caps in force for that run, for records that do not carry their own")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cap_args = dict(c.split("=", 1) for c in a.caps)
    runs: dict[str, list[dict[str, Any]]] = {}
    caps: dict[str, dict[str, int]] = {}
    names: dict[str, str] = {}
    for i, spec in enumerate(a.run):
        label, path = spec.split("=", 1)
        runs[label], names[label] = load_selected(path, use_sidecar=(i == 0))
        large, small = (cap_args.get(label) or "1500:600").split(":")
        caps[label] = {"large": int(large), "small": int(small)}
    t = trajectory(runs, caps)
    out = Path(a.out)
    (out / "trajectory.json").write_text(json.dumps(t, indent=2), encoding="utf-8")
    md = to_markdown(t, names)
    (out / "trajectory.md").write_text(md, encoding="utf-8")
    print(md.split("## Per query")[0])
    print(f"wrote {out / 'trajectory.md'}")


if __name__ == "__main__":
    main()
