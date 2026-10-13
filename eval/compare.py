"""python -m eval.compare --baseline eval/results/<old>[:arch] --fixed eval/results/<new>[:arch] [--name STEM]

Compares two finished runs (or two architectures of one run) of the same dataset, query by query: steps, LLM calls,
latency, tokens, hit rates, hallucination rate, unjudged and uncited sentences, and the guards against a gain
that is not one (final sentences per answer, no-coverage answers, verifier drops). Writes <STEM>.md and <STEM>.json
(default stem: comparison) into the fixed run's folder.

`DIR:single` / `DIR:multi` selects one architecture's records of a folder, so `R:single` vs `R:multi` pairs the two
arms of one eval.run --arch both session. Without a selector a folder must hold one architecture only.

Only query-runs that completed in BOTH sides are paired; the summary rows also say how many were paired.
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

from atlandex_agent.citations import answer_sentences
from atlandex_agent.graph import ARCHS, NO_COVERAGE

from .metrics import (bootstrap_diff, discordant, hallucination_rate, percentile, record_arch, segment_hit,
                      sign_test_p, video_hit)
from .rejudge import SIDECAR, merge_rejudged


def load(path: Path, use_sidecar: bool = False) -> list[dict[str, Any]]:
    records = [json.loads(l) for l in (path / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    side = path / SIDECAR
    if use_sidecar and side.exists():
        records = merge_rejudged(records, [json.loads(l) for l in side.read_text(encoding="utf-8").splitlines() if l.strip()])
    return records


def split_spec(spec: str) -> tuple[Path, str | None]:
    """'DIR' or 'DIR:single' / 'DIR:multi'. Only a known architecture after the last ':' counts, so a Windows
    drive letter ('C:\\runs\\x') is never mistaken for a selector."""
    head, sep, tail = spec.rpartition(":")
    return (Path(head), tail) if sep and tail in ARCHS else (Path(spec), None)


def load_selected(spec: str, use_sidecar: bool = False) -> tuple[list[dict[str, Any]], str]:
    """The records a DIR[:arch] selector names, and a display name for them."""
    path, arch = split_spec(spec)
    records = load(path, use_sidecar)
    present = {record_arch(r) for r in records}
    if arch:
        records = [r for r in records if record_arch(r) == arch]
        return records, f"{path.name}:{arch}"
    if len(present) > 1:
        raise SystemExit(f"{path} holds records of several architectures ({', '.join(sorted(present))}); "
                         "add :single or :multi to pick one")
    return records, path.name


def completed(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["id"], r["model_alias"]): r for r in records
            if not r.get("error") and not r.get("needs_review") and "usage" in r}


def verdicts(r: dict[str, Any]) -> list[str]:
    return [s["verdict"] for s in (r.get("judge") or {}).get("sentences", [])]


def row_stats(r: dict[str, Any]) -> dict[str, Any]:
    v = verdicts(r)
    cc = r.get("citation_check") or {}
    cited = r.get("cited_ids", [])
    return {
        "steps": r["steps"], "latency": r["latency_sec"],
        "tokens_in": r["usage"]["prompt_tokens"], "tokens_out": r["usage"]["completion_tokens"],
        "llm_calls": r["usage"]["llm_calls"] if "llm_calls" in r["usage"] else None,
        "hallucination": hallucination_rate(v),
        "judged": sum(x in ("supported", "unsupported") for x in v), "unjudged": sum(x == "unjudged" for x in v),
        "unsupported": sum(x == "unsupported" for x in v),
        "uncited_dropped": len(cc.get("dropped_uncited", [])), "uncited_flagged": len(cc.get("uncited_first", [])),
        "segment_hit": segment_hit(cited, r["gold_segments"], 60) if r.get("gold_segments") else None,
        "video_hit": video_hit(cited, r["gold_video_ids"]) if r.get("gold_video_ids") else None,
        "final_sentences": len(answer_sentences(r.get("answer", ""))),
        "no_coverage": r.get("answer", "").strip() == NO_COVERAGE,
        "verify_dropped": len((r.get("verify") or {}).get("dropped_unverified", [])),
        "supervisor_fallbacks": sum(1 for t in r.get("supervisor_turns", []) if t.get("fallback")),
        "route": r.get("route"),
    }


def _mean(xs) -> float | None:
    xs = list(xs)
    return mean(xs) if xs else None


def _rate(flags: list[bool | None]) -> float | None:
    flags = [f for f in flags if f is not None]
    return mean(flags) if flags else None


def compare(baseline: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> dict[str, Any]:
    b, f = completed(baseline), completed(fixed)
    keys = sorted(set(b) & set(f))
    rows = [{"id": k[0], "model_alias": k[1], "source": b[k].get("source"),
             "before": row_stats(b[k]), "after": row_stats(f[k])} for k in keys]
    summary: dict[str, Any] = {}
    sources = sorted({b[k].get("source") or "?" for k in keys})
    groups = [(a, lambda k, a=a: k[1] == a) for a in sorted({k[1] for k in keys})]
    groups += [(f"{a}|{s}", lambda k, a=a, s=s: k[1] == a and (b[k].get("source") or "?") == s)
               for a in sorted({k[1] for k in keys}) for s in sources]
    groups.append(("all", lambda k: True))
    for label_, pick in groups:
        sel = [k for k in keys if pick(k)]
        if not sel and label_ != "all":
            continue
        alias = label_.split("|")[0]
        out: dict[str, Any] = {"paired": len(sel),
                               "baseline_completed": sum(1 for k in b if alias in ("all", k[1])),
                               "fixed_completed": sum(1 for k in f if alias in ("all", k[1]))}
        for side, src in (("before", b), ("after", f)):
            recs = [src[k] for k in sel]
            st = [row_stats(r) for r in recs]
            lat = [s["latency"] for s in st]
            calls = [s["llm_calls"] for s in st if s["llm_calls"] is not None]
            out[side] = {
                "steps_mean": mean(s["steps"] for s in st) if st else None,
                "latency_median": median(lat) if lat else None, "latency_p95": percentile(lat, 95),
                "tokens_in_mean": mean(s["tokens_in"] for s in st) if st else None,
                "tokens_out_mean": mean(s["tokens_out"] for s in st) if st else None,
                "llm_calls_mean": _mean(calls), "llm_calls_max": max(calls) if calls else None,
                "hallucination_rate": hallucination_rate(x for r in recs for x in verdicts(r)),
                "judged_sentences": sum(s["judged"] for s in st), "unjudged_sentences": sum(s["unjudged"] for s in st),
                "uncited_flagged": sum(s["uncited_flagged"] for s in st),
                "uncited_dropped": sum(s["uncited_dropped"] for s in st),
                "segment_hit_rate": _rate([s["segment_hit"] for s in st]),
                "segment_hit_n": sum(s["segment_hit"] is not None for s in st),
                "video_hit_rate": _rate([s["video_hit"] for s in st]),
                "final_sentences_mean": _mean(s["final_sentences"] for s in st),
                "no_coverage_answers": sum(s["no_coverage"] for s in st),
                "verify_dropped": sum(s["verify_dropped"] for s in st),
                "supervisor_fallbacks": sum(s["supervisor_fallbacks"] for s in st),
            }
        by_key = {(r["id"], r["model_alias"]): r for r in rows}
        hit_pairs = [(by_key[k]["before"]["segment_hit"], by_key[k]["after"]["segment_hit"]) for k in sel
                     if by_key[k]["before"]["segment_hit"] is not None]
        before_only, after_only = discordant(hit_pairs)
        out["segment_hit_pairs"] = {"before_only": before_only, "after_only": after_only,
                                    "sign_test_p": sign_test_p(before_only, after_only)}
        out["hallucination_diff"] = bootstrap_diff(
            [(by_key[k]["before"]["unsupported"], by_key[k]["before"]["judged"]) for k in sel],
            [(by_key[k]["after"]["unsupported"], by_key[k]["after"]["judged"]) for k in sel])
        out["route_disagreements"] = [k[0] for k in sel if by_key[k]["before"]["route"] != by_key[k]["after"]["route"]]
        summary[label_] = out
    return {"rows": rows, "summary": summary}


def _f(x: float | None, nd: int = 1, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def to_markdown(c: dict[str, Any], baseline_name: str, fixed_name: str) -> str:
    lines = [f"# Before / after: `{baseline_name}` (baseline) vs `{fixed_name}` (fixed version)", "",
             "Paired query-runs only (completed in both). Latency is wall-clock seconds per query-run; tokens and LLM "
             "calls are the agent's own (the judge is not included). Hallucination = judged sentences marked unsupported; "
             "the baseline uses the re-judged verdicts when available, so both sides come from the same judge code.", "",
             "## Summary", "",
             "| agent model | paired | | steps | LLM calls mean / max | latency p50 / p95 (s) | tokens in / out | "
             "hallucination (judged sent.) | unjudged sent. | uncited flagged / dropped |", "|" + "---|" * 10]
    for alias, s in c["summary"].items():
        for label in ("before", "after"):
            x = s[label]
            top = x["llm_calls_max"] if x["llm_calls_max"] is not None else "n/a"
            lines.append(f"| {alias} | {s['paired']} of {s['baseline_completed']}/{s['fixed_completed']} | {label} | "
                         f"{_f(x['steps_mean'])} | {_f(x['llm_calls_mean'])} / {top} | "
                         f"{_f(x['latency_median'])} / {_f(x['latency_p95'])} | "
                         f"{_f(x['tokens_in_mean'], 0)} / {_f(x['tokens_out_mean'], 0)} | "
                         f"{_f(x['hallucination_rate'], pct=True)} ({x['judged_sentences']}) | {x['unjudged_sentences']} | "
                         f"{x['uncited_flagged']} / {x['uncited_dropped']} |")
    lines += ["", "(*paired* = query-runs completed in both runs, then completed in baseline / completed in fixed run.)", "",
              "## Quality and guards", "",
              "Segment hit (±60 s) and video hit over the paired queries that have gold. *Discordant* = queries where only "
              "one side hit; *p* = exact two-sided sign test on them. *Hallucination diff* = after minus before, with a "
              "95% bootstrap interval over queries (not sentences). The guards show whether a lower hallucination rate "
              "came from deleting content: final sentences per answer, no-coverage answers, sentences the verifier dropped.",
              "",
              "| agent model | | segment hit | video hit | discordant before-only / after-only (p) | "
              "hallucination diff [95% CI] | final sent. / answer | no-coverage | verify dropped | supervisor fallbacks | "
              "route disagreements |", "|" + "---|" * 11]
    for alias, s in c["summary"].items():
        p, d = s["segment_hit_pairs"], s["hallucination_diff"]
        pts = lambda v: "n/a" if v is None else f"{v * 100:+.0f}"  # noqa: E731
        diff = f"{pts(d['diff'])} pts [{pts(d['lo'])}, {pts(d['hi'])}]" if d["diff"] is not None else "n/a"
        routes = s["route_disagreements"]
        for label in ("before", "after"):
            x = s[label]
            first = label == "before"
            cells = [alias, label, f"{_f(x['segment_hit_rate'], pct=True)} (n={x['segment_hit_n']})",
                     _f(x["video_hit_rate"], pct=True),
                     f"{p['before_only']} / {p['after_only']} (p={p['sign_test_p']:.3f})" if first else "",
                     diff if first else "",
                     _f(x["final_sentences_mean"]), str(x["no_coverage_answers"]), str(x["verify_dropped"]),
                     str(x["supervisor_fallbacks"]),
                     (f"{len(routes)}" + (f" ({', '.join(routes)})" if routes else "")) if first else ""]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Per query", "",
              "| query | model | steps | latency (s) | tokens in | tokens out | hallucination (judged) | unjudged | "
              "uncited dropped | LLM calls | segment hit | final sent. |", "|" + "---|" * 12]
    yn = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
    for r in c["rows"]:
        a, z = r["before"], r["after"]
        lines.append(f"| {r['id']} | {r['model_alias']} | {a['steps']} -> {z['steps']} | {a['latency']:.0f} -> {z['latency']:.0f} | "
                     f"{a['tokens_in']} -> {z['tokens_in']} | {a['tokens_out']} -> {z['tokens_out']} | "
                     f"{_f(a['hallucination'], pct=True)} -> {_f(z['hallucination'], pct=True)} | "
                     f"{a['unjudged']} -> {z['unjudged']} | {z['uncited_dropped']} | "
                     f"{a['llm_calls'] if a['llm_calls'] is not None else '-'} -> "
                     f"{z['llm_calls'] if z['llm_calls'] is not None else '-'} | "
                     f"{yn(a['segment_hit'])} -> {yn(z['segment_hit'])} | {a['final_sentences']} -> {z['final_sentences']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, metavar="DIR[:arch]")
    p.add_argument("--fixed", required=True, metavar="DIR[:arch]")
    p.add_argument("--name", default="comparison", help="output file stem in the fixed run's folder")
    a = p.parse_args()
    base_records, base_name = load_selected(a.baseline, use_sidecar=True)
    fixed_records, fixed_name = load_selected(a.fixed)
    out_dir = split_spec(a.fixed)[0]
    c = compare(base_records, fixed_records)
    md = to_markdown(c, base_name, fixed_name)
    (out_dir / f"{a.name}.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    (out_dir / f"{a.name}.md").write_text(md, encoding="utf-8")
    print(md.split("## Per query")[0])
    print(f"wrote {out_dir / (a.name + '.md')}")


if __name__ == "__main__":
    main()
