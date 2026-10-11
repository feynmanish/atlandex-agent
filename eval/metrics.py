"""Pure metric functions. Nothing here touches the network."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def parse_segment_id(sid: str) -> tuple[str, int | None]:
    video, _, rest = sid.partition("@")
    return video, int(rest) if rest.isdigit() else None


def segment_matches(cited: str, gold: str, tolerance_sec: int = 0) -> bool:
    """Does a cited segment ID count as the gold one?

    - exact match;
    - a bare gold video ID (no '@') matches any segment of that video;
    - otherwise the same video within `tolerance_sec` when both have a start time.

    Passage start times shift with the question wording, and some passages have no
    start time at all, so exact time matches alone would under-count correct answers.
    """
    if cited == gold:
        return True
    cv, cs = parse_segment_id(cited)
    gv, gs = parse_segment_id(gold)
    if cv != gv:
        return False
    if "@" not in gold:
        return True
    return bool(tolerance_sec) and cs is not None and gs is not None and abs(cs - gs) <= tolerance_sec


def task_success(cited_ids: Iterable[str], gold_ids: Iterable[str], tolerance_sec: int = 0) -> bool:
    """The answer cites at least one gold segment."""
    gold = list(gold_ids)
    return any(segment_matches(c, g, tolerance_sec) for c in cited_ids for g in gold)


def router_correct(route: str | None, expected: str | None) -> bool:
    return route is not None and route == expected


def hallucination_rate(verdicts: Iterable[str]) -> float | None:
    """Share of judged sentences marked 'unsupported'; None when nothing was judged."""
    v = [x for x in verdicts if x in ("supported", "unsupported")]
    return (sum(x == "unsupported" for x in v) / len(v)) if v else None


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize(records: list[dict[str, Any]], tolerance_sec: int = 0) -> dict[str, Any]:
    """Metrics for one group of records (one model, one source)."""
    ok = [r for r in records if not r.get("error")]
    scored = [r for r in records if r.get("gold_segment_ids")]
    verdicts = [s["verdict"] for r in ok for s in (r.get("judge") or {}).get("sentences", [])]
    lat = [r["latency_sec"] for r in ok]
    u = lambda key: [r["usage"][key] for r in ok]  # noqa: E731
    judge_cost = sum(((r.get("judge") or {}).get("usage") or {}).get("cost_usd", 0.0) for r in ok)
    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "task_success": _mean([float(task_success(r.get("cited_ids", []), r["gold_segment_ids"], tolerance_sec))
                               if not r.get("error") else 0.0 for r in scored]),
        "router_accuracy": _mean([float(router_correct(r.get("route"), r.get("expected_route"))) for r in ok
                                  if r.get("expected_route")]),
        "hallucination_rate": hallucination_rate(verdicts),
        "judged_sentences": len([v for v in verdicts if v in ("supported", "unsupported")]),
        "steps_mean": _mean([float(r["steps"]) for r in ok]),
        "latency_p50": percentile(lat, 50),
        "latency_p95": percentile(lat, 95),
        "prompt_tokens_mean": _mean([float(x) for x in u("prompt_tokens")]),
        "completion_tokens_mean": _mean([float(x) for x in u("completion_tokens")]),
        "cost_usd_total": sum(r["usage"]["cost_usd"] for r in ok) + judge_cost,
        "citation_retries": sum(1 for r in ok if r.get("citation_check", {}).get("retried")),
        "citation_drops": sum(len(r.get("citation_check", {}).get("dropped", [])) for r in ok),
    }


def aggregate(records: list[dict[str, Any]], tolerance_sec: int = 0) -> dict[str, Any]:
    """Per model and source. Records with needs_review=True never enter headline numbers."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    unverified: dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("needs_review"):
            unverified[r["model"]].append(r)
        else:
            groups[(r["model"], r["source"])].append(r)
    return {
        "headline": {f"{m}|{s}": summarize(rs, tolerance_sec) for (m, s), rs in sorted(groups.items())},
        "unverified_not_in_headline": {m: summarize(rs, tolerance_sec) for m, rs in sorted(unverified.items())},
    }
