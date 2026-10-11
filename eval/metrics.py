"""Pure metric functions. Nothing here touches the network.

Two hit measures are kept apart on purpose and never merged into one number:

- **video hit**: the answer cites at least one segment of a gold video.
- **segment hit (±N s)**: the answer cites a segment whose start time lies inside a
  gold span, widened by N seconds on each side (default 60).

A video hit says the agent found the right video; it does not say the cited passage
is the right one, so it is never reported as "task success".
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def parse_segment_id(sid: str) -> tuple[str, int | None]:
    """'abc@120' -> ('abc', 120); 'abc@na2' (unknown start) -> ('abc', None)."""
    video, _, rest = sid.partition("@")
    return video, int(rest) if rest.isdigit() else None


def video_hit(cited_ids: Iterable[str], gold_video_ids: Iterable[str]) -> bool:
    gold = set(gold_video_ids)
    return any(parse_segment_id(c)[0] in gold for c in cited_ids)


def segment_hit(cited_ids: Iterable[str], gold_spans: Iterable[dict[str, Any]], tolerance_sec: int = 60) -> bool:
    """A cited segment hits if its start time is within [span.start - tol, span.end + tol].

    A cited segment without a start time (`@na<rank>`) can never be a segment hit.
    """
    spans = list(gold_spans)
    for c in cited_ids:
        video, start = parse_segment_id(c)
        if start is None:
            continue
        for g in spans:
            if g["video"] == video and g["start"] - tolerance_sec <= start <= g["end"] + tolerance_sec:
                return True
    return False


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


def summarize(records: list[dict[str, Any]], tolerance_sec: int = 60) -> dict[str, Any]:
    """Metrics for one group of records (one model, one source).

    Every rate is over *completed* runs. A run that failed because the free model stayed
    unavailable after all retries (`error` set) is counted in `errors`, not as a miss: an
    outage says nothing about whether the agent would have found the right passage.
    """
    ok = [r for r in records if not r.get("error")]
    with_video = [r for r in ok if r.get("gold_video_ids")]
    with_span = [r for r in ok if r.get("gold_segments")]
    verdicts = [s["verdict"] for r in ok for s in (r.get("judge") or {}).get("sentences", [])]
    lat = [r["latency_sec"] for r in ok]
    u = lambda key: [r["usage"][key] for r in ok]  # noqa: E731
    judge_cost = sum(((r.get("judge") or {}).get("usage") or {}).get("cost_usd", 0.0) for r in ok)
    return {
        "n": len(records),
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "video_hit_rate": _mean([float(video_hit(r.get("cited_ids", []), r["gold_video_ids"])) for r in with_video]),
        "segment_hit_rate": _mean([float(segment_hit(r.get("cited_ids", []), r["gold_segments"], tolerance_sec))
                                   for r in with_span]),
        "segment_hit_n": len(with_span),
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


def aggregate(records: list[dict[str, Any]], tolerance_sec: int = 60) -> dict[str, Any]:
    """Per model and source (existing / new_candidate kept apart).

    Records with needs_review=True never enter the headline numbers; they are summarised
    separately per model.
    """
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
