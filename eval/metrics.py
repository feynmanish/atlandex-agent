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
import random
from collections import defaultdict
from typing import Any, Iterable

from atlandex_agent.citations import answer_sentences as split_answer
from atlandex_agent.graph import NO_COVERAGE


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
    """Share of *judged* sentences marked 'unsupported'; None when nothing was judged.

    'unjudged' sentences are neither clean nor unsupported here. They are reported separately
    (see `judge_coverage`); a rate with many unjudged sentences behind it is partial.
    """
    v = [x for x in verdicts if x in ("supported", "unsupported")]
    return (sum(x == "unsupported" for x in v) / len(v)) if v else None


UNJUDGED_WARN_SHARE = 0.10  # above this share of unjudged sentences the hallucination rate is flagged as partial


def judge_coverage(per_answer_verdicts: Iterable[Iterable[str]]) -> dict[str, Any]:
    """How much of what reached the judge actually got a verdict.

    `per_answer_verdicts` has one list of sentence verdicts per answer that was sent to the judge.
    A sentence the judge gave no usable verdict for is 'unjudged'. It is never counted as supported:
    `hallucination_rate` leaves it out of both numerator and denominator, and
    `hallucination_rate_upper_bound` counts it as unsupported, so the two bracket the true rate.
    """
    answers = [list(v) for v in per_answer_verdicts]
    sentences = [x for v in answers for x in v]
    unjudged = sum(x not in ("supported", "unsupported") for x in sentences)
    unsupported = sum(x == "unsupported" for x in sentences)
    fully = sum(bool(v) and all(x not in ("supported", "unsupported") for x in v) for v in answers)
    partly = sum(any(x not in ("supported", "unsupported") for x in v) for v in answers) - fully
    return {
        "judge_answers": len(answers),
        "answers_fully_unjudged": fully,
        "answers_partly_unjudged": partly,
        "unjudged_answer_share": (fully + partly) / len(answers) if answers else None,
        "judge_sentences": len(sentences),
        "unjudged_sentences": unjudged,
        "unjudged_sentence_share": unjudged / len(sentences) if sentences else None,
        "hallucination_rate_upper_bound": (unsupported + unjudged) / len(sentences) if sentences else None,
    }


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
    per_answer = [[s["verdict"] for s in r["judge"]["sentences"]] for r in ok if (r.get("judge") or {}).get("sentences")]
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
        **judge_coverage(per_answer),
        "steps_mean": _mean([float(r["steps"]) for r in ok]),
        "latency_p50": percentile(lat, 50),
        "latency_p95": percentile(lat, 95),
        "prompt_tokens_mean": _mean([float(x) for x in u("prompt_tokens")]),
        "completion_tokens_mean": _mean([float(x) for x in u("completion_tokens")]),
        "cost_usd_total": sum(r["usage"]["cost_usd"] for r in ok) + judge_cost,
        "citation_retries": sum(1 for r in ok if r.get("citation_check", {}).get("retried")),
        "citation_drops": sum(len(r.get("citation_check", {}).get("dropped", [])) for r in ok),
        # factual sentences that cited nothing: flagged in the first answer / still uncited and dropped from the final one
        "uncited_flagged": sum(len(r.get("citation_check", {}).get("uncited_first", [])) for r in ok),
        "uncited_dropped": sum(len(r.get("citation_check", {}).get("dropped_uncited", [])) for r in ok),
        "final_sentences": sum(len(split_answer(r.get("answer", ""))) for r in ok),
        # --- cost and the guards against a gain that is not one (see PLAN.md 6.5) ---------------------------
        # agent calls only (the judge is not included); a record from before these fields existed counts as 0 calls
        "llm_calls_mean": _mean([float(r["usage"].get("llm_calls", 0)) for r in ok]),
        "llm_calls_max": max((r["usage"].get("llm_calls", 0) for r in ok), default=None),
        # HTTP attempts incl. retries after 429/5xx; a cached reply made no request
        "http_attempts_total": sum(c.get("attempts", 1) for r in ok for c in r.get("llm_calls", []) if not c.get("cached")),
        "final_sentences_mean": _mean([float(len(split_answer(r.get("answer", "")))) for r in ok]),
        "no_coverage_answers": sum(1 for r in ok if r.get("answer", "").strip() == NO_COVERAGE),
        "verify_dropped": sum(len((r.get("verify") or {}).get("dropped_unverified", [])) for r in ok),
        "verify_failed": sum(1 for r in ok if (r.get("verify") or {}).get("failed")),
        "verify_skipped_budget": sum(1 for r in ok if (r.get("verify") or {}).get("skipped_budget")),
        # runs where the per-run call cap made the graph go to the answer before the agent had finished searching
        "budget_forced_runs": sum(1 for r in ok if any("call budget: forced answer" in n for n in r.get("notes", []))),
        "supervisor_fallbacks": sum(1 for r in ok for t in r.get("supervisor_turns", []) if t.get("fallback")),
    }


def record_arch(r: dict[str, Any]) -> str:
    """The architecture a record was produced by; records from before the multi-agent variant are single."""
    return r.get("arch") or "single"


def aggregate_by_arch(records: list[dict[str, Any]], tolerance_sec: int = 60) -> dict[str, dict[str, Any]]:
    """aggregate() separately for each architecture present, in the order single, multi."""
    archs = sorted({record_arch(r) for r in records}, key=lambda a: (a != "single", a))
    return {a: aggregate([r for r in records if record_arch(r) == a], tolerance_sec) for a in archs}


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


# --- paired comparison of two arms ------------------------------------------------------------------------------

def discordant(pairs: Iterable[tuple[bool, bool]]) -> tuple[int, int]:
    """(a_only, b_only): pairs where only arm A hit, and pairs where only arm B hit. Concordant pairs say nothing."""
    p = list(pairs)
    return sum(a and not b for a, b in p), sum(b and not a for a, b in p)


def sign_test_p(a_only: int, b_only: int) -> float:
    """Exact two-sided sign test on the discordant pairs: 2 * P(X <= min), X ~ Binomial(a_only + b_only, 1/2)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_diff(per_query_a: list[tuple[int, int]], per_query_b: list[tuple[int, int]], n: int = 2000,
                   seed: int = 0) -> dict[str, Any]:
    """Bootstrap 95% interval for the difference in hallucination rate, B minus A, resampling QUERIES.

    `per_query_x[i]` is (unsupported sentences, judged sentences) of query i in arm x; the two lists are paired
    by position. A rate is the pooled share over the resampled queries, as in the headline. Queries are the unit
    because the sentences of one answer are not independent. Resamples where an arm has no judged sentence are
    skipped. Deterministic for a given seed.
    """
    assert len(per_query_a) == len(per_query_b)
    m = len(per_query_a)

    def rate(rows: list[tuple[int, int]], idx: list[int]) -> float | None:
        judged = sum(rows[i][1] for i in idx)
        return sum(rows[i][0] for i in idx) / judged if judged else None

    point_a, point_b = rate(per_query_a, list(range(m))), rate(per_query_b, list(range(m)))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n if m else 0):
        idx = [rng.randrange(m) for _ in range(m)]
        ra, rb = rate(per_query_a, idx), rate(per_query_b, idx)
        if ra is not None and rb is not None:
            diffs.append(rb - ra)
    diffs.sort()
    return {"queries": m, "rate_a": point_a, "rate_b": point_b,
            "diff": (point_b - point_a) if point_a is not None and point_b is not None else None,
            "lo": percentile(diffs, 2.5), "hi": percentile(diffs, 97.5), "resamples": len(diffs)}
