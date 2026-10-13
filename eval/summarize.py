"""python -m eval.summarize eval/results/<run>

Recomputes summary.json and summary.md of a finished run from its records.jsonl (for example
after a metrics change), without calling any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import aggregate, aggregate_by_arch
from .run import summary_markdown


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results")
    a = p.parse_args()
    d = Path(a.results)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in (d / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in records:
        r.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        r.setdefault("steps", 0)
        r.setdefault("latency_sec", 0.0)
    tol = summary["gold_tolerance_sec"]
    by_arch = aggregate_by_arch(records, tol)
    agg = next(iter(by_arch.values())) if by_arch else aggregate(records, tol)  # the first (single) at the top level
    summary.update(agg)
    summary.pop("by_arch", None)
    if len(by_arch) > 1:
        summary["by_arch"] = by_arch
    summary["n_records"] = len(records)
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args = argparse.Namespace(model=summary["command"].get("model", "both"), subset=summary["command"].get("subset"),
                              gold_tolerance_sec=tol, arch=summary["command"].get("arch", "single"))
    (d / "summary.md").write_text(summary_markdown(agg, d, args, by_arch if len(by_arch) > 1 else None),
                                  encoding="utf-8")
    print(f"rewrote summary for {len(records)} records")


if __name__ == "__main__":
    main()
