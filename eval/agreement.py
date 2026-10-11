"""python -m eval.agreement

Agreement between the LLM judge and your labels: raw agreement % and Cohen's kappa.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from atlandex_agent.config import ROOT

LABELS_DIR = ROOT / "eval" / "labels"
_ALIASES = {"supported": "supported", "s": "supported", "1": "supported", "yes": "supported",
            "unsupported": "unsupported", "u": "unsupported", "0": "unsupported", "no": "unsupported"}


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1.0:
        return 1.0 if po == 1.0 else None  # undefined when both raters only ever use one class
    return (po - pe) / (1 - pe)


def compute(human: list[str], judge: list[str]) -> dict[str, Any]:
    n = len(human)
    if n == 0:
        return {"status": "not yet labelled", "n": 0}
    agree = sum(h == j for h, j in zip(human, judge))
    return {"status": "ok", "n": n, "agreement": agree / n, "kappa": cohens_kappa(human, judge),
            "human_unsupported": human.count("unsupported"), "judge_unsupported": judge.count("unsupported")}


def load_pairs(labels_dir: Path) -> tuple[list[str], list[str]]:
    with (labels_dir / "to_label.csv").open(encoding="utf-8", newline="") as f:
        human = {r["row_id"]: _ALIASES.get(r["human_label"].strip().lower()) for r in csv.DictReader(f)}
    with (labels_dir / "judge_verdicts.csv").open(encoding="utf-8", newline="") as f:
        judge = {r["row_id"]: r["judge_verdict"] for r in csv.DictReader(f)}
    ids = [i for i in human if human[i] and i in judge]
    return [human[i] for i in ids], [judge[i] for i in ids]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        human, judge = load_pairs(LABELS_DIR)
    except FileNotFoundError:
        print("not yet labelled (run python -m eval.sample_for_labels first)")
        return
    r = compute(human, judge)
    if r["status"] != "ok":
        print("not yet labelled")
        return
    k = "undefined" if r["kappa"] is None else f"{r['kappa']:.2f}"
    print(f"labelled rows: {r['n']}\nagreement: {r['agreement'] * 100:.1f}%\nCohen's kappa: {k}\n"
          f"unsupported: human {r['human_unsupported']}, judge {r['judge_unsupported']}")


if __name__ == "__main__":
    main()
