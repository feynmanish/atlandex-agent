"""python -m eval.agreement [--alt-judge eval/results/<run>/judge_<name>.jsonl]

Agreement (raw % and Cohen's kappa) between an LLM judge and the reference labels in
eval/labels/to_label.csv. The reference labels are written by Claude, not by a human.
With --alt-judge, the alternative judge is scored the same way and also compared with the
main judge over every judged sentence of the run.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def compute(reference: list[str], judge: list[str]) -> dict[str, Any]:
    n = len(reference)
    if n == 0:
        return {"status": "not yet labelled", "n": 0}
    agree = sum(h == j for h, j in zip(reference, judge))
    return {"status": "ok", "n": n, "agreement": agree / n, "kappa": cohens_kappa(reference, judge),
            "reference_unsupported": reference.count("unsupported"), "judge_unsupported": judge.count("unsupported")}


def load_reference(labels_dir: Path) -> dict[str, str]:
    with (labels_dir / "to_label.csv").open(encoding="utf-8", newline="") as f:
        lab = {r["row_id"]: _ALIASES.get(r["reference_label"].strip().lower()) for r in csv.DictReader(f)}
    return {k: v for k, v in lab.items() if v}


def load_verdict_rows(labels_dir: Path) -> dict[str, dict[str, str]]:
    with (labels_dir / "judge_verdicts.csv").open(encoding="utf-8", newline="") as f:
        return {r["row_id"]: r for r in csv.DictReader(f)}


def load_pairs(labels_dir: Path) -> tuple[list[str], list[str]]:
    """(reference labels, main-judge verdicts) for the rows that have a reference label."""
    ref, rows = load_reference(labels_dir), load_verdict_rows(labels_dir)
    ids = [i for i in ref if i in rows]
    return [ref[i] for i in ids], [rows[i]["judge_verdict"] for i in ids]


def load_alt_verdicts(path: Path) -> dict[tuple[str, str, int], str]:
    """(record id, model, sentence index) -> verdict from a judge_<name>.jsonl file."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k, s in enumerate((r.get("judge") or {}).get("sentences", [])):
            out[(r["id"], r["model"], k)] = s["verdict"]
    return out


def alt_vs_reference(labels_dir: Path, alt: dict[tuple[str, str, int], str]) -> tuple[list[str], list[str]]:
    ref, rows = load_reference(labels_dir), load_verdict_rows(labels_dir)
    a, b = [], []
    for i, lab in ref.items():
        r = rows.get(i)
        key = (r["record_id"], r["model"], int(r["sentence_index"])) if r else None
        if key in alt and alt[key] in ("supported", "unsupported"):
            a.append(lab)
            b.append(alt[key])
    return a, b


def judge_vs_judge(results: Path, alt: dict[tuple[str, str, int], str]) -> tuple[list[str], list[str]]:
    """Main vs alternative verdicts over every sentence both judged."""
    main_v, alt_v = [], []
    for line in (results / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k, s in enumerate((r.get("judge") or {}).get("sentences", [])):
            key = (r["id"], r["model"], k)
            if s["verdict"] in ("supported", "unsupported") and alt.get(key) in ("supported", "unsupported"):
                main_v.append(s["verdict"])
                alt_v.append(alt[key])
    return main_v, alt_v


def _fmt(r: dict[str, Any]) -> str:
    if r["status"] != "ok":
        return "not yet labelled"
    k = "undefined" if r["kappa"] is None else f"{r['kappa']:.2f}"
    return (f"n={r['n']}, agreement {r['agreement'] * 100:.1f}%, Cohen's kappa {k}, "
            f"unsupported: reference {r['reference_unsupported']} vs judge {r['judge_unsupported']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alt-judge", default=None, help="judge_<name>.jsonl from eval.rejudge")
    args = p.parse_args()
    try:
        ref, main_v = load_pairs(LABELS_DIR)
    except FileNotFoundError:
        print("not yet labelled (run python -m eval.sample_for_labels first)")
        return
    print("main judge vs reference labels:", _fmt(compute(ref, main_v)))
    if args.alt_judge:
        alt_path = Path(args.alt_judge)
        alt = load_alt_verdicts(alt_path)
        a, b = alt_vs_reference(LABELS_DIR, alt)
        print("alternative judge vs reference labels:", _fmt(compute(a, b)))
        m, x = judge_vs_judge(alt_path.parent, alt)
        r = compute(m, x)
        print("main vs alternative judge, all sentences:",
              _fmt({**r, "reference_unsupported": r.get("reference_unsupported", 0)}) if r["status"] == "ok" else "no overlap")


if __name__ == "__main__":
    main()
