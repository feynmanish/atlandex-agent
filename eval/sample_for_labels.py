"""python -m eval.sample_for_labels --n 30

Draws judged answer sentences from a results run and writes
  eval/labels/to_label.csv       what you label (no judge verdict visible)
  eval/labels/judge_verdicts.csv the judge's verdicts, keyed by row_id (look only after labelling)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from atlandex_agent.config import ROOT

from .run import RESULTS_DIR

LABELS_DIR = ROOT / "eval" / "labels"


def latest_results() -> Path:
    runs = sorted(p for p in RESULTS_DIR.glob("*") if (p / "records.jsonl").exists())
    if not runs:
        raise SystemExit("No results found in eval/results/. Run python -m eval.run first.")
    return runs[-1]


def collect(results: Path, arch: str | None = None) -> list[dict]:
    rows = []
    for line in (results / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("needs_review") or not r.get("judge") or (arch and (r.get("arch") or "single") != arch):
            continue
        for k, s in enumerate(r["judge"]["sentences"]):
            if s["verdict"] not in ("supported", "unsupported"):
                continue
            seg_text = "\n".join(
                f"[{c}] {r['segments'][c]['text']}" for c in s["cited_ids"] if c in r["segments"]) or "(no cited segment)"
            rows.append({"record": r["id"], "model": r["model"], "arch": r.get("arch") or "single",
                         "question": r["question"], "k": k,
                         "sentence": s["text"], "segment_text": seg_text, "verdict": s["verdict"], "reason": s["reason"]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--results", default=None, help="results directory (default: the latest)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["single", "multi"], default=None,
                   help="only draw sentences from the answers of this architecture (default: all of them)")
    p.add_argument("--stratify", action="store_true",
                   help="draw half the rows from sentences the judge marked unsupported, half from supported")
    args = p.parse_args()

    results = Path(args.results) if args.results else latest_results()
    pool = collect(results, args.arch)
    rng = random.Random(args.seed)
    if args.stratify:
        # Half from each judge class (the verdict is used only to balance the sample; it is
        # not written to to_label.csv). Without this, a low hallucination rate would leave
        # almost no 'unsupported' rows and kappa would be meaningless.
        half = args.n // 2
        uns = rng.sample([s for s in pool if s["verdict"] == "unsupported"],
                         min(half, sum(s["verdict"] == "unsupported" for s in pool)))
        sup = [s for s in pool if s["verdict"] == "supported"]
        sample = uns + rng.sample(sup, min(args.n - len(uns), len(sup)))
        rng.shuffle(sample)
    else:
        sample = rng.sample(pool, min(args.n, len(pool)))
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with (LABELS_DIR / "to_label.csv").open("w", newline="", encoding="utf-8") as f, \
            (LABELS_DIR / "judge_verdicts.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(f)
        w.writerow(["row_id", "question", "answer_sentence", "cited_segment_text", "reference_label"])
        v = csv.writer(g)
        v.writerow(["row_id", "judge_verdict", "judge_reason", "results_run", "record_id", "model", "sentence_index", "arch"])
        for i, s in enumerate(sample):
            w.writerow([i, s["question"], s["sentence"], s["segment_text"], ""])
            v.writerow([i, s["verdict"], s["reason"], results.name, s["record"], s["model"], s["k"], s["arch"]])
    print(f"Wrote {len(sample)} rows (pool of {len(pool)}) from {results.name}.")
    print("Fill reference_label with 'supported' or 'unsupported' in eval/labels/to_label.csv "
          "without opening judge_verdicts.csv, then run python -m eval.agreement.")


if __name__ == "__main__":
    main()
