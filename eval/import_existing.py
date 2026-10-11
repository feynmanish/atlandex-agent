"""python -m eval.import_existing PATH

Imports the existing retrieval eval set (queries + gold segment IDs) into
eval/dataset.jsonl as source=existing records, keeping any new_candidate records.
Accepts .json (list) or .jsonl, with `question`/`query` and a list under
`gold_segment_ids`/`gold_ids`/`gold`/`segment_ids`. Edit FIELD_ALIASES if yours differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import DEFAULT_DATASET

FIELD_ALIASES = {
    "question": ("question", "query", "q"),
    "gold": ("gold_segment_ids", "gold_ids", "gold", "segment_ids", "relevant_ids"),
    "route": ("expected_route", "route"),
}


def _pick(row: dict, names: tuple[str, ...]):
    for n in names:
        if n in row:
            return row[n]
    return None


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("queries", data.get("questions", []))


def convert(rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows, 1):
        gold = _pick(row, FIELD_ALIASES["gold"]) or []
        out.append({
            "id": f"ex{i:02d}",
            "question": _pick(row, FIELD_ALIASES["question"]),
            "gold_segment_ids": [gold] if isinstance(gold, str) else list(gold),
            "expected_route": _pick(row, FIELD_ALIASES["route"]) or "simple",
            "source": "existing",
            "needs_review": False,
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    args = p.parse_args()
    existing = convert(read_rows(Path(args.path)))
    kept = []
    if DEFAULT_DATASET.exists():
        kept = [json.loads(l) for l in DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["source"] != "existing"]
    DEFAULT_DATASET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in existing + kept) + "\n", encoding="utf-8")
    print(f"Imported {len(existing)} existing queries; kept {len(kept)} other records.")


if __name__ == "__main__":
    main()
