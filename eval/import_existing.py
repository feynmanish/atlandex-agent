"""python -m eval.import_existing

Builds the `existing` records of eval/dataset.jsonl from the 22-query set copied into
eval/existing/ (see DECISIONS.md for where it came from) and the chunk-to-time map made
by `python -m eval.chunk_map`. Other records (new candidates) are kept as they are.

Each query is tied to one video in the original set (it scored retrieval inside that
video), so the question given to the agent names the video; the agent still has to
search it and cite the right passage. `question_raw` keeps the original wording.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .run import DEFAULT_DATASET

HERE = Path(__file__).parent / "existing"


def build_records(queries: dict[str, Any], chunk_map: dict[str, Any]) -> list[dict[str, Any]]:
    reliable = bool(chunk_map.get("reliable"))
    out = []
    for i, q in enumerate(queries["queries"], 1):
        vid, chunks = q["videoid"], q["relevant_chunk_ids"]
        spans = []
        if reliable:
            for c in chunks:
                s = chunk_map["spans"].get(vid, {}).get(str(c))
                if s:
                    spans.append({"video": vid, "start": s["start"], "end": s["end"], "chunk": c})
        out.append({
            "id": f"ex{i:02d}",
            "question": f'About the video "{q["title"]}" (https://www.youtube.com/watch?v={vid}): {q["question"]}',
            "question_raw": q["question"],
            "gold_video_ids": [vid],
            "gold_chunks": chunks,
            "gold_segments": spans,
            "gold_basis": ("chunk_index mapped to a time span via eval/existing/chunk_time_map.json"
                           if spans else "video-level only: no reliable chunk-to-time mapping for this query"),
            "expected_route": "simple",
            "source": "existing",
            "needs_review": False,
            "label_source": "hand-labeled by the repo owner (binary relevance on chunk_index)",
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", default=str(HERE / "rag_query_set.json"))
    p.add_argument("--map", default=str(HERE / "chunk_time_map.json"))
    args = p.parse_args()
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    chunk_map = json.loads(Path(args.map).read_text(encoding="utf-8"))
    existing = build_records(queries, chunk_map)
    kept = []
    if DEFAULT_DATASET.exists():
        kept = [r for r in (json.loads(l) for l in DEFAULT_DATASET.read_text(encoding="utf-8").splitlines() if l.strip())
                if r["source"] != "existing"]
    DEFAULT_DATASET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in existing + kept) + "\n",
                               encoding="utf-8")
    timed = sum(1 for r in existing if r["gold_segments"])
    print(f"{len(existing)} existing queries ({timed} with segment-level gold, {len(existing) - timed} video-level only); "
          f"kept {len(kept)} other records.")


if __name__ == "__main__":
    main()
