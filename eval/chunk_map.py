"""Map the existing eval set's gold ``chunk_index`` labels to time spans.

The existing 22-query set labels gold as the 0-based ``chunk_index`` of a chapter-bounded
embedding chunk of one video. The MCP tools expose no chunk index, only each passage's
``start_sec``. The bridge is the production run that scored the same 22 queries: it
saved, per query, the chunk indices that FAISS L2 ranked top-3. ``search_video`` on the
same question returns passages in rank order, so passage *r* is taken to come from chunk
``faiss_l2_top[r]``, which yields (video, chunk_index, start_sec) observations.

Reliability check: chunks are time-ordered, so across all observations of one video the
chunk index must never decrease as ``start_sec`` increases. Any violation means the
rank alignment is wrong for that video and its mapping must not be used.

    python -m eval.chunk_map eval/existing/calibration_raw.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent / "existing"


def observations(calibration: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """(video, chunk_index, start_sec) from rank-aligned production/MCP results."""
    out = []
    for q in calibration:
        for rank, (_r, start, _d) in enumerate(q.get("mcp", [])):
            if start is not None and rank < len(q["l2"]):
                out.append((q["videoid"], int(q["l2"][rank]), int(start)))
    return sorted(set(out))


def violations(obs: list[tuple[str, int, int]]) -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    """Pairs where a higher chunk index starts earlier than a lower one, per video."""
    bad = []
    by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for v, c, s in obs:
        by_video[v].append((c, s))
    for v, pts in by_video.items():
        for c1, s1 in pts:
            for c2, s2 in pts:
                if c1 < c2 and s1 > s2:
                    bad.append((v, (c1, s1), (c2, s2)))
    return bad


def build_spans(obs: list[tuple[str, int, int]]) -> dict[str, dict[int, dict[str, Any]]]:
    """Time span per observed chunk.

    The span covers the chunk's observed passage starts. Where the adjacent chunk was also
    observed, the edge is moved to the midpoint between the two, since the chunk boundary
    lies somewhere in that gap.
    """
    raw: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for v, c, s in obs:
        raw[v][c].append(s)
    spans: dict[str, dict[int, dict[str, Any]]] = {}
    for v, chunks in raw.items():
        spans[v] = {}
        for c, starts in chunks.items():
            lo, hi = min(starts), max(starts)
            if c - 1 in chunks:
                lo = (max(chunks[c - 1]) + lo) // 2
            if c + 1 in chunks:
                hi = (hi + min(chunks[c + 1])) // 2
            spans[v][c] = {"start": lo, "end": hi, "observations": len(starts)}
    return spans


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("calibration", help="JSON written by the calibration run (see DECISIONS.md)")
    args = p.parse_args()
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    obs = observations(cal)
    bad = violations(obs)
    spans = build_spans(obs)
    result = {
        "observations": obs,
        "monotonic_violations": bad,
        "reliable": not bad,
        "spans": {v: {str(c): s for c, s in sorted(cs.items())} for v, cs in spans.items()},
    }
    (HERE / "chunk_time_map.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{len(obs)} observations, {len(bad)} ordering violations -> reliable={not bad}")
    for v, cs in spans.items():
        print(v, {c: (s["start"], s["end"]) for c, s in sorted(cs.items())})


if __name__ == "__main__":
    main()
