"""Segment IDs, tool-output normalisation and citation extraction.

The MCP tools return passages without an ID of their own, so the agent derives
one: ``<video_id>@<start_sec>`` (``<video_id>@na<rank>`` when the start time is
unknown). The model cites ``[<segment id>]``; the CLI maps IDs back to the
tools' ready-made ``citation`` links.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any

# 11-char YouTube id, '@', digits or na<rank>
ID_PATTERN = r"[A-Za-z0-9_-]{11}@(?:\d+|na\d+)"
CITE_RE = re.compile(rf"\[({ID_PATTERN})\]")

PASSAGE_CHARS = 900  # cap per passage shown to the model

# A sentence ends at . ! ? plus any [id] citations written after it, before a new sentence starts.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s*\[[^\]]*\])*(?=\s+[A-Z0-9\"'(\[]|\s*$)")


@dataclass
class Segment:
    id: str
    video_id: str
    title: str
    timestamp: str | None
    text: str
    citation: str
    kind: str  # passage | chapter | quote

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def segment_id(video_id: str, start_sec: Any, rank: int = 0) -> str:
    if start_sec is None:
        return f"{video_id}@na{rank}"
    return f"{video_id}@{int(start_sec)}"


# Models sometimes cite as 【id】, (id) or [id1, id2] instead of [id].
_LOOSE_CITE = re.compile(rf"[\[【［(]\s*({ID_PATTERN}(?:\s*[,;]\s*{ID_PATTERN})*)\s*[\]】］)]")


def normalize_citations(text: str) -> str:
    """Rewrite loose citation styles to the canonical `[id] [id]` form."""
    return _LOOSE_CITE.sub(lambda m: " ".join(f"[{i}]" for i in re.split(r"\s*[,;]\s*", m.group(1))), text)


def extract_citations(text: str) -> list[str]:
    """Segment IDs cited in `text`, in order, de-duplicated."""
    seen: list[str] = []
    for m in CITE_RE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps a trailing [id] citation with its sentence."""
    text = re.sub(r"\s+", " ", text.strip())
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[start : m.end()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def parse_tool_output(tool: str, raw: str) -> tuple[list[Segment], str]:
    """Return (segments, text for the model). Non-JSON output (errors) passes through."""
    try:
        data = json.loads(raw)
    except ValueError:
        return [], raw.strip()[:600]
    if not isinstance(data, dict):
        return [], raw.strip()[:600]

    segs: list[Segment] = []
    lines: list[str] = []
    if tool == "find_videos":
        lines.append(f"find_videos term={data.get('term')!r}: {data.get('total_matches', 0)} matches")
        for i, v in enumerate(data.get("videos", [])):
            sid = segment_id(v["video_id"], v.get("start_sec"), i)
            segs.append(Segment(sid, v["video_id"], v.get("title", ""), v.get("timestamp"),
                                f"Chapter '{v.get('chapter_title', '')}' covers this term.",
                                v.get("citation", ""), "chapter"))
            lines.append(
                f"- video_id={v['video_id']} \"{v.get('title', '')}\" ({v.get('channel', '')}) "
                f"searchable={v.get('searchable')} chapter '{v.get('chapter_title', '')}' at "
                f"{v.get('timestamp')} [{sid}]"
            )
        if data.get("hint"):
            lines.append(f"hint: {data['hint']}")
    elif tool == "search_video":
        lines.append(f"search_video video_id={data.get('video_id')} \"{data.get('title', '')}\"")
        for p in data.get("passages", []):
            vid = data["video_id"]
            sid = segment_id(vid, p.get("start_sec"), p.get("rank", 0))
            text = p.get("text", "")[:PASSAGE_CHARS]
            segs.append(Segment(sid, vid, data.get("title", ""), p.get("timestamp"), text,
                                p.get("citation", ""), "passage"))
            lines.append(f"[{sid}] ({p.get('timestamp') or 'time unknown'}) {text}")
        if not data.get("passages"):
            lines.append("no passages returned")
        if data.get("note"):
            lines.append(f"note: {data['note']}")
    elif tool == "locate_quote":
        if data.get("found"):
            sid = segment_id(data["video_id"], data.get("start_sec"))
            segs.append(Segment(sid, data["video_id"], data.get("title", ""), data.get("timestamp"),
                                "(time pointer only; the tool returns no transcript text)",
                                data.get("citation", ""), "quote"))
            lines.append(f"locate_quote: phrase found around {data.get('timestamp')} [{sid}] "
                         "(approximate; no text returned)")
        else:
            lines.append("locate_quote: phrase not found")
        if data.get("hint"):
            lines.append(f"hint: {data['hint']}")
    else:
        return [], raw.strip()[:600]
    return segs, "\n".join(lines)
