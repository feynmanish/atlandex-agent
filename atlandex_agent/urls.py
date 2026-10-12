"""YouTube links in a question.

All 22 questions of the existing eval set name their video with a youtube.com URL. The index is searched
by video ID, so the agent can read that video straight away instead of finding it first.
"""
from __future__ import annotations

import re

# watch?v=ID (with other query parameters before or after), youtu.be/ID, /shorts/ID, /embed/ID, /live/ID
_URL = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?(?:[^\s#)\]]*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})[^\s)\]]*")


def video_ids_in(text: str) -> list[str]:
    """The distinct YouTube video IDs linked in `text`, in order of appearance."""
    seen: list[str] = []
    for m in _URL.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def strip_urls(text: str) -> str:
    """`text` without the YouTube links (and the empty brackets they leave), whitespace tidied."""
    out = _URL.sub("", text)
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
    out = re.sub(r"\s+([:;,.?!])", r"\1", out)
    return re.sub(r"\s+", " ", out).strip()
