"""Deterministic citation checks (no LLM involved)."""
from __future__ import annotations

from typing import Iterable

from .segments import CITE_RE, extract_citations, split_sentences


def unsupported_ids(answer: str, retrieved_ids: Iterable[str]) -> list[str]:
    """Cited segment IDs that were never retrieved in this run."""
    have = set(retrieved_ids)
    return [c for c in extract_citations(answer) if c not in have]


def drop_unsupported(answer: str, retrieved_ids: Iterable[str]) -> tuple[str, list[str]]:
    """Remove every sentence that cites a segment which was not retrieved.

    Returns (cleaned answer, dropped sentences). Line breaks are preserved.
    """
    have = set(retrieved_ids)
    kept_lines: list[str] = []
    dropped: list[str] = []
    for line in answer.splitlines():
        kept: list[str] = []
        for sentence in split_sentences(line):
            bad = [c for c in CITE_RE.findall(sentence) if c not in have]
            if bad:
                dropped.append(sentence)
            else:
                kept.append(sentence)
        if kept:
            kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines).strip(), dropped
