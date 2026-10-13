"""Deterministic citation checks (no LLM involved)."""
from __future__ import annotations

import re
from typing import Iterable

from .segments import CITE_RE, ID_PATTERN, extract_citations, split_sentences

# A sentence that only says the segments do not cover something asserts nothing about a video, so it
# needs no citation (the judge prompt treats it the same way).
_NO_COVERAGE_STATEMENT = re.compile(
    r"(?i)\b(?:do(?:es)?\s+not|don't|doesn't|did\s+not|cannot|can't|can\s+not|unable\s+to|not\s+able\s+to)\b"
    r".{0,80}?\b(?:contain|mention|discuss|cover|include|provide|address|say|state|have|speak|explain|describe|"
    r"answer|find|extract|support|show|identify|specif)"
    r"|\bno\s+(?:indexed\s+)?(?:video|segment|information|mention|passage|statement|content|evidence)s?\b"
    r"|\bnone\s+of\s+the\b|\bnot\s+covered\b|\bonly\s+(?:contain|include)s?\s+(?:chapter|time)")
_MARKUP = re.compile(r"[*_`#>]+|^\s*[-•]\s+|^\s*\d+[.)]\s+")


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


def answer_sentences(answer: str) -> list[str]:
    """The sentences of an answer, line by line (the unit the judge and the citation check both use)."""
    out: list[str] = []
    for line in answer.splitlines():
        out += split_sentences(line)
    return out


def is_uncited_claim(sentence: str) -> bool:
    """True if the sentence states something about the video but names no segment.

    Not a claim: a sentence that cites; a fragment, label or list of IDs (fewer than 4 words once
    markup and IDs are removed); a heading ending in ':'; a statement that the segments do not cover
    something. Everything else is treated as a factual claim, including 'Thus, ...' conclusions.
    """
    if CITE_RE.search(sentence):
        return False
    text = re.sub(ID_PATTERN, " ", _MARKUP.sub(" ", sentence))
    text = re.sub(r"\[[^\]]*\]", " ", text).strip()
    if text.endswith(":") or len(re.findall(r"[A-Za-z0-9']+", text)) < 4:
        return False
    return not _NO_COVERAGE_STATEMENT.search(text)


def uncited_claims(answer: str) -> list[str]:
    """Factual sentences of the answer that cite no segment ID."""
    return [s for s in answer_sentences(answer) if is_uncited_claim(s)]


def drop_uncited(answer: str) -> tuple[str, list[str]]:
    """Remove every uncited factual sentence. Returns (cleaned answer, dropped sentences); lines are kept."""
    kept_lines: list[str] = []
    dropped: list[str] = []
    for line in answer.splitlines():
        kept = []
        for sentence in split_sentences(line):
            (dropped if is_uncited_claim(sentence) else kept).append(sentence)
        if kept:
            kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines).strip(), dropped


def drop_sentences(answer: str, indices: Iterable[int]) -> str:
    """Remove the sentences at `indices` (positions in `answer_sentences(answer)`); line breaks are kept.

    A line none of whose sentences are removed stays exactly as it was; the others are rejoined from their
    remaining sentences, like `drop_uncited`. An answer with nothing left returns "".
    """
    drop = set(indices)
    kept_lines: list[str] = []
    k = 0
    for line in answer.splitlines():
        sentences = split_sentences(line)
        flagged = [k + j in drop for j in range(len(sentences))]
        k += len(sentences)
        if not any(flagged):
            if line.strip():
                kept_lines.append(line)
            continue
        kept = [s for s, f in zip(sentences, flagged) if not f]
        if kept:
            kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines).strip()
