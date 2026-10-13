"""The verification agent's pure parts: what it is shown, how its reply is read, and what the rewrite is told.

No LLM and no I/O here (the call itself is made by the `verify` node in multi_agent.py). This module must not
import from eval/: the dependency between the agent package and the eval harness runs only the other way, and
the verifier's input is written on its own rather than copied from the judge's (see build_verify_input).
"""
from __future__ import annotations

import re
from typing import Any

from .graph import parse_json_object
from .segments import CITE_RE, extract_citations

MAX_FEEDBACK_SENTENCES = 8  # sentences named in the rewrite request
SENTENCE_CHARS = 200  # each one is cut here in the rewrite request


def cited_indices(sentences: list[str]) -> list[int]:
    """Positions of the sentences that cite at least one segment. Only these are sent to the verifier: a sentence
    without a citation was already handled by the citation check, and a 'the segments do not cover X' remark
    claims nothing about a video."""
    return [i for i, s in enumerate(sentences) if extract_citations(s)]


def build_verify_input(sentences: list[str], segments: dict[str, dict[str, Any]]) -> str:
    """The sentences that cite something, numbered by their position in the whole answer, each followed by the
    text of the segments it cites. The numbers are positions in `sentences`, so the reply maps straight back."""
    blocks = []
    for i in cited_indices(sentences):
        claim = re.sub(r"\s+([.!?,;:])", r"\1", " ".join(CITE_RE.sub("", sentences[i]).split()))
        sources = []
        for sid in extract_citations(sentences[i]):
            seg = segments.get(sid)
            sources.append(f"  source [{sid}]: {seg['text']}" if seg else f"  source [{sid}]: (never retrieved)")
        blocks.append(f"S{i}: {claim}\n" + "\n".join(sources))
    return "\n\n".join(blocks)


def parse_verify_reply(text: str, valid: set[int]) -> dict[int, str] | None:
    """{sentence index: reason} for the sentences the verifier calls unsupported, from its raw reply.

    Read with the same tolerant JSON extraction as the router (code fences, prose around it, <think> tags).
    None when there is no usable reply (empty, not an object, no "unsupported" list): the caller treats that as a
    verifier failure and drops nothing. An empty dict means the verifier answered and found nothing wrong.
    Indices outside `valid` (never sent) are ignored, duplicates collapse, numeric strings are accepted.
    """
    obj = parse_json_object(text)
    if not obj or not isinstance(obj.get("unsupported"), list):
        return None
    reasons = obj.get("reasons") if isinstance(obj.get("reasons"), dict) else {}
    out: dict[int, str] = {}
    for raw in obj["unsupported"]:
        if isinstance(raw, bool):
            continue
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if i in valid and i not in out:
            out[i] = str(reasons.get(str(i), reasons.get(i, "")))[:200]
    return dict(sorted(out.items()))


def verify_feedback(flagged: list[tuple[str, str]]) -> str:
    """The rewrite request for sentences the verifier flagged: (sentence, reason) pairs."""
    shown = "\n".join(f"- {s[:SENTENCE_CHARS]}" + (f" ({r})" if r else "") for s, r in flagged[:MAX_FEEDBACK_SENTENCES])
    return ("Rewrite the answer. These sentences are not supported by the segments they cite:\n"
            f"{shown}\n"
            "Rewrite them so they say only what the cited segment says, or delete them. "
            "Do not add anything new.")
