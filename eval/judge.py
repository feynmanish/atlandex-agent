"""LLM judge: is each answer sentence supported by the segments it cites?"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from atlandex_agent.graph import NO_COVERAGE, parse_json_object
from atlandex_agent.llm import LLM, usage_summary
from atlandex_agent.segments import CITE_RE, extract_citations, split_sentences

JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.md").read_text(encoding="utf-8")

# The free judge is a reasoning model: with the global 1500-token cap it often spent the whole budget
# thinking (empty reply) or was cut off mid-JSON, and 27% of sentences ended up unjudged. Answers need
# about 1100 tokens, so give it room. A retry asks only for the sentences still missing, a few at a time,
# so it is a smaller, different request (an identical resend at temperature 0 fails the same way).
JUDGE_MAX_TOKENS = 4000
RETRY_BATCH = 3

# One complete verdict object, used to salvage a reply that was cut off before the closing brackets.
_VERDICT_RE = re.compile(
    r'"i"\s*:\s*(\d+)\s*,\s*"verdict"\s*:\s*"(supported|unsupported)"(?:\s*,\s*"reason"\s*:\s*"((?:[^"\\]|\\.)*)")?')


def parse_verdicts(text: str) -> dict[int, dict[str, Any]]:
    """{sentence index: {"verdict", "reason"}} from a judge reply.

    Reads the JSON object when it parses. If the reply was cut off, keeps every verdict object that is
    complete; the rest stay missing and are asked for again.
    """
    out: dict[int, dict[str, Any]] = {}
    obj = parse_json_object(text)
    items = obj.get("verdicts") if obj and isinstance(obj.get("verdicts"), list) else None
    if items is not None:
        for v in items:
            if isinstance(v, dict) and isinstance(v.get("i"), int) and v.get("verdict") in ("supported", "unsupported"):
                out[v["i"]] = v
        return out
    for m in _VERDICT_RE.finditer(re.sub(r"<think>.*?</think>", "", text, flags=re.S)):
        out.setdefault(int(m.group(1)), {"verdict": m.group(2), "reason": m.group(3) or ""})
    return out


def answer_sentences(answer: str) -> list[str]:
    out: list[str] = []
    for line in answer.splitlines():
        out += split_sentences(line)
    return out


def build_judge_input(sentences: list[str], segments: dict[str, dict[str, Any]]) -> str:
    blocks = []
    for i, s in enumerate(sentences):
        cited = extract_citations(s)
        evidence = "\n".join(
            f"  - [{c}] {segments[c]['text']}" if c in segments else f"  - [{c}] (segment was not retrieved)"
            for c in cited) or "  none"
        clean = CITE_RE.sub("", s).strip()
        blocks.append(f"{i}. {clean}\n   Cited segments:\n{evidence}")
    return "\n\n".join(blocks)


async def judge_answer(llm: LLM, answer: str, segments: dict[str, dict[str, Any]],
                       max_tokens: int = JUDGE_MAX_TOKENS) -> dict[str, Any] | None:
    """One judge call per answer. Sentences without a verdict are asked for again, `RETRY_BATCH` at a time.

    Returns None when there is nothing to judge (empty answer or the 'no coverage' reply).
    """
    if not answer.strip() or answer.strip() == NO_COVERAGE:
        return None
    sentences = answer_sentences(answer)
    if not sentences:
        return None
    first = len(llm.calls)
    verdicts: dict[int, dict[str, Any]] = {}

    async def ask(indices: list[int]) -> None:
        r = await llm.chat(
            [{"role": "system", "content": JUDGE_PROMPT},
             {"role": "user", "content": build_judge_input([sentences[i] for i in indices], segments)}],
            role="judge", max_tokens=max_tokens)
        for k, v in parse_verdicts(r.content).items():
            if 0 <= k < len(indices):
                verdicts[indices[k]] = v  # k numbers the sentences of this request, not of the answer

    await ask(list(range(len(sentences))))
    missing = [i for i in range(len(sentences)) if i not in verdicts]
    for start in range(0, len(missing), RETRY_BATCH):
        batch = [i for i in missing[start:start + RETRY_BATCH] if i not in verdicts]
        if batch:
            await ask(batch)
    out = []
    for i, s in enumerate(sentences):
        v = verdicts.get(i)
        out.append({
            "text": s,
            "cited_ids": extract_citations(s),
            "verdict": v["verdict"] if v else "unjudged",
            "reason": str(v.get("reason", "")) if v else "judge gave no verdict for this sentence",
        })
    return {"model": llm.model, "sentences": out, "usage": usage_summary(llm.calls[first:])}
