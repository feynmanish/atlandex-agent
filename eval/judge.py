"""LLM judge: is each answer sentence supported by the segments it cites?"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlandex_agent.graph import NO_COVERAGE, parse_json_object
from atlandex_agent.llm import LLM, usage_summary
from atlandex_agent.segments import CITE_RE, extract_citations, split_sentences

JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.md").read_text(encoding="utf-8")


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


async def judge_answer(llm: LLM, answer: str, segments: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """One judge call per answer (a second only if the first reply is unparseable).

    Returns None when there is nothing to judge (empty answer or the 'no coverage' reply).
    """
    if not answer.strip() or answer.strip() == NO_COVERAGE:
        return None
    sentences = answer_sentences(answer)
    if not sentences:
        return None
    first = len(llm.calls)
    verdicts: dict[int, dict[str, Any]] = {}
    for _ in range(2):
        r = await llm.chat(
            [{"role": "system", "content": JUDGE_PROMPT},
             {"role": "user", "content": build_judge_input(sentences, segments)}], role="judge")
        obj = parse_json_object(r.content) or {}
        for v in obj.get("verdicts", []) if isinstance(obj.get("verdicts"), list) else []:
            if isinstance(v, dict) and isinstance(v.get("i"), int) and v.get("verdict") in ("supported", "unsupported"):
                verdicts[v["i"]] = v
        if len(verdicts) == len(sentences):
            break
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
