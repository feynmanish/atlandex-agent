"""Offline fakes: a scripted LLM and a scripted toolbox. No network anywhere in tests."""
from __future__ import annotations

import json
from typing import Any

from atlandex_agent.llm import CallRecord, LLMResponse


class FakeLLM:
    """Replies come from per-role queues; the last reply of a queue repeats."""

    def __init__(self, replies: dict[str, list[Any]], model: str = "fake/model:free"):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.model = model
        self.calls: list[CallRecord] = []
        self.prompts: list[tuple[str, list[dict]]] = []
        self.max_tokens: list[int | None] = []  # the per-call cap each chat() was given
        self.tools_offered: list[list[str]] = []  # tool names offered on each chat()

    async def chat(self, messages, *, tools=None, role: str = "", max_tokens=None) -> LLMResponse:
        self.prompts.append((role, messages))
        self.max_tokens.append(max_tokens)
        self.tools_offered.append([t["function"]["name"] for t in tools or []])
        queue = self.replies[role]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        rec = CallRecord(role=role, model=self.model, prompt_tokens=100, completion_tokens=10)
        self.calls.append(rec)
        if isinstance(item, dict) and "text" in item:  # a plain reply with a finish reason, e.g. cut off at the cap
            return LLMResponse(item["text"], [], rec, item.get("finish_reason"))
        if isinstance(item, dict):  # tool calls
            return LLMResponse("", [{"id": f"c{len(self.calls)}", **tc} for tc in item["tools"]], rec)
        return LLMResponse(item, [], rec)

    def roles(self) -> list[str]:
        return [r for r, _ in self.prompts]


def tool_turn(*calls: tuple[str, dict]) -> dict:
    return {"tools": [{"name": n, "args": a} for n, a in calls]}


class FakeToolBox:
    def __init__(self, outputs: dict[str, str | list[str]]):
        self.outputs = {k: ([v] if isinstance(v, str) else list(v)) for k, v in outputs.items()}
        self.calls: list[tuple[str, dict]] = []

    def openai_specs(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in self.outputs]

    async def call(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        q = self.outputs[name]
        return q.pop(0) if len(q) > 1 else q[0]


VID_A, VID_B = "AAAAAAAAAAA", "BBBBBBBBBBB"


def find_json(*videos: tuple[str, int, bool]) -> str:
    return json.dumps({"term": "t", "total_matches": len(videos), "videos": [
        {"video_id": v, "title": f"Video {v[0]}", "channel": "Ch", "chapter_title": "Part 1", "start_sec": s,
         "timestamp": f"{s // 60}:{s % 60:02d}", "searchable": ok,
         "citation": f"[Video {v[0]}, chapter](https://youtu.be/{v}?t={s})"} for v, s, ok in videos]})


def search_json(video: str, *passages: tuple[int, str]) -> str:
    return json.dumps({"video_id": video, "title": f"Video {video[0]}", "passages": [
        {"rank": i + 1, "text": text, "start_sec": s, "timestamp": f"{s // 60}:{s % 60:02d}",
         "citation": f"[Video {video[0]}, {s // 60}:{s % 60:02d}](https://youtu.be/{video}?t={s})"}
        for i, (s, text) in enumerate(passages)]})
