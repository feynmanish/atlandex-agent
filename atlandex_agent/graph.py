"""The LangGraph agent: router -> (clarify | agent loop) -> answer -> citation_check."""
from __future__ import annotations

import json
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .citations import drop_unsupported, unsupported_ids
from .config import Settings
from .llm import LLM, usage_summary, records_to_dicts
from .mcp_tools import ToolBox
from .segments import Segment, extract_citations, parse_tool_output

ROUTES = ("simple", "multi_step", "unclear")
NO_COVERAGE = "No indexed video covered this question well enough to answer it."


class State(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    answer: str
    clarification: bool
    # agent loop
    evidence: list[dict[str, str]]  # [{"header", "body"}]
    pending: list[dict[str, Any]]  # tool calls awaiting execution
    tool_log: list[dict[str, Any]]
    segments: dict[str, dict[str, Any]]
    agent_done: bool
    nudged: bool
    # citation check
    answer_attempts: int
    feedback: str
    citation_check: dict[str, Any]
    notes: list[str]


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction: code fences, prose before/after, reasoning tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def bound_evidence(entries: list[dict[str, str]], budget: int) -> str:
    """Newest results in full while they fit; older ones shrink to their header line."""
    out: list[str] = []
    remaining = budget
    for e in reversed(entries):
        full = f"{e['header']}\n{e['body']}"
        if len(full) <= remaining:
            out.append(full)
            remaining -= len(full)
        else:
            out.append(f"{e['header']}\n(older result omitted to save context)")
            remaining -= len(e["header"]) + 40
    return "\n\n".join(reversed(out)) if out else "(none yet)"


def build_graph(llm: LLM, tools: ToolBox, settings: Settings):
    specs = tools.openai_specs()

    async def router(state: State) -> State:
        notes = list(state.get("notes", []))
        for attempt in range(2):
            r = await llm.chat(
                [{"role": "system", "content": prompts.ROUTER_SYSTEM},
                 {"role": "user", "content": state["question"]}], role="router")
            obj = parse_json_object(r.content)
            if obj and obj.get("route") in ROUTES:
                return {"route": obj["route"], "route_reason": str(obj.get("reason", "")), "notes": notes}
            notes.append(f"router output unparseable (attempt {attempt + 1})")
        return {"route": "multi_step", "route_reason": "router fallback", "notes": notes + ["router fell back to multi_step"]}

    async def clarify(state: State) -> State:
        r = await llm.chat(
            [{"role": "system", "content": prompts.CLARIFY_SYSTEM},
             {"role": "user", "content": state["question"]}], role="clarify")
        q = r.content.strip() or "Which topic, person or video do you mean?"
        return {"answer": q, "clarification": True}

    async def agent(state: State) -> State:
        steps = len(state.get("tool_log", []))
        hint = "Route: simple; one search is probably enough." if state["route"] == "simple" else \
            "Route: multi_step; plan the searches you need."
        user = (
            f"Question: {state['question']}\n{hint}\nTool calls used: {steps}/{settings.max_tool_steps}.\n\n"
            f"Evidence so far:\n{bound_evidence(state.get('evidence', []), settings.evidence_char_budget)}\n\n"
            "Call a tool if you still need evidence; otherwise reply with exactly: READY"
        )
        msgs = [{"role": "system", "content": prompts.AGENT_SYSTEM.format(max_steps=settings.max_tool_steps)},
                {"role": "user", "content": user}]
        r = await llm.chat(msgs, tools=specs, role="agent")
        if r.tool_calls:
            return {"pending": r.tool_calls, "agent_done": False}
        if steps == 0 and not state.get("nudged"):
            # A model that answers before any search would produce an ungrounded answer.
            return {"pending": [], "agent_done": False, "nudged": True,
                    "notes": state.get("notes", []) + ["agent skipped tools on first turn; nudged"],
                    "evidence": state.get("evidence", []) + [
                        {"header": "system", "body": "You must call find_videos or search_video before answering."}]}
        return {"pending": [], "agent_done": True}

    async def run_tools(state: State) -> State:
        log = list(state.get("tool_log", []))
        evidence = list(state.get("evidence", []))
        segments = dict(state.get("segments", {}))
        seen = {(t["name"], json.dumps(t["args"], sort_keys=True)) for t in log}
        for call in state.get("pending", []):
            if len(log) >= settings.max_tool_steps:
                break
            key = (call["name"], json.dumps(call["args"], sort_keys=True))
            header = f"Step {len(log) + 1}: {call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
            if key in seen:
                body, segs = "Already called with these exact arguments; no new results.", []
            else:
                raw = await tools.call(call["name"], call["args"])
                segs, body = parse_tool_output(call["name"], raw)
            seen.add(key)
            for s in segs:
                segments.setdefault(s.id, s.to_dict())
            log.append({"name": call["name"], "args": call["args"], "segments": [s.id for s in segs],
                        "error": body.startswith("Error") or "no searchable transcript" in body})
            evidence.append({"header": header, "body": body})
        return {"tool_log": log, "evidence": evidence, "segments": segments, "pending": []}

    def after_agent(state: State) -> str:
        if state.get("agent_done") or len(state.get("tool_log", [])) >= settings.max_tool_steps:
            return "answer"
        return "tools" if state.get("pending") else "agent"

    def after_tools(state: State) -> str:
        return "answer" if len(state.get("tool_log", [])) >= settings.max_tool_steps else "agent"

    async def answer(state: State) -> State:
        segments = state.get("segments", {})
        if not segments:
            return {"answer": NO_COVERAGE, "answer_attempts": state.get("answer_attempts", 0) + 1}
        listing = "\n".join(
            f"[{s['id']}] {s['title']} ({s['timestamp'] or 'time unknown'}, {s['kind']}): {s['text']}"
            for s in segments.values())
        user = f"Question: {state['question']}\n\nSegments:\n{listing}"
        if state.get("feedback"):
            user += f"\n\n{state['feedback']}"
        r = await llm.chat([{"role": "system", "content": prompts.ANSWER_SYSTEM},
                            {"role": "user", "content": user}], role="answer")
        return {"answer": r.content.strip(), "answer_attempts": state.get("answer_attempts", 0) + 1}

    async def citation_check(state: State) -> State:
        retrieved = state.get("segments", {}).keys()
        bad = unsupported_ids(state["answer"], retrieved)
        info = dict(state.get("citation_check") or {"first_failed": False, "retried": False, "dropped": []})
        if not bad:
            return {"citation_check": info, "feedback": ""}
        if state.get("answer_attempts", 1) < 2:
            info.update(first_failed=True, retried=True, unsupported_first=bad)
            fb = (f"Your previous answer cited IDs that were never retrieved: {', '.join(bad)}. "
                  "Rewrite the answer and cite only IDs from the list above.")
            return {"citation_check": info, "feedback": fb}
        cleaned, dropped = drop_unsupported(state["answer"], retrieved)
        info.update(dropped=dropped)
        return {"answer": cleaned or NO_COVERAGE, "citation_check": info, "feedback": ""}

    def after_check(state: State) -> str:
        return "answer" if state.get("feedback") else END

    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("clarify", clarify)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("answer", answer)
    g.add_node("citation_check", citation_check)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: "clarify" if s["route"] == "unclear" else "agent",
                            {"clarify": "clarify", "agent": "agent"})
    g.add_edge("clarify", END)
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "agent": "agent", "answer": "answer"})
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", "answer": "answer"})
    g.add_edge("answer", "citation_check")
    g.add_conditional_edges("citation_check", after_check, {"answer": "answer", END: END})
    return g.compile()


async def run_agent(question: str, llm: LLM, tools: ToolBox, settings: Settings) -> dict[str, Any]:
    """Run one question; returns a JSON-serialisable result with usage for this run only."""
    graph = build_graph(llm, tools, settings)
    first_call = len(llm.calls)
    t0 = time.perf_counter()
    final = await graph.ainvoke({"question": question}, {"recursion_limit": 60})
    latency = time.perf_counter() - t0
    records = llm.calls[first_call:]
    segments = final.get("segments", {})
    answer = final.get("answer", "")
    cited = extract_citations(answer)
    return {
        "question": question,
        "model": llm.model,
        "route": final.get("route"),
        "route_reason": final.get("route_reason", ""),
        "clarification": bool(final.get("clarification")),
        "answer": answer,
        "cited_ids": cited,
        "citations": {c: segments[c]["citation"] for c in cited if c in segments},
        "retrieved_ids": list(segments),
        "segments": segments,
        "tool_calls": final.get("tool_log", []),
        "steps": len(final.get("tool_log", [])),
        "citation_check": final.get("citation_check", {}),
        "notes": final.get("notes", []),
        "latency_sec": round(latency, 2),
        "usage": usage_summary(records),
        "llm_calls": records_to_dicts(records),
    }
