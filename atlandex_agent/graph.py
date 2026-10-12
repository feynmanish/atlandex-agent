"""The LangGraph agent: router -> (clarify | agent loop) -> answer -> citation_check."""
from __future__ import annotations

import json
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .citations import drop_uncited, drop_unsupported, uncited_claims, unsupported_ids
from .config import Settings
from .llm import LLM, usage_summary, records_to_dicts
from .mcp_tools import ToolBox
from .segments import extract_citations, normalize_citations, parse_tool_output
from .tracing import NullTracer, TraceInfo, tracer_for
from .urls import strip_urls, video_ids_in

ROUTES = ("simple", "multi_step", "unclear")
MAX_SEEDED_VIDEOS = 3  # videos linked in the question that are read before the agent loop starts

# A no-argument tool that is not an MCP tool: the model calls it to say the evidence is enough. It is offered
# next to the real tools, never executed, and never counted as a step.
FINISH_TOOL = "finish"
FINISH_SPEC = {"type": "function", "function": {
    "name": FINISH_TOOL,
    "description": "Call this when the evidence collected so far is enough to answer, or nothing more can be found. "
                   "Takes no arguments.",
    "parameters": {"type": "object", "properties": {}}}}
NO_COVERAGE = "No indexed video covered this question well enough to answer it."


class EmptyAnswerError(RuntimeError):
    """The model returned an empty answer even with a larger budget. The run is failed, not passed."""


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
    agent_turns: list[dict[str, Any]]  # what each agent turn did (for diagnosing caps and drafts)
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
            "Call a tool if you still need evidence; otherwise call the finish tool (or reply with exactly: READY)"
        )
        msgs = [{"role": "system", "content": prompts.AGENT_SYSTEM.format(max_steps=settings.max_tool_steps)},
                {"role": "user", "content": user}]
        # The capped reply can only be a tool call, `finish`, or a short READY; it cannot turn into a draft answer.
        # The cap depends on the model: hidden reasoning tokens count against it (see Settings.agent_cap_for).
        cap = settings.agent_cap_for(llm.model)
        r = await llm.chat(msgs, tools=specs + [FINISH_SPEC], role="agent", max_tokens=cap)
        calls = [c for c in r.tool_calls if c["name"] != FINISH_TOOL]
        turns = state.get("agent_turns", []) + [{
            "tool_calls": [c["name"] for c in r.tool_calls], "finish": len(calls) < len(r.tool_calls),
            "text_chars": len(r.content or ""), "completion_tokens": r.record.completion_tokens,
            "finish_reason": r.finish_reason, "cap": cap}]
        if calls:  # `finish` next to real calls is ignored: the real calls run, the model can finish next turn
            return {"pending": calls, "agent_done": False, "agent_turns": turns}
        if steps == 0 and not state.get("nudged"):
            # A model that answers before any search would produce an ungrounded answer.
            return {"pending": [], "agent_done": False, "nudged": True, "agent_turns": turns,
                    "notes": state.get("notes", []) + ["agent skipped tools on first turn; nudged"],
                    "evidence": state.get("evidence", []) + [
                        {"header": "system", "body": "You must call find_videos or search_video before answering."}]}
        notes = state.get("notes", [])
        if r.finish_reason == "length" and not r.tool_calls:
            notes = notes + ["agent turn hit the token cap without a tool call; treated as done"]
        return {"pending": [], "agent_done": True, "notes": notes, "agent_turns": turns}

    async def seed(state: State) -> State:
        """Questions that link their video (all 22 existing eval questions do): read that video first."""
        ids = video_ids_in(state["question"])[:MAX_SEEDED_VIDEOS]
        if not ids:
            return {"pending": []}
        question = strip_urls(state["question"])
        return {"pending": [{"id": f"seed{i}", "name": "search_video", "args": {"video": v, "question": question}}
                            for i, v in enumerate(ids)],
                "notes": state.get("notes", []) + [f"read the video(s) named in the question first: {', '.join(ids)}"]}

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
        msgs = [{"role": "system", "content": prompts.ANSWER_SYSTEM}, {"role": "user", "content": user}]
        r = await llm.chat(msgs, role="answer")
        notes = state.get("notes", [])
        if not r.content.strip():
            # A reasoning model can spend the whole answer budget thinking and return nothing. Ask once more
            # with a bigger budget; if that is empty too, citation_check fails the run.
            r = await llm.chat(msgs, role="answer", max_tokens=settings.answer_retry_max_tokens)
            notes = notes + ["answer was empty; asked again with a larger budget"
                             + ("" if r.content.strip() else " (still empty)")]
        return {"answer": normalize_citations(r.content.strip()), "notes": notes,
                "answer_attempts": state.get("answer_attempts", 0) + 1}

    async def citation_check(state: State) -> State:
        retrieved = state.get("segments", {}).keys()
        answer = state["answer"]
        if not answer.strip():  # nothing to check is not a pass: an empty answer is a failed run
            raise EmptyAnswerError("the model returned an empty answer, also with a larger token budget")
        bad = unsupported_ids(answer, retrieved)
        # Factual sentences that cite nothing (the judge flagged these most often). Only meaningful when
        # something was retrieved; the fixed "no coverage" reply has no segments to cite.
        claims = uncited_claims(answer) if retrieved and answer != NO_COVERAGE else []
        info = dict(state.get("citation_check") or {"first_failed": False, "retried": False, "dropped": []})
        if not bad and not claims:
            if retrieved and answer != NO_COVERAGE and not extract_citations(answer):
                info["no_citations"] = True  # nothing to check, but the answer cites no segment at all
            return {"citation_check": info, "feedback": ""}
        if state.get("answer_attempts", 1) < 2:
            info.update(first_failed=True, retried=True, unsupported_first=bad, uncited_first=claims)
            parts = ["Rewrite the answer."]
            if bad:
                parts.append(f"Your previous answer cited IDs that were never retrieved: {', '.join(bad)}. "
                             "Cite only IDs from the list above.")
            if claims:
                shown = "\n".join(f"- {c[:200]}" for c in claims[:8])
                parts.append("These sentences of your previous answer state facts but end without a segment ID:\n"
                             f"{shown}\nPut the ID of the supporting segment in square brackets, e.g. "
                             "[AbCdEfGhIjK@657], at the end of each such sentence, or delete the sentence. "
                             "Do not add introductions, summaries or conclusions of your own.")
            return {"citation_check": info, "feedback": " ".join(parts)}
        # Second answer still has problems: drop the offending sentences and keep the rest.
        cleaned, dropped = drop_unsupported(answer, retrieved)
        cleaned, dropped_uncited = drop_uncited(cleaned)
        info.update(dropped=dropped, dropped_uncited=dropped_uncited)
        if retrieved and cleaned and not extract_citations(cleaned):
            info["no_citations"] = True  # nothing left that cites a segment (only 'not covered' statements)
        return {"answer": cleaned or NO_COVERAGE, "citation_check": info, "feedback": ""}

    def after_check(state: State) -> str:
        return "answer" if state.get("feedback") else END

    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("clarify", clarify)
    g.add_node("seed", seed)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("answer", answer)
    g.add_node("citation_check", citation_check)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: "clarify" if s["route"] == "unclear" else "seed",
                            {"clarify": "clarify", "seed": "seed"})
    g.add_edge("clarify", END)
    g.add_conditional_edges("seed", lambda s: "tools" if s.get("pending") else "agent",
                            {"tools": "tools", "agent": "agent"})
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "agent": "agent", "answer": "answer"})
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", "answer": "answer"})
    g.add_edge("answer", "citation_check")
    g.add_conditional_edges("citation_check", after_check, {"answer": "answer", END: END})
    return g.compile()


async def run_agent(question: str, llm: LLM, tools: ToolBox, settings: Settings,
                    trace: TraceInfo | None = None, tracer: NullTracer | None = None) -> dict[str, Any]:
    """Run one question; returns a JSON-serialisable result with usage for this run only.

    With Langfuse configured (see tracing.py) the run is also recorded as one trace; its id is
    returned as `trace_id`. Without keys `trace_id` is None and nothing else changes.
    """
    tracer = tracer or tracer_for(settings)
    with tracer.run(question, llm.model, trace) as run:
        llm, tools = run.llm(llm), run.tools(tools)
        graph = build_graph(llm, tools, settings)
        first_call = len(llm.calls)
        t0 = time.perf_counter()
        final = await graph.ainvoke({"question": question}, {"recursion_limit": 60, **run.config})
        latency = time.perf_counter() - t0
        records = llm.calls[first_call:]
        segments = final.get("segments", {})
        answer = final.get("answer", "")
        cited = extract_citations(answer)
        result = {
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
            "agent_turns": final.get("agent_turns", []),
        }
        run.finish({k: result[k] for k in ("answer", "model", "route", "route_reason", "cited_ids", "retrieved_ids",
                                           "steps", "citation_check", "notes", "latency_sec", "usage")})
        result["trace_id"] = run.trace_id
        return result
