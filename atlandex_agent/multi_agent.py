"""The multi-agent graph: router -> seed -> supervisor <-> search agent (+ tools) -> answer -> citation_check -> verify.

It reuses the single agent's nodes (router, clarify, seed, the agent loop as the "search agent", tools, answer,
citation_check) and adds a supervisor that hands out search tasks, a verifier that checks each answer sentence
against the segments it cites, and a node that drops what the verifier flagged. Everything is bounded: at most
`max_supervisor_turns` supervisor turns, the shared `max_tool_steps`, at most two answer drafts, at most two
verification rounds, and the per-run call cap enforced by CallBudget (see CALL_RESERVE in graph.py).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from . import prompts
from .citations import answer_sentences, drop_sentences
from .config import Settings
from .graph import CALL_RESERVE, NO_COVERAGE, State, bound_evidence, make_nodes, parse_json_object
from .llm import CallBudget
from .mcp_tools import ToolBox
from .verification import build_verify_input, cited_indices, parse_verify_reply, verify_feedback

SUPERVISOR_ACTIONS = ("search", "answer")
REWRITE_CALLS = 2  # an answer draft can need its empty-answer retry, so a rewrite needs room for two calls


def new_verify_info() -> dict[str, Any]:
    return {"rounds": 0, "flagged_first": [], "flagged_second": [], "dropped_unverified": [], "failed": False,
            "skipped_budget": False, "calls": []}


def build_multi_agent_graph(llm: CallBudget, tools: ToolBox, settings: Settings, budget: CallBudget):
    nodes = make_nodes(llm, tools, settings, budget)
    cap = settings.agent_cap_for(llm.model)  # a supervisor turn is a short control decision, like an agent turn
    verify_cap = settings.verify_max_tokens or settings.max_tokens

    async def supervisor(state: State) -> State:
        turns = list(state.get("supervisor_turns", []))
        notes = list(state.get("notes", []))
        n = len(turns) + 1
        steps = len(state.get("tool_log", []))
        last = n >= settings.max_supervisor_turns

        def turn(action: str, task: str = "", r=None, fallback: bool = False, forced: str | None = None) -> dict:
            return {"action": action, "task": task, "completion_tokens": r.record.completion_tokens if r else 0,
                    "finish_reason": r.finish_reason if r else None, "cap": cap, "fallback": fallback,
                    "forced": forced}

        if budget.left() - CALL_RESERVE < 1:
            return {"supervisor_turns": turns + [turn("answer", forced="budget")], "force_answer": True,
                    "notes": notes + ["call budget: forced answer"]}
        if steps >= settings.max_tool_steps:
            return {"supervisor_turns": turns + [turn("answer", forced="steps")],
                    "notes": notes + ["no tool steps left: went to the answer"]}

        given = "\n".join(f"- {t['task']}" for t in turns if t["action"] == "search") or "(none yet)"
        user = (
            f"Question: {state['question']}\nRoute: {state['route']}\n"
            f"Turn {n} of {settings.max_supervisor_turns}"
            f"{' (the last turn: the only action left is answer)' if last else ''}. "
            f"Tool calls used: {steps}/{settings.max_tool_steps}.\n\n"
            f"Tasks already given to the search agent:\n{given}\n\n"
            f"Evidence so far:\n{bound_evidence(state.get('evidence', []), settings.evidence_char_budget)}\n\n"
            "Reply with the JSON object.")
        r = await llm.chat([{"role": "system", "content": prompts.SUPERVISOR_SYSTEM},
                            {"role": "user", "content": user}], role="supervisor", max_tokens=cap)
        obj = parse_json_object(r.content) or {}
        action = obj.get("action")
        task = str(obj.get("task") or "").strip()
        fallback = False
        if action not in SUPERVISOR_ACTIONS or (action == "search" and not task):
            fallback = True
            action, task = ("answer", "") if state.get("segments") else ("search", state["question"])
            notes.append(f"supervisor output unparseable (turn {n}); fell back to {action}")
        if action == "search" and last:
            action, task = "answer", ""
            notes.append("supervisor asked to search on its last turn; went to the answer")
        elif action == "answer" and steps == 0 and not last:
            # the counterpart of the agent's nudge: an answer written before any search would be ungrounded
            action, task = "search", state["question"]
            notes.append("supervisor chose answer before any search; searched the question instead")
        out: State = {"supervisor_turns": turns + [turn(action, task, r, fallback)], "notes": notes}
        if action == "search":
            out.update(task=task, task_steps_start=steps, agent_done=False, pending=[])
        return out

    async def verify(state: State) -> State:
        answer = state["answer"]
        segments = state.get("segments", {})
        info = {**new_verify_info(), **(state.get("verify") or {})}
        sentences = answer_sentences(answer)
        todo = cited_indices(sentences)
        if answer == NO_COVERAGE or not segments or not todo:
            return {"verify": info, "feedback": "", "verify_drop": []}
        if budget.left() < 1:
            info["skipped_budget"] = True
            return {"verify": info, "feedback": "", "verify_drop": [],
                    "notes": state.get("notes", []) + ["call budget: verification skipped"]}
        r = await llm.chat(
            [{"role": "system", "content": prompts.VERIFY_SYSTEM},
             {"role": "user", "content": build_verify_input(sentences, segments)}],
            role="verify", max_tokens=verify_cap)
        info["rounds"] += 1
        flagged = parse_verify_reply(r.content, set(todo))
        info["calls"] = info["calls"] + [{
            "completion_tokens": r.record.completion_tokens, "finish_reason": r.finish_reason,
            "empty": not r.content.strip(), "parsed": flagged is not None}]
        if flagged is None:  # fail open: a broken verifier must never delete content
            info["failed"] = True
            return {"verify": info, "feedback": "", "verify_drop": [],
                    "notes": state.get("notes", []) + [f"verifier reply unusable (round {info['rounds']}); "
                                                       "nothing dropped"]}
        entries = [{"text": sentences[i], "reason": why} for i, why in flagged.items()]
        info["flagged_first" if info["rounds"] == 1 else "flagged_second"] = entries
        if not flagged:
            return {"verify": info, "feedback": "", "verify_drop": []}
        # One rewrite slot per run, shared with the citation check (answer_attempts), and only if the call budget
        # leaves room for it; otherwise the flagged sentences are dropped.
        if state.get("answer_attempts", 1) < 2 and budget.left() >= REWRITE_CALLS:
            return {"verify": info, "verify_drop": [],
                    "feedback": verify_feedback([(e["text"], e["reason"]) for e in entries])}
        return {"verify": info, "feedback": "", "verify_drop": list(flagged)}

    async def drop_unverified(state: State) -> State:
        info = {**new_verify_info(), **(state.get("verify") or {})}
        sentences = answer_sentences(state["answer"])
        idx = [i for i in state.get("verify_drop", []) if i < len(sentences)]
        info["dropped_unverified"] = info["dropped_unverified"] + [sentences[i] for i in idx]
        return {"answer": drop_sentences(state["answer"], idx) or NO_COVERAGE, "verify": info, "verify_drop": []}

    def after_supervisor(state: State) -> str:
        return "search_agent" if state["supervisor_turns"][-1]["action"] == "search" else "answer"

    def after_search_agent(state: State) -> str:
        if state.get("force_answer") or len(state.get("tool_log", [])) >= settings.max_tool_steps:
            return "answer"
        if state.get("agent_done"):
            return "supervisor"
        return "tools" if state.get("pending") else "search_agent"

    def after_tools(state: State) -> str:
        log = len(state.get("tool_log", []))
        if log >= settings.max_tool_steps:
            return "answer"
        if not state.get("task"):  # the linked videos, read before the first supervisor turn
            return "supervisor"
        used = log - state.get("task_steps_start", 0)
        return "supervisor" if used >= settings.search_steps_per_task else "search_agent"

    def after_check(state: State) -> str:
        return "answer" if state.get("feedback") else "verify"

    def after_verify(state: State) -> str:
        if state.get("feedback"):
            return "answer"
        return "drop_unverified" if state.get("verify_drop") else END

    g = StateGraph(State)
    for name in ("router", "clarify", "seed"):
        g.add_node(name, nodes[name])
    g.add_node("supervisor", supervisor)
    g.add_node("search_agent", nodes["agent"])
    g.add_node("tools", nodes["tools"])
    g.add_node("answer", nodes["answer"])
    g.add_node("citation_check", nodes["citation_check"])
    g.add_node("verify", verify)
    g.add_node("drop_unverified", drop_unverified)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: "clarify" if s["route"] == "unclear" else "seed",
                            {"clarify": "clarify", "seed": "seed"})
    g.add_edge("clarify", END)
    g.add_conditional_edges("seed", lambda s: "tools" if s.get("pending") else "supervisor",
                            {"tools": "tools", "supervisor": "supervisor"})
    g.add_conditional_edges("supervisor", after_supervisor, {"search_agent": "search_agent", "answer": "answer"})
    g.add_conditional_edges("search_agent", after_search_agent,
                            {"tools": "tools", "search_agent": "search_agent", "supervisor": "supervisor",
                             "answer": "answer"})
    g.add_conditional_edges("tools", after_tools,
                            {"search_agent": "search_agent", "supervisor": "supervisor", "answer": "answer"})
    g.add_edge("answer", "citation_check")
    g.add_conditional_edges("citation_check", after_check, {"answer": "answer", "verify": "verify"})
    g.add_conditional_edges("verify", after_verify,
                            {"answer": "answer", "drop_unverified": "drop_unverified", END: END})
    g.add_edge("drop_unverified", END)
    return g.compile()
