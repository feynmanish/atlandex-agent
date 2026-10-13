"""python -m atlandex_agent "question" --model small|large"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import Settings
from .graph import EmptyAnswerError, run_agent
from .llm import OpenRouterClient
from .mcp_tools import open_toolbox
from .tracing import tracer_for


def format_result(res: dict) -> str:
    u = res["usage"]
    lines = [res["answer"], ""]
    if res["citations"]:
        lines.append("Citations:")
        lines += [f"  [{sid}] {link}" for sid, link in res["citations"].items()]
        lines.append("")
    cc = res["citation_check"]
    lines.append(f"route: {res['route']} ({res['route_reason']})")
    lines.append(f"steps: {res['steps']} tool call(s): " + ", ".join(t["name"] for t in res["tool_calls"]))
    lines.append(f"latency: {res['latency_sec']}s | LLM calls: {u['llm_calls']} (cached {u['cached_calls']}) | "
                 f"tokens: {u['prompt_tokens']} in / {u['completion_tokens']} out | cost: ${u['cost_usd']:.4f}")
    if cc.get("retried") or cc.get("dropped"):
        lines.append(f"citation check: retried={cc.get('retried')} dropped={len(cc.get('dropped', []))} claim(s)")
    if res.get("arch") == "multi":
        for t in res["supervisor_turns"]:
            lines.append(f"supervisor: {t['action']}" + (f" -> {t['task']}" if t["task"] else "")
                         + (f" (forced: {t['forced']})" if t.get("forced") else ""))
        v = res["verify"]
        if v:
            lines.append(f"verify: {v['rounds']} round(s), flagged {len(v['flagged_first'])}/{len(v['flagged_second'])}, "
                         f"dropped {len(v['dropped_unverified'])}" + (", verifier failed" if v["failed"] else ""))
        lines.append(f"LLM calls: {res['llm_calls_used']} of at most {res['llm_calls_cap']}")
    for n in res["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.no_cache:
        settings.cache_enabled = False
    if args.no_trace:
        settings.tracing = False
    tracer = tracer_for(settings)
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 2
    client = OpenRouterClient(settings)
    llm = client.bind(settings.model_for(args.model))
    try:
        async with open_toolbox(settings) as tools:
            res = await run_agent(args.question, llm, tools, settings, arch=args.arch or settings.arch)
    except EmptyAnswerError as e:  # a failed run, not an answer
        print(f"Run failed: {e}.", file=sys.stderr)
        return 1
    finally:
        await client.aclose()
        tracer.flush()
    if res.get("trace_id"):
        print(f"trace: {tracer.trace_url(res['trace_id']) or res['trace_id']}", file=sys.stderr)
    print(json.dumps(res, indent=2, ensure_ascii=False) if args.json else format_result(res))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="atlandex_agent", description=__doc__)
    p.add_argument("question")
    p.add_argument("--model", default="small", help="small | large | an explicit OpenRouter model id")
    p.add_argument("--arch", choices=["single", "multi"], default=None,
                   help="single: router -> agent loop -> answer; multi: adds a supervisor and a verification agent "
                        "(default: AGENT_ARCH, else single)")
    p.add_argument("--no-cache", action="store_true", help="disable the on-disk LLM response cache")
    p.add_argument("--no-trace", action="store_true", help="do not send this run to Langfuse even if keys are set")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    sys.exit(asyncio.run(amain(args)))
