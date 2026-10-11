"""python -m eval.rejudge --results eval/results/<run> --judge-model <openrouter id> --name <short>

Re-judges every answer of a finished run with a different model (an alternative judge) and
writes judge_<name>.jsonl next to records.jsonl. Only :free models unless ALLOW_PAID=true.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path

from atlandex_agent.config import Settings
from atlandex_agent.llm import LLMError, LLMUnavailable, OpenRouterClient

from .agreement import LABELS_DIR
from .judge import judge_answer


async def amain(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = OpenRouterClient(settings)
    llm = client.bind(args.judge_model)
    results = Path(args.results)
    out = results / f"judge_{args.name}.jsonl"
    done = 0
    only = None
    if args.labelled_only:  # just the answers behind the rows in eval/labels (saves free-tier requests)
        with (LABELS_DIR / "judge_verdicts.csv").open(encoding="utf-8", newline="") as vf:
            only = {(row["record_id"], row["model"]) for row in csv.DictReader(vf)}
    with out.open("w", encoding="utf-8") as f:
        for line in (results / "records.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("error") or r.get("clarification"):
                continue
            if only is not None and (r["id"], r["model"]) not in only:
                continue
            try:
                j = await judge_answer(llm, r["answer"], r["segments"])
            except (LLMUnavailable, LLMError) as e:
                j, err = None, f"{type(e).__name__}: {e}"
                f.write(json.dumps({"id": r["id"], "model": r["model"], "judge": None, "error": err}) + "\n")
                continue
            f.write(json.dumps({"id": r["id"], "model": r["model"], "judge": j}, ensure_ascii=False) + "\n")
            done += 1
            print(f"[{r['id']}] {r['model']}: {'skipped' if j is None else len(j['sentences'])} sentences", flush=True)
    await client.aclose()
    print(f"wrote {out} ({done} answers); paid spend ${client.paid_spend_usd:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--labelled-only", action="store_true",
                   help="only re-judge the answers that contain rows of eval/labels/judge_verdicts.csv")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
