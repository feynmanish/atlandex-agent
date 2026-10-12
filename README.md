# atlandex-agent

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that answers questions about long-form video by calling the three tools of [`atlandex-mcp`](https://github.com/feynmanish/atlandex-mcp) (`find_videos`, `search_video`, `locate_quote`), plus an evaluation harness that measures video hit, segment hit (±60 s), router accuracy and hallucination rate with an LLM judge. All models run through [OpenRouter](https://openrouter.ai), on free `:free` models by default.

**Status:** the agent, the eval and two full free-model evaluations (the Phase 2 baseline and the fixed version) are done; see [Results](#results). The judge was validated against reference labels from Claude, not against human labels.

## How it works

```mermaid
flowchart TD
    Q([question]) --> R[router]
    R -- unclear --> C[clarify: one question back]
    C --> E1([end])
    R -- simple / multi_step --> S[seed: read videos linked in the question]
    S --> A[agent loop]
    A -- tool calls --> T[run MCP tools]
    S -- linked video --> T
    T -- steps left --> A
    A -- finish --> AN[answer]
    T -- 6 steps used --> AN
    AN --> CC{citation_check}
    CC -- bad IDs or sentences without an ID, first try --> AN
    CC -- still bad --> D[drop those sentences]
    CC -- ok --> E2([end])
    D --> E2
```

1. **router** classifies the question as `simple`, `multi_step` or `unclear` (JSON output).
2. **clarify** returns one clarifying question for `unclear` and stops.
3. **seed**: if the question links a YouTube video (all 22 existing eval questions do), that video (up to 3) is read first with `search_video` on its ID, so the agent does not spend steps finding it. It counts as a step.
4. **agent loop**: the model sees the question and a size-capped digest of the evidence so far and either calls tools or calls the no-argument `finish` tool (a bare `READY` still works). Agent turns are capped at 600 output tokens so a turn cannot turn into a draft answer. At most 6 tool calls in total. Context is rebuilt each turn, not accumulated.
5. **answer** writes from the retrieved passages only; every factual sentence ends with `[video_id@seconds]`.
6. **citation_check** verifies, without an LLM, that every cited ID was retrieved in this run and that every factual sentence cites a segment (a sentence that only says the segments do not cover something, a heading or a fragment is not a claim). On failure the answer is retried once with the offending sentences named; if it still fails, those sentences are dropped and that is recorded (`dropped`, `dropped_uncited`). An answer with nothing left becomes the "no coverage" reply.

Passages carry no ID of their own, so the agent derives one: `video_id@start_sec` (`video_id@na<rank>` if the start time is unknown). The CLI maps IDs back to the tools' ready-made `citation` links.

## Setup

Requires Python 3.10+ and a working [`atlandex-mcp`](https://github.com/feynmanish/atlandex-mcp) install.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env    # then fill it in
```

`.env` (never committed):

| Variable | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | your OpenRouter key |
| `ATLANDEX_API_URL` | Atlandex backend `/api` base URL (the variable atlandex-mcp reads) |
| `ATLANDEX_MCP_COMMAND` | path to the installed `atlandex-mcp` entry point |
| `ALLOW_PAID` | default `false`; with false, any non-`:free` model raises an error |
| `BUDGET_USD` | default `0`; paid calls that could pass it are refused before they are sent |

Optional: `MODEL_SMALL`, `MODEL_LARGE`, `PAID_MODEL_SMALL`, `PAID_MODEL_LARGE`, `REQUESTS_PER_MINUTE` (20), `MAX_TOOL_STEPS` (6), `LLM_CACHE` (true).

Default models: small `poolside/laguna-xs-2.1:free`, large `nvidia/nemotron-3-super-120b-a12b:free` (also the judge). Why: [DECISIONS.md](DECISIONS.md).

## Run the agent

```bash
python -m atlandex_agent "What does Reiner Pope say about splitting compute between pre-training, RL and inference?" --model small
```

Prints the answer, the citation links, route, steps, latency, tokens and cost. `--model large`, `--no-cache` and `--json` are available. LLM responses are cached on disk under `.cache/`, keyed by model and prompt. Requests are throttled to 20 per minute and retried with backoff on 429/5xx.

## Run the eval

```bash
python -m eval.run --model both --subset 5 --include-review    # 5-query trial
python -m eval.run --model both --include-review               # full run: 22 existing + 10 new candidates, both models
python -m eval.run --resume eval/results/<run> --model both --include-review   # re-run only the runs that errored
python -m eval.summarize eval/results/<run>                    # rebuild summary.json/md from records.jsonl
python -m eval.rejudge --results eval/results/<run> --only-unjudged   # re-judge only answers with unjudged sentences (sidecar + report)
python -m eval.compare --baseline eval/results/<old> --fixed eval/results/<new>   # per-query before/after (comparison.md)
python -m eval.sample_for_labels --n 30 --stratify             # eval/labels/to_label.csv (no verdicts) + judge_verdicts.csv
python -m eval.agreement                                       # main judge vs the reference labels
python -m eval.rejudge --results eval/results/<run> --judge-model <id> --name <short> --labelled-only
python -m eval.agreement --alt-judge eval/results/<run>/judge_<short>.jsonl
```

- `eval/dataset.jsonl`, one record per query: `id`, `question`, `gold_video_ids`, `gold_segments` (`{video, start, end}` in seconds), `gold_basis`, `expected_route`, `source` (`existing` | `new_candidate`), `needs_review`, `verified_by`.
- **`existing`**: the 22 hand-labelled retrieval queries from the project's own earlier retrieval eval (3 videos). Their gold labels are chunk indices, mapped to time spans with `eval/chunk_map.py` (56 observations, no ordering violations; see [DECISIONS.md](DECISIONS.md)). **`new_candidate`**: 10 multi-step questions drafted for this repo; 7 pass a check by Claude (`verified_by: claude`), 3 do not and stay `needs_review: true`.
- Two hit measures, kept separate and never merged: **video hit** (the answer cites a segment of a gold video) and **segment hit (±60 s)** (it cites a segment whose start lies inside a gold span, widened by 60 s on each side). Video hit is not task success: it only says the right video was found. A segment without a start time can never be a segment hit.
- Rates are over completed runs. A run that failed because the free model stayed unavailable after 6 retries shows up in *runs ok/total* and is not counted as a miss.
- `needs_review: true` records run only with `--include-review` and are always listed apart from the headline numbers.
- Each run writes `eval/results/<timestamp>/`: `records.jsonl` (full answers, tool calls, retrieved and cited IDs, tokens, cost; not committed), `summary.json` and `summary.md`.
- Other metrics: router accuracy, hallucination rate (share of *judged* answer sentences the judge marks unsupported by their cited text), the **unjudged share** (sentences and answers the judge gave no verdict for), steps, latency p50/p95, tokens and cost. The judge prompt is [`eval/judge_prompt.md`](eval/judge_prompt.md).
- **Unjudged is not clean.** A sentence the judge returned no usable verdict for is `unjudged`. The hallucination rate is computed on judged sentences only, never treats an unjudged sentence as supported, and is `n/a` (not 0%) when nothing was judged. Next to it the summary prints the unjudged share (sentences and answers) and an upper bound that counts every unjudged sentence as unsupported; above 10% unjudged sentences it adds a coverage warning. Read the rate together with that share.

## Observability

Optional [Langfuse](https://langfuse.com) tracing. Install the extra and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` in `.env`:

```bash
pip install -e ".[tracing]"
```

Without the keys (or without the extra) nothing changes: tracing is off, `langfuse` is never imported, and the tests and CI need neither. `--no-trace` (agent and eval) or `LANGFUSE_TRACING=false` turns it off even when keys are set. The CLI prints the trace URL.

**What a trace shows.** One agent run is one trace named `atlandex-agent`. Its input is the question and its output is the answer with route, cited and retrieved IDs, citation-check result, notes, latency and token/cost totals. Below it is the LangGraph run with one span per node: `router` (its output is the route and reason), `agent` and `tools` (once per loop turn), `answer`, and `citation_check` (retry or dropped claims). Inside the nodes are one `llm:<role>` generation per model call (model, messages, reply or tool calls, input/output tokens, cost, and whether it was a cache hit or a retry) and one tool span per MCP call (arguments and result; failed calls are marked as errors). Cost is `0` on free models.

**Eval runs.** Each trace is tagged `run:<results dir name>`, `model:<model id>` and `query:<record id>`, and the run id is also the Langfuse session, so one eval run is one session. After the judge has run, these scores are attached to the trace: `hallucination_rate` (share of judged sentences marked unsupported; the comment lists them), `task_success` (cites a gold segment, only for records with gold segments), `video_hit` (cites a gold video), and `router_correct`, plus `unjudged_share` (share of the answer's sentences the judge gave no verdict for). When the judge gave no verdict at all there is no `hallucination_rate` score: an unjudged answer is never scored as clean. The judge's own calls are not traced.

Prompts and retrieved passages are sent to Langfuse as trace content, and tool results longer than 8000 characters are cut in the trace.

## Tests

```bash
pytest
```

All tests run offline with a fake LLM, a fake toolbox and a mock HTTP transport: router branches, the step limit, the paid-call lock, retry/throttle/cache/budget, citation check (pass, retry, drop), metrics and the agreement script. CI runs them on Python 3.10 and 3.13 without secrets.

## Results

Two runs of the same 29 verified queries, never mixed in one table: the **fixed version** (this section) and the **Phase 2 baseline** from before the fixes (further down). Free models only, paid spend $0.00, tracing on. The judge is `nvidia/nemotron-3-super-120b-a12b:free`. Treat all of it as a small-sample snapshot, not a ranking: 22 and 7 queries per model.

### Fixed version

Run [`eval/results/20261012-062749`](eval/results/20261012-062749/summary.md) (`python -m eval.run --model both --no-cache`, then four `--resume` passes for runs the free small model dropped on upstream HTTP 429s; all 58 query-runs completed). Code: everything in [DECISIONS.md](DECISIONS.md) "Plan: three fixes" plus fix 4: judge budget and batched retry, uncited sentences checked and dropped, linked videos read first, `finish` tool with a 600-token cap on agent turns.

**Existing 22 queries**

| model | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged sentences) | unjudged sentences | steps (mean) | latency p50 / p95 (s) | tokens in / out (agent, mean) |
|---|---|---|---|---|---|---|---|---|---|
| nemotron-3-super-120b (large) | 22/22 | 82% | 73% | 86% | 15% (34) | 0 of 34 | 1.2 | 26.4 / 78.0 | 3,998 / 1,374 |
| laguna-xs-2.1 (small) | 22/22 | 82% | 77% | 82% | 23% (62) | 0 of 62 | 1.9 | 83.4 / 193.2 | 5,326 / 1,022 |

**New multi-step questions, the 7 verified by Claude**

| model | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged sentences) | unjudged sentences | steps (mean) | latency p50 / p95 (s) | tokens in / out (agent, mean) |
|---|---|---|---|---|---|---|---|---|---|
| nemotron-3-super-120b (large) | 7/7 | 57% | 43% | 100% | 33% (12) | 0 of 12 | 3.9 | 58.8 / 115.2 | 12,349 / 2,645 |
| laguna-xs-2.1 (small) | 7/7 | 86% | 86% | 86% | 38% (21) | 0 of 21 | 5.7 | 167.4 / 264.5 | 14,399 / 1,960 |

Task success here means the two hit rates above, kept apart as before (video hit: cites a gold video; segment hit: cites a segment inside a gold span ±60 s). Unjudged sentences are 0 everywhere, so the hallucination rate and its upper bound are the same number. Hit rates are over 22 or 7 queries, so one query is 4.5 or 14 points.

- **Uncited sentences:** across the 58 runs the answer check flagged 11 factual sentences without an ID (153 final sentences in total) and dropped **1** after the retry. Replaying the old Phase 2 answers through the same check had flagged 22 of 139 sentences (16%), so the new answer prompt did most of the work and the drop is the safety net.
- **The 600-token cap on agent turns hurts the large model.** An agent turn hit the cap in 19 of 58 runs (large 16, small 3). The large model is a reasoning model whose hidden thinking counts against the cap, so a turn can be cut off before it emits a tool call; the agent then treats it as done (noted in `notes`). This is the main reason the large model's existing-set segment hit fell from 82% to 73%: `ex01`, `ex05`, `ex09` lost their hit, and the multi-step `nc03` went from 6 steps to 1. A diagnostic re-run of those queries with the cap at 1500 (large model only, 6 queries, not part of any table above) recovered `ex01`, `ex05` and `ex09`. See DECISIONS.md.
- **Empty answers:** 2 large-model answers are empty because the answer step ran out of its own 1500-token budget on hidden reasoning (the Phase 2 baseline had 2 such answer calls as well). An empty answer currently passes the citation check, counts as no hit, and is not retried. Not fixed here.
- **The small model's latency is mostly provider noise.** 61–64% of its LLM calls needed retries after HTTP 429 in both runs, and about 85% of its LLM time was spent in those retried calls, so its latency rises and falls with the free tier's load rather than with the code.

### Before / after (same queries, paired)

Only query-runs completed in both runs are compared (large 29, small 24 of 29). Latency is wall-clock seconds per query-run and tokens are the agent's own calls (the judge is excluded). The baseline's hallucination rate here uses the re-judged verdicts (judge fix applied to the Phase 2 records), so both sides come from the same judge code. Per-query rows: [`comparison.md`](eval/results/20261012-062749/comparison.md) (rebuild with `python -m eval.compare --baseline eval/results/20261011-125600 --fixed eval/results/20261012-062749`).

| agent model | paired runs | | steps (mean) | latency p50 / p95 (s) | tokens in / out | hallucination (judged sentences) | unjudged sentences |
|---|---|---|---|---|---|---|---|
| nemotron (large) | 29 | before | 2.3 | 37.6 / 113.4 | 7,308 / 2,376 | 42% (50) | 0 |
| | | after | 1.8 | 33.8 / 81.4 | 6,014 / 1,681 | 20% (46) | 0 |
| laguna (small) | 24 | before | 2.0 | 54.2 / 162.9 | 5,448 / 1,018 | 40% (80) | 0 |
| | | after | 2.2 | 85.1 / 196.4 | 6,298 / 1,191 | 22% (68) | 0 |

Large: fewer steps, less latency and fewer tokens (the linked video is read first, and no more draft answers in the "done" turn), and the hallucination rate halves. Small: steps and tokens go up slightly (the first search result is now in every later prompt) and the latency change is dominated by the 429 retries described above, so it says nothing about the code. The hallucination drop for both models comes mostly from uncited sentences no longer appearing. Over all completed runs, 53 of 130 judged baseline sentences were unsupported and 25 of those cited nothing; in the fixed run 31 of 129 were unsupported and 8 cited nothing. Among sentences that do cite, the unsupported share also fell (29% to 19%), which I cannot attribute to a cause.

### Phase 2 baseline (before the fixes)

**This is the baseline run, from before the judge fix, the uncited-sentence check, the linked-video step and the finish tool. Do not compare it with the fixed-version tables above except through the paired table.**

Source: [`eval/results/20261011-125600/summary.json`](eval/results/20261011-125600/summary.json) (rendered in `summary.md`), one run, free models only, paid spend $0.00. The agent was run twice per query, once with each model; the judge is `nvidia/nemotron-3-super-120b-a12b:free`. Hit rates are over completed runs (*ok/total* shows how many runs finished). Read these as a small-sample snapshot, not a ranking: 22 and 7 queries.

**Existing 22 queries** (hand-labelled retrieval probes, each naming its video, so video hit is close to guaranteed by construction; segment hit is the informative number)

| model | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judge, upper bound) | steps (mean) | latency p50 / p95 (s) |
|---|---|---|---|---|---|---|---|
| nemotron-3-super-120b (large) | 22/22 | 82% | 82% | 86% | 41% (27 judged sentences) | 1.5 | 33.5 / 62.9 |
| laguna-xs-2.1 (small) | 22/22 | 82% | 77% | 82% | 39% (57 judged sentences) | 1.7 | 53.4 / 132.9 |

**New multi-step questions, the 7 verified by Claude** (comparison and find-then-follow-up questions; segment gold is single start times of passages Claude judged relevant)

| model | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judge, upper bound) | steps (mean) | latency p50 / p95 (s) |
|---|---|---|---|---|---|---|---|
| nemotron-3-super-120b (large) | 7/7 | 71% | 57% | 100% | 78% (9 judged sentences) | 5.0 | 89.6 / 141.8 |
| laguna-xs-2.1 (small) | **2/7** | 50% (n=2) | 50% (n=2) | 100% | 40% (10 judged sentences) | 6.0 | 123.1 / 163.6 |

The small model's 5 missing runs are upstream HTTP 429 outages (it was retried in three passes; I stopped to keep the daily free quota), so its multi-step row rests on 2 queries and says almost nothing. The 3 candidates that failed Claude's check (`nc01`, `nc07`, `nc08`) ran too but are not in the headline: large 3/3 completed, small 1/3.

Cost: $0.00 (free models). Mean tokens per completed run, existing set: large 5,129 in / 1,785 out, small 4,940 in / 970 out.

How to read the hallucination column: the judge over-flags (see below), 24 of the 44 sentences it flagged cited nothing at all, and 27 sentences got no verdict and are excluded (see the re-judge below). In the 30-row validation sample the judge's flags were right 9 times out of 15, so the true rate under the reference rubric is probably well below these numbers. Treat them as an upper bound.

**Re-judging the unjudged sentences.** The 27 (30 with the candidates the headline leaves out) were unjudged because the free judge ran out of its 1500-token budget. The judge now gets 4000 tokens, keeps the verdicts of a cut-off reply and re-asks only for the missing sentences (`python -m eval.rejudge --results eval/results/20261011-125600 --only-unjudged`). On this run it re-judged 7 answers and left 0 sentences unjudged. The table above is the original, as published; the re-judged numbers are in [`rejudge_report.json`](eval/results/20261011-125600/rejudge_report.json):

| model | source | hallucination, judged only (upper bound), original | after re-judge | unjudged sentences, original | after |
|---|---|---|---|---|---|
| nemotron-3-super-120b (large) | existing | 41% (54%), 27 judged | 40% (40%), 35 judged | 8 of 35 | 0 |
| laguna-xs-2.1 (small) | existing | 39% (50%), 57 judged | 40% (40%), 70 judged | 13 of 70 | 0 |
| nemotron-3-super-120b (large) | verified candidates | 78% (87%), 9 judged | 47% (47%), 15 judged | 6 of 15 | 0 |
| laguna-xs-2.1 (small) | verified candidates | 40% (40%), 10 judged | 40% (40%), 10 judged | 0 of 10 | 0 |

The upper bound counts an unjudged sentence as unsupported; once every sentence has a verdict, rate and upper bound are the same number.

### Judge validation

**Validated against reference labels from a stronger model (Claude), not human labels.** Claude (Sonnet 5.5) labelled 30 judged answer sentences against the rubric in `eval/judge_prompt.md` before looking at any judge verdict; the sample is stratified (half from sentences the judge flagged, half from the rest), so agreement is not the judge's accuracy on the natural mix. Labels: [`eval/labels/to_label.csv`](eval/labels/to_label.csv); verdicts: `judge_verdicts.csv`.

| judge | rows | agreement with reference | Cohen's kappa | notes |
|---|---|---|---|---|
| main: nemotron-3-super-120b:free | 30 | 80.0% | 0.60 | caught 9 of 9 reference-unsupported sentences, 6 false alarms (flagged 15, reference says 9) |
| alternative: cohere/north-mini-code:free | 25 | 76.0% | 0.53 | answers behind 5 rows were not re-judged |
| main vs alternative | 55 sentences | 92.7% | 0.85 | the 22 re-judged answers only |

So the main judge is usable but strict: it rarely misses an unsupported claim and often flags a supported one.

## Limitations

- The eval set is small (22 existing queries, 7 verified new ones), so differences of a few points are noise. The small model has only 2 completed multi-step runs.
- The existing set's gold labels are chunk indices, mapped to time spans by inference (consistent across 56 observations, but not read from the database). Some spans are narrow, so segment hit can under-count there. Its questions name the video, so it measures citing the right passage, not finding the video.
- The 7 verified candidates were checked by Claude, not by a human, and their segment gold comes from the same retrieval the agent uses. Three candidates failed that check and are excluded from the headline.
- The judge's agreement was measured against Claude's labels on 30 stratified rows. Both are LLMs and the labeller wrote the rubric, so this is a consistency check, not human ground truth.
- The "small" model is a ~3B-active MoE, not a 7-14B model; no free tool-calling model of that size exists on OpenRouter. Free models are rate-limited and intermittently down: a full run took about 1.5 hours and needed several re-run passes, and some runs still failed.
- Passage start times move with the question wording, and a minority of passages come back with no start time (0.5% to 7% in three samples, higher in an earlier ad-hoc one), which can never be a segment hit.
- `find_videos` matches index terms, not names, so questions about a person who is not an indexed term cost the agent steps (9 completed runs used all 6). These are atlandex-mcp findings, not fixed here; see [DECISIONS.md](DECISIONS.md).
- The Langfuse `task_success` score (added with the tracing) is the segment-hit measure above; it is not video hit.
- The 600-token cap on agent turns truncates the large (reasoning) model in about 28% of its agent turns; a diagnostic with a cap of 1500 recovered 3 of the 4 queries it broke (see Results and DECISIONS.md). The cap is kept as specified, so the fixed-version numbers for the large model include that cost.
- An empty answer (the answer step spends its 1500 tokens on hidden reasoning) is not detected or retried; 2 of 58 fixed-version runs and 2 baseline answer calls hit this.
- The uncited-claim check is a heuristic (see DECISIONS.md): it can miss an uncited claim phrased like a "not covered" statement, and it treats a bare ID without brackets as uncited.
