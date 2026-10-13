# Decisions

Running log of choices, deviations from the task spec, and problems hit. Newest last within each section.

## Models (2026-10-11)

- **Free limit is 1,000 requests/day** (`/api/v1/key` → `free_model_daily_requests`), so the 50/day stop condition did not apply. 15 `:free` models exist, 14 with tool calling.
- **Large: `nvidia/nemotron-3-super-120b-a12b:free`** (120B total / 12B active, 262K context). Zero failed or retried calls in the smoke test. Also the judge, as the spec asks for the large model. Paid fallback: same model without `:free`, $0.08 / $0.45 per M tokens.
- **Small: `poolside/laguna-xs-2.1:free`** (33B total / ~3B active, 262K context). Paid fallback: same model, $0.06 / $0.12.
- **Deviation: not 7-14B.** No free model with tool calling is in that range on OpenRouter today. The only smaller one, `liquid/lfm-2.5-2.6b:free`, has a 65K context and is documented as unsuited to agentic use, and I did not test it. All other candidates are MoE models with 3-4B active parameters, so "small" here means small active size, not small total size. Poolside and NVIDIA are different families.
- **Why not the other small candidates.** Gemma 4 26B A4B was my first pick (different family, cheap paid fallback) but returned HTTP 429 on every call across two runs; Cohere North Mini Code worked but has no paid variant and looped on one query. Details in `COST_ESTIMATE.md`. Gemma is worth re-trying in Phase 2 via `MODEL_SMALL` since its failures looked intermittent.
- **The key has a $0 spend cap** (`limit: 0`). Paid fallback will not work until it is raised. The code does not depend on this; it only matters if you approve paid calls.

## Paid-call lock

- `ALLOW_PAID` defaults to false and `.env` sets it false. A non-`:free` model raises `PaidCallBlocked` before any request is built, tested with a mock transport that asserts nothing was sent.
- Free outage with `ALLOW_PAID=false`: retry with exponential backoff (2, 4, 8, 16, 32, 60 s, honouring `Retry-After`), then raise `LLMUnavailable`. No silent fallback.
- Budget check uses a pessimistic worst case (prompt chars / 2.5 as tokens, plus `max_tokens` of output) priced from `/api/v1/models`, so a paid call is refused *before* it is sent if it could pass `BUDGET_USD`. A model with no listed price is refused.

## Agent design

- **Own thin OpenRouter client instead of `ChatOpenAI`.** The cache, throttle, paid lock, budget and per-call usage all live in one place and are testable with a mock transport. Tools are still loaded with `langchain-mcp-adapters` and converted to OpenAI tool specs.
- **Bounded context.** Each agent turn sends the system prompt plus one user message: the question and a size-capped digest of the evidence so far (default 12,000 chars, newest results in full, older ones shrunk to their header). History is never replayed as a message list. This avoids tool-message protocol problems with small models and keeps prompts under the ~40K limit the task mentioned.
- **Structured output by tolerant JSON parsing**, not `response_format`. Free providers differ in support. The router and judge ask for JSON and the parser strips code fences and reasoning tags. The router falls back to `multi_step` (and notes it) if two attempts are unparseable.
- **Step limit counts every executed tool call**, including a repeated identical call (which returns "already called" without hitting the tool). A looping small model therefore still ends after 6.
- **Agent that answers before any search** gets one nudge ("you must call a tool first"), then proceeds. Otherwise an ungrounded answer could be produced from no evidence.
- **Answer node with no retrieved segments** returns a fixed "no indexed video covered this" reply without an LLM call.

## Segment IDs and gold matching (updated in Phase 2)

- **The tools return no segment ID**, so the agent derives `video_id@start_sec`, or `video_id@na<rank>` when the start time is null. The model cites `[id]`; the CLI maps it back to the tool's own `citation` link.
- **Two hit measures, never merged.** *Video hit*: the answer cites a segment of a gold video. *Segment hit (±60 s)*: it cites a segment whose start is inside a gold span widened by 60 s (an untimed `@na` citation can never be a segment hit). Video-level matching is never called "task success". The existing 22 and the new 10 are reported apart. Record fields: `gold_video_ids`, `gold_segments` (`{video, start, end}`), `gold_basis`.
- **Rates are over completed runs.** A run that failed because the free model stayed unavailable after 6 retries is shown in `runs ok/total`, not scored as a miss (an outage says nothing about retrieval). This changed late: the first summary scored outages as misses, which made the small model look much worse on the candidates than it was.
- **Passage start times move with the question wording** (the MCP server picks a 110-word window), so there is no stable passage ID; gold is therefore a time span, not an ID.

## Citations

- Added beyond the spec, after seeing real model output: citations written as `【id】`, `(id)` or `[id1, id2]`, and IDs containing Unicode hyphens (U+2011), are normalised to `[id]` before the check. Before this, the check silently passed answers whose citations it could not see.
- Also added: an answer with segments retrieved but **no citation at all** is retried once with feedback; if it still has none it is kept and flagged `no_citations` in the result.
- Citation drop works on sentences (split keeps a trailing `[id]` with its sentence) and keeps line breaks.

## Eval mechanics

- **`eval.run` skips `needs_review` records unless `--include-review` is passed.** The Phase 2 run used `--include-review` so the 3 failed candidates still ran; they are listed apart from the headline table.
- **Smoke set** `eval/smoke.jsonl` (3 queries, `source: smoke`, no gold) was used only for the Phase 1 measurement; it is not part of any reported metric.
- **Labelling deviation from the original spec:** the judge verdict does not sit in a hidden column of `to_label.csv` but in a separate `eval/labels/judge_verdicts.csv` keyed by `row_id`, so it can be left unopened while labelling. The label column is `reference_label` (not `human_label`) because the labels are Claude's.
- **Judge prompt calibration.** In the first smoke outputs the judge marked introductory lines and "not covered" statements as unsupported, so I changed the prompt once to count those as supported. That was done from 6 outputs, with no tuning against a score.
- **Uncited sentences count as unsupported** by the judge ("makes a factual claim while citing nothing"). In the final run, 24 of the 44 sentences the judge flagged unsupported (in completed headline runs) cited nothing.
- **Eval records** are in `eval/results/<run>/records.jsonl` (full answers; not committed); `summary.json`, `summary.md` and `judge_north-mini.jsonl` are committed for the run reported in the README. `python -m eval.summarize` rebuilds the summary from the records without calling a model.

## Judge coverage (measured 2026-10-11)

Source: the four `eval/results/20261011-12*` runs (`eval.run`, current code), de-duplicated so each (query, agent model) counts once, latest run wins. Free judge `nvidia/nemotron-3-super-120b-a12b:free`, `max_tokens` 1500. Answers that never reached the judge (clarifications, the fixed "no coverage" reply, errored runs) are excluded.

| agent model | answers judged | answers with unjudged sentences | sentences | unjudged sentences |
|---|---|---|---|---|
| `poolside/laguna-xs-2.1:free` | 10 | 2 (20%) | 43 | 11 (26%) |
| `nvidia/nemotron-3-super-120b-a12b:free` | 13 | 2 (15%) | 28 | 8 (29%) |
| all | 23 | 4 (17%) | 71 | 19 (27%) |

Far above the 10% line. A later snapshot, taken while an eval run was still adding records (`existing` queries only, new summary code), gives the same picture: 25% of 32 sentences unjudged for `nemotron` as agent model (2 of 15 answers) and 24% of 46 for `laguna` (2 of 11). Facts behind it:

- **Failure is all-or-nothing per answer:** all 4 affected answers were fully unjudged (no sentence got a verdict).
- **Cause is the token budget, not the output format.** In all 4 cases the judge made 2 calls with exactly 3000 completion tokens, i.e. 2 x the 1500 cap. Successful judgements took one call and about 1100 tokens, so they sit close to the cap. In the response cache (which does not record the role), 15 replies of any role used the whole 1500 tokens: 5 were judge replies that are valid JSON cut off mid-array, 8 had empty content (consistent with a reasoning model spending its whole budget, but the role is not recorded, so I cannot say they were all judge calls) and 2 were something else. 45 of 50 judge-shaped replies in the cache finished under the cap.
- **The retry cannot help:** it resends the identical request at temperature 0 (and the on-disk cache returns the identical reply), so it fails the same way.
- **Longer answers fail more** (5.0 sentences on average for unjudged answers vs 3.0 for judged ones), so the missing verdicts are not random: dropping them biases the hallucination rate down.
- A paid judge is **not** needed: the free model can do the job when it has room. For reference only: at the documented paid prices ($0.08 / $0.45 per M tokens) one judgement (~820 in / ~1100 out) costs about $0.0006, so 64 answers would be about $0.04 (about $0.12 even at a 4000-token ceiling). `ALLOW_PAID` stays false, and the key has a $0 cap anyway.

Reporting rule (implemented in `eval/metrics.py`, `eval/run.py`, README): an `unjudged` sentence is never counted as clean. The hallucination rate is on judged sentences only and is `n/a` when none were judged; the summary shows the unjudged share (sentences and answers), an upper bound counting unjudged as unsupported, and a warning above 10%. Langfuse gets an `unjudged_share` score and no `hallucination_rate` score for a fully unjudged answer.

## Plan: three fixes (written before implementing; all four are implemented, see each heading)

Written while an eval run was still going, so no agent code was touched until it had finished. Order: 3 first (it makes existing results usable), then 1 and 2 (they change agent behaviour, so results before and after are not comparable and must not be mixed in one table).

### 1. YouTube URL in the question -> video ID (implemented in the agent-fixes PR)

**As built:** `atlandex_agent/urls.py` finds the IDs (`watch?v=` with other parameters before or after, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`). A `seed` node between the router and the agent loop runs `search_video(video=<id>, question=<question without the link>)` for up to 3 linked videos, through the normal tool node, so it is a counted, traced, citable step. Not run for `unclear` questions. If the video has no searchable transcript the result is an error step and the agent carries on with `find_videos` as before. The agent prompt tells the model not to search a linked video again.

- **Evidence:** all 22 `existing` queries contain a `youtube.com/watch?v=` URL. In trace `ex03` the agent spent 3 `find_videos` calls (two returned 0 matches) and about 25 s before it called `search_video` on a video ID that was already in the question.
- **Change:** a small pure function `video_ids_in(question)` (regex for `watch?v=`, `youtu.be/`, `shorts/`; 11-character IDs). In the graph, when it finds IDs, run `search_video(video=<id>, question=<question without the URL>)` once before the agent loop, and add the result to the evidence like any tool call (it counts as a step, is traced, and its segments are citable). The agent loop then continues as before, so comparison questions can still search further. If the tool says the video is not searchable or returns an error, nothing special happens: the agent proceeds with `find_videos` as today.
- **Not done:** no hard skip of the agent for URL questions, and no change to `find_videos`.
- **Tests:** ID extraction (the 3 URL shapes, extra query params, no URL, two URLs); pre-seeded step appears in `tool_calls` and `segments`; non-searchable video falls through; offline with the fake toolbox.
- **Check:** re-run the 22 `existing` queries on both models; expect fewer steps and lower latency, video hit not worse.

### 2. Done-signal for the final agent turn (implemented in the agent-fixes PR)

**As built:** the agent is offered a no-argument `finish` tool next to the MCP tools (never sent to the server, never a step); a bare `READY` still works; `finish` next to real calls is ignored so the real calls run. Agent turns get `max_tokens` 600 (`AGENT_MAX_TOKENS`); other roles keep their caps. A reply that is cut off at the cap with no tool call is treated as done and noted (`LLMResponse.finish_reason` is now recorded, and cached). The prompt says the answer is written in a later step. Not done: no retry with a bigger cap.

- **Evidence:** the "reply READY" turn is where the model writes a full answer instead: its completion tokens have median 713 (p90 1455, max 1500) vs median 196 for tool-call turns; 18 of 25 final turns are over 300 tokens; it is about 17% of total latency. In trace `ex03` it took 49.8 s and 1226 tokens, and the text is thrown away.
- **Change:** (a) a per-call `max_tokens` on `LLM.chat` (shared with fix 3); agent turns get a cap of 600 (observed tool-call turns: p90 461, max 588; hidden reasoning counts toward it, so not lower). (b) Prompt: "When you have enough evidence, call the `finish` tool" and offer a no-argument `finish()` tool next to the MCP tools; any reply with no tool call is treated as finished (as today). The cap bounds the cost either way; the tool gives a structured signal instead of a text convention.
- **Risk:** a tool call truncated by the cap would be lost. Mitigation: if the finish reason is length with no parseable call, treat it as done (and note it), never retry with a bigger cap.
- **Tests:** READY text, `finish` call, prose reply and truncated reply all end the loop; the cap is passed; the `finish` call is not counted as a step and not offered to the answer node.
- **Check:** re-run; expect final-turn tokens mostly under 100 and latency down by about the 17% above.

### 3. Judge coverage (implemented in the judge-fix PR)

**Result on the Phase 2 records** (`eval/results/20261011-125600`, `eval.rejudge --only-unjudged`, main judge, free, $0.00; 7 answers re-judged, 30 sentences): unjudged sentences went from 8 of 35 (23%) to 0 for the large agent model and from 13 of 70 (19%) to 0 for the small one on the `existing` set, and from 6 of 15 (40%) to 0 on the large model's verified candidates. Hallucination rate on judged sentences (upper bound in brackets, which counts unjudged as unsupported): large/existing 41% (54%) -> 40% (40%); small/existing 39% (50%) -> 40% (40%); large/candidates 78% (87%) -> 47% (47%); small/candidates 40% -> 40%. The rate barely moves on the `existing` set, so the old numbers were not badly biased there; on the candidates they were (the missing sentences were mostly supported). Original `summary.json`/`summary.md` are unchanged; the new verdicts are in `judge_rejudged.jsonl` and the before/after table in `rejudge_report.json`. Not done: (c) below, to keep the validated judge prompt exactly as it was.

Original plan:

- **Change:** (a) per-call `max_tokens` for the judge: 4000 (the replies that finished needed about 1100; the empty ones were reasoning). (b) Make the second attempt different: if the first reply is cut off or empty, retry only the sentences that got no verdict, in batches of at most 3, so the retry is a smaller task and a different request (also bypasses the cache hit). (c) Shorter output: reasons of at most 12 words, and in the prompt "reason only for unsupported". (d) Ask the provider to turn reasoning down only if OpenRouter documents it for this model; not assumed.
- **Not chosen:** a paid judge (not needed, see above); another free judge model (the cause is budget, not capability; `eval/rejudge.py` stays available for a second opinion).
- **Existing results:** add `--only-unjudged` to `eval.rejudge`, which re-judges just the answers with unjudged sentences and writes the new verdicts into a sidecar file. Implemented without a merge step: `records.jsonl` and the original summary stay as they were, and `rejudge_report.json` holds the before/after numbers.
- **Tests:** a judge fake that truncates on the first call and succeeds on the batched retry; the cap is passed; still `unjudged` (not `supported`) if the retry also fails.
- **Done when:** unjudged sentences are at or below 10% on a re-run of the current records (target: near 0), otherwise report the remaining share as is.

### 4. Uncited sentences (added after the plan above; implemented in the uncited-sentences PR)

- **Evidence:** 24 of the 44 sentences the Phase 2 judge flagged unsupported cited nothing. Read one by one, the uncited sentences were: real claims with no ID ("Vapnik expresses doubt ...", "Thus, according to Vapnik ..."), "the segments do not cover X" statements (which are not claims), headings and ID lists, and fragments caused by the sentence splitter breaking after "Dr.".
- **Change:** (a) the answer prompt now requires an ID on every factual sentence including the first, the last and every bullet, forbids introductions, "Thus ..." conclusions and separate ID lists, and says sentences that only state non-coverage need none. (b) `citation_check` flags factual sentences without an ID, retries the answer once with those sentences named, and drops the ones still uncited from the second answer (an answer with nothing left becomes the "no coverage" reply). This replaces the old rule that kept an uncited second answer and only flagged it. (c) the sentence splitter no longer ends a sentence after Dr./Mr./Mrs./Ms./Prof./Sr./Jr./St./vs./e.g./i.e.
- **What counts as a claim** (deterministic, no LLM): a sentence with no `[id]` that has at least 4 words once markup and IDs are removed, does not end in ":", and is not a statement that the segments do not cover something (regex on "does not contain/mention/cover ...", "no segment/information ...", "none of the ...", "cannot answer ..."). This is a heuristic: an uncited claim phrased like a non-coverage statement slips through, and an unusual non-claim could be dropped. Sentences with a bare ID instead of `[id]` are treated as uncited.
- **Replay on the Phase 2 answers, no model calls:** 22 of 139 answer sentences (16%) are uncited claims, in 18 of 48 answers (nemotron 9 of 57 sentences, laguna 13 of 82). With the new prompt the model should produce fewer; the count of sentences actually flagged and dropped in a real run is reported in the README (`uncited_flagged` / `uncited_dropped` in `summary.json`).
- **Not done:** no LLM-based decision about what is factual, and the judge prompt is unchanged (it still calls uncited sentences unsupported, which is now mostly moot because they are dropped before judging).

## Fixed-version eval (2026-10-12)

- **Run:** `eval/results/20261012-062749`, `python -m eval.run --model both --no-cache` (no cache so latency is real), then `--resume` four times for runs the free small model dropped on upstream 429s (10, then 5, then 3, then 1 left; all 58 completed). Tracing on. Paid spend $0.00.
- **Request budget:** checked before starting (8 of 1000 free requests used that day; Phase 2 averaged about 6 LLM calls per run, so about 350 expected, 580 at the Phase 2 maximum), and a guard process (`--resume` runs too) was set to stop the eval below 60 remaining. It never triggered. The whole session, including the re-judge, smoke tests and the diagnostic below, used about 380 of the 1000.
- **Result in one line:** hallucination on judged sentences about halves (41% to 24% over all completed runs; unjudged 0%), the large model needs fewer steps/tokens/less time, and the large model's segment hit on the existing set falls from 82% to 73% because of the 600-token cap (below). Tables in the README.
- **Why the cap hurts the large model.** `nvidia/nemotron-3-super` is a reasoning model; its hidden thinking counts against `max_tokens`. In the baseline 26 of its 89 agent turns used 600 tokens or more (median 350, p90 1231); with the cap, 17 of 60 turns were cut at 600. A cut-off turn has no tool call, so the loop treated it as done. That is what the spec'd behaviour does ("treat as done"), and it cost: `ex01`, `ex05`, `ex09` lost their segment hit and `nc03` went from 6 steps to 1. For the small model the cap barely matters (3 of 81 turns).
- **Diagnostic, not part of any reported table** (`AGENT_MAX_TOKENS=1500`, large model, the six queries the cap affected, 6 runs, `eval/results/20261012-085736`, records not committed): `ex01`, `ex05`, `ex09` recover their segment hit; `ex03` keeps the hit it gained; `nc03` still fails but for a different reason (empty answer, below); `nc06` fails as before. So raising the cap for the large model would restore most of the loss. **Recommendation:** make the agent cap per model (about 1500 for reasoning models, 600 for the small one) or drop it for the large model; then re-run the large model's half. I did not change it, because the cap value was specified.
- **Empty answers (pre-existing, found here).** `ex05` and `nc06` (large) ended with an empty answer because the answer step used its whole 1500-token budget on hidden reasoning; the baseline had 2 such answer calls too. An empty answer passes `citation_check` (nothing to check), is not retried, and counts as no hit. Fix to consider: a larger answer budget for the large model and an "empty answer" retry. Not done.
- **Uncited sentences:** 11 flagged (all but one in the small model's multi-step runs), 1 dropped, out of 153 final sentences; 4 of 58 runs needed the citation retry. Replay of the Phase 2 answers had flagged 22 of 139 (16%): the answer prompt did most of it.
- **Latency of the small model is not a code signal:** 61% (baseline) and 64% (fixed) of its LLM calls needed retries after 429, and about 85% of its LLM time was spent in those calls.
- **A tool backend error** (`locate_quote`: "Atlandex backend did not respond properly") appeared once during the resume passes; the agent carried on with what it had.

## Fixed v2: per-model agent cap, empty-answer retry (2026-10-12)

Follow-up to the fixed-v1 finding above. Fixed v1 is kept as the intermediate step; the Phase 2 baseline is untouched.

- **Cause, stated plainly.** `nemotron-3-super-120b-a12b` is a reasoning model. The tokens it spends thinking are hidden but count against `max_tokens`, so a cap that is generous for a tool call (about 150-300 visible tokens) can be used up before the call is emitted. In the baseline 26 of its 89 agent turns used 600 tokens or more; with a 600 cap, 17 of 60 turns were cut off with no tool call, which the loop reads as "done". The small model (`laguna-xs-2.1`) does not reason: 3 of its 52 turns reached 600. The same effect, at the 1500 answer budget, produces empty answers: the model thinks until the budget is gone and returns nothing.
- **Change 1, cap per model, in config.** `Settings.agent_max_tokens_large = 1500`, `agent_max_tokens_small = 600` (`AGENT_MAX_TOKENS_LARGE` / `_SMALL`), looked up by model id with `agent_cap_for` (the paid twin of a free model gets the same cap; an explicit model id gets the small cap). `AGENT_MAX_TOKENS` still exists as an override for both, for experiments. Nothing about the cap is hard-coded in the graph.
- **Change 2, empty answers.** If the answer is empty (after stripping whitespace) it is asked again once with `answer_retry_max_tokens = 4000` (`ANSWER_RETRY_MAX_TOKENS`) and a note is kept. If it is still empty, `citation_check` raises `EmptyAnswerError`: the run fails. It is recorded by `eval.run` as an error (so `--resume` re-runs it and the *runs ok/total* column shows it), and the CLI prints "Run failed" and exits 1. An empty answer used to pass the check (nothing to check) and count as a miss.
- **Change 3, data.** Every agent turn is now recorded in the result (`agent_turns`: tool calls, whether `finish` was called, visible text length, completion tokens, finish reason, the cap in force), so capped turns and drafts written instead of calling `finish` can be counted exactly instead of guessed from token counts. `eval.trajectory` follows queries through several runs with those counts.
- **Small-model results kept.** The 29 small-model answers of fixed v1 include no empty answer (0 of 29; the two empty answers were both large-model runs), and the small cap is unchanged, so neither change could have affected a small-model run. Fixed v2 therefore reuses v1's 29 small-model records (copied into the v2 folder, then `--resume` re-ran only the large model); they are identical by construction and labelled that way.
- **v2 large-model run:** 29/29 completed, 0 failed, 0 empty final answers, 5 runs rescued by the empty-answer retry (`ex01`, `ex10`, `ex14`, `nc03`, `nc05`). 71 agent turns: 48 tool calls, 21 `finish`, 2 cut off at 1500 (`nc02` turn 4, `nc09` turn 5), 0 drafts. Segment hit on the existing set 91% (v1 73%, baseline 82%), hallucination on judged sentences 13% / 21% (existing / candidates). Cost: p50 latency 52 s per query vs 34 s (v1) and 38 s (baseline); the `finish` turns alone take a median of 545 tokens. 160 free requests used (584 to 424 remaining), guard never triggered; traces: 29 in Langfuse session `20261012-fixed-v2`.
- **Not done / open:** the 2 remaining cut-offs show 1500 is not a hard guarantee; the lower-risk option would be no cap for the large model, at the price of the draft-answer problem the cap was introduced for (0 drafts in 71 turns here, but the finish turn is still the longest). The default 1500-token answer budget is tight for the large model (5 of 29 runs needed the retry).

## Multi-agent comparison (2026-10-13)

Design and reasoning are in `PLAN.md`; this section is the **pre-registration**, written and committed before any full run, and the log of what happened afterwards. Results are added below the line marked *Results*, and nothing above that line is edited after the full run starts (corrections go below it, dated).

### What is compared

- `single` (fixed v2 as it is on `main`: router, agent loop, answer, citation check) against `multi` (router, supervisor that hands out search tasks, the same agent loop as search agent, answer, citation check, and a verification agent that checks each answer sentence against the segments it cites). Same queries, same judge, same model for every role, same token caps, same per-run request budget. The single arm is pinned by golden transcripts (`tests/test_single_unchanged.py`), generated from `main` before the refactor.
- Queries: the 29 verified ones (22 existing + 7 verified candidates), `--include-review` off. Both models, large and small, each model's two arms in one session (`--arch both`, order alternating per query, one shared throttle and judge, `--no-cache`). Large model on day 1, small on day 2.
- **Equal maximum budget, not equal spend.** Both arms are capped at 13 logical LLM calls per run (the most a single run can make, so the cap never binds for `single`); `multi` will use more on average, and calls, tokens and latency are reported next to quality. No claim of "better at equal cost".

### Primary metrics (per model; existing and candidate sets apart)

1. **Segment hit (±60 s)** on the query-runs completed in both arms.
2. **Hallucination rate on judged sentences** (unjudged sentences are neither clean nor counted), same judge, same prompt, same 4000-token cap for both arms. The judge sees only the final answer and the retrieved segments, never the architecture.

Secondary, always reported: video hit, LLM calls (mean/max), HTTP attempts, tokens, latency, final sentences per answer, no-coverage answers, sentences the verifier dropped, verifier failures and skips, supervisor fallbacks, route disagreements between the arms (the router is not cached, so the arms can route the same query differently; these are listed, not hidden).

### What counts as a result

- **Segment hit:** exact two-sided sign test on the queries where only one arm hit (discordant pairs). Only `p < 0.05` is reported as a difference, and then as "on this set" (29 queries; one query is 3.4 points, 4.5 on the existing set). Everything else is reported as **"no measurable difference"**.
- **Hallucination rate:** the difference (multi minus single) with a 95% bootstrap interval that resamples *queries*, not sentences. It is reported as a difference only if the interval excludes 0 **and** the deletion guards below pass. (PLAN.md fixes the interval and the guards but not this combination; this is the operational reading.)
- **Winning by deleting.** A hallucination gain does **not** count if, on the same model and set, final sentences per answer fall by more than 25% relative to `single`, or segment hit loses more than 1 query net. An empty answer has 0% hallucination; the table therefore always shows judged sentences in absolute terms, no-coverage answers and verifier drops next to the rate.

### Fixed before the full run

- **Verify token cap.** `verify_max_tokens` defaults to `max_tokens` (1500). If more than 20% of the *large* model's verify calls in the smoke run are empty or cut off (`verify.calls[*].empty` or `finish_reason == "length"`), it becomes 4000 before the full run and this section says so. The decision comes from the smoke run only, never from full-run metrics. The small model's verify cap stays at the same value as the large model's, so the two arms of both models use the same cap.
- **Prompts and settings are frozen** at a commit before the full run (`run_config` in `summary.json` records the commit, prompt hashes and every cap; `git_dirty` must be false). Prompt work uses only `eval/smoke.jsonl` and the three unverified candidates (`nc01`, `nc07`, `nc08`), never the 29 headline queries. Any change after the full run means a new full run of both arms.
- **Sanity band.** The fresh `single` arm is also compared with fixed v2. Segment hit within ±2 queries of fixed v2 (large model: 91% on the existing set, 20 of 22) is expected. A larger gap means the conditions moved (free-tier load, backend passage windows, uncached router replies) and is reported as such. Fixed v2's small-model half is fixed v1's records, which is a fine comparison point for segment hit but not for latency.
- **Request budget.** `--min-free-requests 60`: the run stops cleanly before a query when `GET /api/v1/key` reports fewer than 60 free-model requests left, checked once per (query, model), so both arms stop at the same point; pairing then uses only query-runs completed in both. Estimate: about 15 requests per query per model (435 per model before 429 retries; the small model needed retries on about 60% of its calls in v1, so about 700 attempts), which is why the models run on separate days.
- **Verifier is the judge's model in the large run.** The large model verifies and judges. Dropping what it flags tunes the answer to the judge's own preferences, so a lower rate can come with no better answer. Mitigations: `VERIFY_SYSTEM` is written apart from `judge_prompt.md` (a test checks they share no line longer than 40 characters); the small arm verifies with `laguna`, not the judge; and a second-opinion judge (`cohere/north-mini-code:free`, via `eval.rejudge`) re-judges both arms' large-model answers and is reported next to the main judge. Optional: 30 reference labels on a stratified sample of multi answers, same procedure as the existing judge validation.

### Choices made while implementing PLAN.md (gaps, not changes)

- A supervisor that answers before anything was searched (no linked video, no evidence) is turned into a search of the question: the supervisor's counterpart of the agent's nudge. Recorded in `notes`.
- On the last supervisor turn the LLM is still called but only "answer" is accepted (a "search" is overridden and noted). The call is spent for a decision that is not in doubt; it is kept because PLAN.md counts it as a supervisor turn.
- A verifier-requested rewrite needs room for two calls in the budget (answer plus its empty-answer retry). With less, the flagged sentences are dropped instead. The citation-check rewrite is unchanged.
- The free-request guard is checked per (query, model), not per arm, so a stop never separates the arms of one query. A check that cannot be made (network, unexpected reply) is reported and does not stop the run.

### Smoke run (2026-10-13, before the freeze; large model, `multi` only, `eval/smoke.jsonl` + `nc01`, `nc07`, `nc08`)

- **Run twice.** First with the verifier prompt as pre-registered, then with the fix below. 6 queries each, 12 query-runs, 142 free requests used that day in total (the three smoke passes, the CLI check and the earlier key reads). Records are scratch (not committed).
- **Calls.** 9-11 per run first time, 8-12 second time; never above the cap of 13, no `CallBudgetExceeded`, no supervisor fallback (every supervisor reply parsed).
- **Prompt defect found and fixed (smoke only, as the plan allows).** A live CLI run on a near-copy of `smoke1` ended with *no coverage*: the verifier flagged a correct, cited sentence twice with the reason "attributes statement to Reiner Pope not in source" / "adds speaker name not in source". The verifier saw only the transcript text, and the speaker's name is in the video metadata, not in the passage; the answer prompt tells the answerer to attribute passages to the speaker, so every such sentence would have been flagged and deleted. The judge, which also sees only the passage, does not flag these. Fix: `VERIFY_SYSTEM` says naming the speaker, host or video is fine and flags only numbers, dates, places, products and *other* people that the source lacks; each source line in the verifier's input now carries the video title. The fixed prompt flagged 2 sentences in 6 runs (one rewrite ended as a "the segments do not cover this" answer, which the guards count as a no-coverage-like answer; none was dropped). Tests: `test_the_verifier_is_shown_each_video_title_and_told_that_naming_the_speaker_is_fine`.
- **Verify token cap (the pre-registered rule).** Empty or cut-off verifier replies: 1 of 6 with the original prompt (16.7%), 0 of 6 with the fixed prompt. Neither is above 20%, so **`verify_max_tokens` stays at the default 1500** for the full run. Verifier replies used up to 1500 of 1500 tokens once (the cut-off one) and 1381 once, so a few failures in the full run are expected; they fail open, are counted (`verify_failed`) and are reported.
- **The call cap binds in most multi-step runs.** 4 of 6 runs in the first smoke and 3 of 6 in the second ended with `call budget: forced answer`: router 1 + supervisor 2 + about 5 search-agent turns (each task ends with a `finish` turn that is a full call) leaves fewer than 6 calls, so the second search task is cut after 1-2 turns. This follows from the plan's equal-maximum-budget rule (13 calls, 5 reserved), not from a bug, and it limits how much more `multi` can search than `single`. It is counted as `budget_forced_runs` in the summaries and comparison and will be read together with the results; the cap is not changed for the full run.
- **Trace check (CLI, `--arch multi`).** The Langfuse trace has the spans `router`, `seed`, `supervisor`, `search_agent`, `tools` (with `find_videos` / `search_video`), `answer`, `citation_check`, `verify`, `drop_unverified` and the `llm:<role>` generations under them.
- **Frozen after this section:** prompts, caps and limits as in the commit that carries this text. `git_dirty` must be false in the full runs.

### Results

*(added after the runs)*

## The existing 22-query set (Phase 2)

- **Source.** The author's earlier, private retrieval-evaluation project (not public; its name, paths and commit identifiers are omitted here). The query set was not committed code there: it existed as untracked files in a local clone, with an identical copy in a stash. I read the files and modified nothing in that project.
- **Files**: the query-set JSON (the data, copied to `eval/existing/rag_query_set.json`); the project's scorer script (TF-IDF cosine vs FAISS L2, recall@3 and nDCG@3, RRF fusion; read, not copied); and its saved output (only the per-query FAISS top-3 chunk IDs were used, and they are kept in `eval/existing/calibration_raw.json`).
- **Gold scheme.** 22 queries over 3 videos (`Burge5lagdM`: 10, `4oq91rzQcO8`: 8, `ViqYWhLimGg`: 4). Gold is a list of `relevant_chunk_ids`: the 0-based `chunk_index` into that video's `yt_vid_embeddings` rows (chapter-bounded chunks, binary relevance, production retrieve k=3). The queries are keyword-style retrieval probes, each scoped to one video.
- **Mapping to `video@seconds`.** The MCP tools expose no chunk index. The saved production run records, per query, the chunk indices FAISS ranked top-3. `search_video` on the same question returns passages in rank order, so passage *r* is taken to come from chunk `faiss_l2_top[r]`; that gives (video, chunk, `start_sec`) observations. Check: chunks are time-ordered, so a higher chunk index must never start earlier than a lower one. Result: **56 observations, 0 ordering violations**, so the mapping is accepted (`eval/chunk_map.py`, output `eval/existing/chunk_time_map.json`). A chunk's span is its observed passage starts, with the edge moved to the midpoint toward an adjacent observed chunk.
- **Limits of the mapping.** It is inferred, not read from the database. All 22 queries have segment-level gold, but two of them list a second gold chunk that was never retrieved (`ex17` chunk 9, `ex21` chunk 6), so only the first has a span. Where the neighbouring chunk was not observed the span is narrow (`ex21` 791-816 s, `ex15` chunk 13 1870-1892 s), so ±60 s can under-count there.
- **Question wording.** Because the original probes were scoped to a video, the question given to the agent names it (`About the video "<title>" (<url>): <original query>`); `question_raw` keeps the original. So video hit is nearly guaranteed by construction on this set and is not evidence of discovery ability; segment hit is the informative number. Expected route is `simple`.

## The 10 new candidates, verified by Claude (Phase 2)

- Verified by me (Claude), not by a human, by reading the top-3 passages of each gold video for the question. **7 pass** (`nc02`, `nc03`, `nc04`, `nc05`, `nc06`, `nc09`, `nc10`): `verified_by: "claude"`, `needs_review: false`, and `gold_segments` are single start times of the passages I judged relevant (so segment hit for these is biased toward what `search_video` returns). **3 fail** and stay `needs_review: true` with the reason in `verification_note`: `nc01` (the Dwarkesh passages give no view on scaling or timing), `nc07` (the Klarna passages describe outcomes, not organisational obstacles), `nc08` (the question is too vague for the passages). Weakest passes: `nc05` (the Andreessen side is indirect) and `nc03` (the memory side is one short clip).

## Judge validation (Phase 2)

- **Reference labels are Claude's, not human.** I labelled 30 judged answer sentences myself, using the rubric in `eval/judge_prompt.md`, **before opening `judge_verdicts.csv`** (the verdicts live in a separate file; `to_label.csv` has none). The README describes this as "validated against reference labels from a stronger model (Claude)". The labeller (Claude Sonnet 5.5) is the same session that wrote the repo, so it is not independent of the rubric.
- **Stratified sample.** Half of the rows were drawn from sentences the judge flagged unsupported, half from supported (the verdict is used only to balance the draw), because the plain rate left too few unsupported rows for kappa. Agreement % is therefore not the judge's accuracy on the natural mix.
- **Alternative judge**: `cohere/north-mini-code:free` (a different family from the NVIDIA main judge), run only on the 22 answers behind the labelled rows to save free-tier requests; 25 of the 30 rows were covered.
- **Rubric choices I applied**: a factual claim with no citation is unsupported; headings, introductions and "not covered" statements are supported; small paraphrase glosses are supported.
- **Judge failure mode seen**: 27 sentences got no verdict at all (reported as `unjudged` and excluded from the rate).

## Findings about atlandex-mcp / the backend (not changed; documented only)

1. **Null `start_sec` in `search_video` passages.** Such passages have no timestamp in their `citation`. Measured in three samples: **5 of 66 (7.6%)** passages for the 22 existing-set questions (top-3 each; 3/24 and 2/12 in two of the three videos, 0/30 in the third); **4 of 63 (6.3%)** in the evidence fetch for the 10 candidates; **1 of 184 (0.5%)** distinct passages retrieved during the full eval run. An earlier ad-hoc sample gave 6 of 20. So it is a minority of passages, and it varies by video and by time. It coincided with periods when the backend answered slowly (calls took 12-35 s and several timed out at 30 s while I collected the first two samples); the caption seek failing under load is a plausible cause, which I could not verify. Effect here: an untimed citation can never be a segment hit.
2. **`find_videos` does not match person names unless the name is an indexed term.** "Yann LeCun" returns 0 matches while two LeCun videos are indexed (they show up under "AGI"); "Eric Jang" does match (1 video). Agents that search for a person spend steps on empty results: 9 of the completed runs used the full 6 steps.
3. Smaller: `search_video` returns `title: null` for some videos that `find_videos` titles.

## What went wrong and how it was handled

- **I overwrote a finished results file.** I called `eval.run --resume` together with `--ids` while recovering from free-model failures; the first version of `--resume` rewrote `records.jsonl` from only the selected rows. The full run (64 query-runs, about 1.5 h) was lost, and I re-ran it from scratch without the cache. The fix: `--resume` now backs up to `records.jsonl.bak`, keeps every finished record and merges (`merge_records`, with a test). That is the committed version.
- **A second session** worked in this folder in parallel, added optional Langfuse tracing and merged it into `main`. It also created `eval/results/` runs of its own. My evaluation ran with `--no-trace`, so nothing from it was sent to Langfuse.
- **Free-tier failures.** Laguna XS 2.1 returned HTTP 429 for long stretches. After three passes, 7 small-model runs on the new candidates were still failing, and I stopped retrying to keep the daily quota (945 of 1,000 requests used). Those 7 are reported as missing, not as agent failures.
- The first small-model pick (Gemma 4 26B) and why it was replaced are covered above.
- `dotenv` finds `.env` relative to the calling script, not the CWD; helper scripts outside the repo failed until given an explicit path. `Settings.from_env` uses the repo root.
- Windows PowerShell 5.1 writes UTF-8 with a BOM through `Set-Content -Encoding utf8`; Python accepts it in `.py` files, but `config.py` was rewritten that way once.
