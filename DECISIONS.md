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

## The existing 22-query set (Phase 2)

- **Source.** `feynmanish/Merchs`, but **not as committed code on `alef`**. The files are untracked in the working tree of the local clone `C:\Users\DTCSP\Merchs` (branch `footnotes-pilot`, HEAD `f4157715b4f0bf85fa8a3b6822ca0594bbbc69f2`) and are also in a stash on `alef`: `stash@{0}` = `d190d813daa3c11c184bb7f08e51cb9a964ad6b5` ("wip other files during ci push", 2026-09-08), base commit `23a9b48c6113d9b5216d1d9000dccf377666ad54`, untracked-files commit `774aec7803a7bee5d0138a6dcf2e50f94509e462`. The stashed and working-tree copies of the query set are identical apart from line endings. The tip of `alef` (`6107096ba335d950fd57286d2159f930d3d1bbd9`, the same as a fresh clone of `origin/alef`) does not contain them, and no commit on any branch ever did (`git log --all` finds none). I read them with `git show`/`git grep` and plain reads; nothing in Merchs was modified or checked out.
- **Files**: `speech_backend/eval/rag_query_set.json` (the data, copied to `eval/existing/rag_query_set.json`), `scripts/eval_rag_retrieval.py` (the scorer: TF-IDF cosine vs FAISS L2, recall@3 and nDCG@3, RRF fusion; read, not copied), `tmp/rag_retrieval_eval.json` (its saved output; only the per-query FAISS top-3 chunk IDs were used, and they are kept in `eval/existing/calibration_raw.json`).
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
