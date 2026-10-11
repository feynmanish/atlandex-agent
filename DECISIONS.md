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

## Segment IDs and gold matching

- **The tools return no segment ID**, so the agent derives `video_id@start_sec`, or `video_id@na<rank>` when the start time is null. The model cites `[id]`; the CLI maps it back to the tool's own `citation` link.
- **Passage IDs are not stable across questions.** The MCP server picks a 110-word window that depends on the question wording, and many passages come back with `start_sec: null` ("could not be located in the captions": 6 of the 20 passages I looked at while drafting the candidates, spread over six different videos). So exact-ID gold matching would under-count correct answers.
- **Matching rule** (`eval/metrics.py`): exact ID; or a bare video ID in `gold_segment_ids` matches any segment of that video; or same video with start times within `--gold-tolerance-sec` (default **60**, reported in `summary.json`). The 10 new candidates use bare video IDs. **This is a choice I made without seeing your 22-query set**; if its gold IDs use another scheme, `eval/import_existing.py` and `segment_matches` are the two places to change.
- "Task success = cites at least one gold segment" is the spec's definition. For comparison questions with two gold videos it is satisfied by citing either, which is lenient. Not changed.

## Citations

- Added beyond the spec, after seeing real model output: citations written as `【id】`, `(id)` or `[id1, id2]`, and IDs containing Unicode hyphens (U+2011), are normalised to `[id]` before the check. Before this, the check silently passed answers whose citations it could not see.
- Also added: an answer with segments retrieved but **no citation at all** is retried once with feedback; if it still has none it is kept and flagged `no_citations` in the result.
- Citation drop works on sentences (split keeps a trailing `[id]` with its sentence) and keeps line breaks.

## Eval

- **`EXISTING_EVAL_PATH` was `<FILL IN>`.** I searched `atlandex-mcp/eval/` (a 10-question live-run set with no gold segment IDs, not your 22-query set), `~/rag`, `~/proj` and Downloads, and found no 22-query file with gold IDs. `eval/dataset.jsonl` therefore holds only the 10 new candidates. Once you have the path: `python -m eval.import_existing PATH`.
- **The 10 new questions** (`nc01`-`nc10`) were drafted from `find_videos` and `search_video` calls against the live index (candidate evidence in each record). All are `source: new_candidate`, `needs_review: true`, and are excluded from headline numbers. The evidence is from one top-1 passage per video, so nothing about them is verified.
- **`eval.run` skips `needs_review` records unless `--include-review` is passed**, so a default run over today's dataset selects nothing. Pass `--include-review` to run the candidates (they stay out of the headline table).
- **Smoke set** `eval/smoke.jsonl` (3 queries, `source: smoke`, no gold) was used only for the Phase 1 measurement; it is not part of any reported metric.
- **Labelling deviation:** the spec says the judge verdict sits in a hidden column of `to_label.csv`. I put it in a separate `eval/labels/judge_verdicts.csv` keyed by `row_id`, so you can't see it while labelling.
- **Judge prompt calibration.** In the first smoke outputs the judge marked introductory lines ("Here is what the video says:") and "not covered" statements as unsupported. I changed the prompt to count those as supported (they assert nothing about a video). This was done once, from 6 outputs, with no tuning against a score. Whether the judge is trustworthy is exactly what `eval.agreement` is for; until you label rows, the hallucination rate is unvalidated.
- **Uncited sentences count as unsupported** by the judge, per the rule "makes a factual claim while citing nothing".

## Problems hit

- 2026-10-11 first smoke run: the small model (Gemma) was rate-limited upstream on every call; the large model showed the citation-format bugs above. Fixed what was agent-side, swapped the small model.
- `dotenv` finds `.env` relative to the calling script, not the CWD; helper scripts outside the repo failed until given an explicit path. `Settings.from_env` uses the repo root.
- Windows PowerShell 5.1 writes UTF-8 with a BOM through `Set-Content -Encoding utf8`; Python accepts it in `.py` files, but `config.py` was rewritten that way once.

## atlandex-mcp observations (not changed; per the task I don't modify that repo)

- 6 of the 20 `search_video` passages I saw had `start_sec: null`, and their `citation` has no timestamp. That makes timestamp-based gold matching and citations less useful on those videos. The atlandex-mcp README already documents this as a known limit.
- `search_video` returned `title: null` and `channel: null` for several searchable videos where `find_videos` has the title. The citation falls back to the bare video ID.
- `find_videos` matches index terms only: "Yann LeCun" has 0 matches although two LeCun videos are indexed (they appear under "AGI"). Models that search for a person's name waste steps; the agent prompt now says to fall back to the topic term, which helped only partly.
