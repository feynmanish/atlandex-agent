# Cost estimate (end of Phase 1)

Written 2026-10-11 from a 3-query smoke test. Paid spend so far: **$0.00** (OpenRouter credits `total_usage` was 0.04195678 before and after; `ALLOW_PAID=false` throughout).

## Free tier on this key

| Item | Value |
|---|---|
| Daily free-model request limit | **1,000** (not 50); 90 used by the end of Phase 1 |
| Credits | $15.00 total, $0.04 used |
| Key-level spend cap | `limit: 0`, `limit_remaining: 0`. If that is a real $0 cap, paid calls on this key will be refused until you raise it. I can't verify without making a paid call. |

## Models

| Role | Model | Free ID | Paid fallback and price (per M tokens in / out) |
|---|---|---|---|
| small | Poolside Laguna XS 2.1 (33B total, ~3B active) | `poolside/laguna-xs-2.1:free` | `poolside/laguna-xs-2.1`, $0.06 / $0.12 |
| large, and judge | NVIDIA Nemotron 3 Super (120B total, 12B active) | `nvidia/nemotron-3-super-120b-a12b:free` | `nvidia/nemotron-3-super-120b-a12b`, $0.08 / $0.45 |

No free tool-calling model in the 7-14B range exists on OpenRouter today; the smallest usable ones are ~3B-active MoE models. See `DECISIONS.md`.

## Measured per query (3 queries, free models, no cache)

Per-query rows are in `docs/phase1-measurements.json`. Averages over the 3 queries:

| | Agent LLM calls | Agent tokens in / out | Tool steps | Agent latency | Judge calls | Judge tokens in / out |
|---|---|---|---|---|---|---|
| small (Laguna XS 2.1) | 6.3 | 7,959 / 866 | 3.3 | 81 s | 1.0 | 1,038 / 1,088 |
| large (Nemotron 3 Super) | 6.0 | 8,542 / 2,638 | 3.3 | 53 s | 1.3 | 1,229 / 1,177 |

Judge calls always go to the large model, also when the agent is the small one. The judge latency was not recorded. Three queries is a tiny sample; one of them hit the 6-step limit on both models, and the averages will move.

## One full eval run (32 queries x both models)

- Requests: (6.3 + 1.0) + (6.0 + 1.3) = 14.7 per query pair, so **about 469 requests per full run**. That is 47% of the daily free limit, so roughly 2 full runs per day if no request is wasted.
- Failed attempts probably count against the daily limit (not verified). Laguna needed a retry on 14 of 19 calls in the test, which would add about 150% to its requests; Nemotron needed none.
- Time at 20 requests/minute: at least 23.5 minutes if only the throttle mattered. Measured agent latency alone gives **about 71 minutes** (sequential), plus the judge calls. Expect around 1.5 hours.

## Scenarios

**A. All free.** $0. A full run is about 469 requests, so 2 runs per day at most. Free models are flaky (see below); some queries may fail and need a re-run, which the cache makes cheap.

**B. Every call on the paid fallbacks.** Computed from the measured tokens and the prices above:

| | per query pair | one full run (32 pairs) |
|---|---|---|
| small agent + large judge | $0.00115 | $0.037 |
| large agent + large judge | $0.00250 | $0.080 |
| **total** | **$0.00365** | **$0.117** |

**C. Realistic development budget** (ten 5-query subset runs + 2 full runs, about 1,670 requests):

| | cost | days of free limit |
|---|---|---|
| free only | $0 | about 2 days (more with retries) |
| fully paid | **about $0.42** | n/a |

## Recommended settings

- `ALLOW_PAID=false` first. Run Phase 2 free-only; the paid fallback only triggers when a free model is unavailable after all retries.
- `BUDGET_USD=1.00` if you do approve paid calls. That is about 2.4x the fully paid development scenario, enough headroom for reasoning-model token variance (one smoke query used 4,200 output tokens on its own). The code refuses any paid call whose worst case would pass the budget.
- Raise the key's spend cap above $0 if you want paid fallback to work.

## Free-model failures seen in the 3-query test

- `google/gemma-4-26b-a4b-it:free` (my first small pick): every call returned HTTP 429 "Provider returned error" through all 6 retries in two separate runs; all 6 query-runs failed. A single probe request succeeded between those runs, so it is intermittent, not dead. Not used.
- `google/gemma-4-31b-it:free`: HTTP 429 in a probe. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`: HTTP 502 "ResourceExhausted" in a probe.
- `poolside/laguna-xs-2.1:free`: 14 of 19 calls needed 1-2 retries (429), all eventually succeeded; slowest model (81 s per query).
- `cohere/north-mini-code:free` (tried as small): worked, but one query looped on the same `find_videos` call five times and ended with "no coverage". It has no paid variant, so I did not pick it.
- `nvidia/nemotron-3-super-120b-a12b:free`: no errors or retries; all tool calls well-formed.
- Agent-side problems found and fixed: citations written as `【id】` or with a Unicode non-breaking hyphen inside the ID slipped past the citation check (now normalised); an answer with no citations at all passed the check (now retried once, then flagged).
- Not fixed, model behaviour: on the Yann LeCun / Eric Jang query both models hit the 6-step limit searching names (`find_videos` matches index terms, and "Yann LeCun" has no match). The answer correctly said LeCun was not covered.
