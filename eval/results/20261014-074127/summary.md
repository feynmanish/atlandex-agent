# Eval run 20261014-074127

Command: `python -m eval.run --model small --arch both`

*Video hit* = the answer cites a segment of a gold video. *Segment hit* = it cites a segment whose start is inside a gold span (±60 s). They are different measures; neither is called task success. Rates are over completed runs; runs that failed because the free model stayed unavailable are shown in *runs ok/total*, not counted as misses.

## Headline, single architecture (needs_review records excluded)

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| poolside/laguna-xs-2.1:free | existing | 21/22 | 86% | 76% (n=21) | 81% | 24% (49 sent.; <= 24% if unjudged were unsupported) | 0% of 49 sent., 0 of 18 ans. | 2.0 | 86.5 / 188.8 | 5502 / 1104 | 0.0000 |
| poolside/laguna-xs-2.1:free | new_candidate | 5/7 | 80% | 80% (n=5) | 100% | 24% (21 sent.; <= 24% if unjudged were unsupported) | 0% of 21 sent., 0 of 5 ans. | 6.0 | 148.4 / 254.3 | 14605 / 2002 | 0.0000 |

Calls and guards, single: LLM calls are the agent's own (the judge is not included); HTTP attempts include retries after 429/5xx.

| model | source | LLM calls mean / max | HTTP attempts | final sentences / answer | no-coverage answers | verify dropped / failed / skipped | supervisor fallbacks | budget-forced runs |
|---|---|---|---|---|---|---|---|---|
| poolside/laguna-xs-2.1:free | existing | 4.0 / 7 | 208 | 2.5 | 0 | 0 / 0 / 0 | 0 | 0 |
| poolside/laguna-xs-2.1:free | new_candidate | 8.4 / 9 | 102 | 4.2 | 0 | 0 / 0 / 0 | 0 | 0 |

## Headline, multi architecture (needs_review records excluded)

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| poolside/laguna-xs-2.1:free | existing | 20/22 | 80% | 50% (n=20) | 80% | 19% (53 sent.; <= 19% if unjudged were unsupported) | 0% of 53 sent., 0 of 16 ans. | 1.9 | 123.9 / 296.2 | 8368 / 2217 | 0.0000 |
| poolside/laguna-xs-2.1:free | new_candidate | 5/7 | 100% | 80% (n=5) | 80% | 60% (15 sent.; <= 60% if unjudged were unsupported) | 0% of 15 sent., 0 of 5 ans. | 5.0 | 144.3 / 193.7 | 13506 / 2438 | 0.0000 |

Calls and guards, multi: LLM calls are the agent's own (the judge is not included); HTTP attempts include retries after 429/5xx.

| model | source | LLM calls mean / max | HTTP attempts | final sentences / answer | no-coverage answers | verify dropped / failed / skipped | supervisor fallbacks | budget-forced runs |
|---|---|---|---|---|---|---|---|---|
| poolside/laguna-xs-2.1:free | existing | 6.2 / 11 | 292 | 2.9 | 0 | 0 / 4 / 0 | 1 | 3 |
| poolside/laguna-xs-2.1:free | new_candidate | 10.6 / 11 | 103 | 3.0 | 0 | 0 / 0 / 0 | 0 | 5 |

## Unverified (needs_review: true), single, NOT part of the headline

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none |

## Unverified (needs_review: true), multi, NOT part of the headline

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none |

