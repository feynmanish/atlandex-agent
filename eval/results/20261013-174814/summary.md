# Eval run 20261013-174814

Command: `python -m eval.run --model large --arch both`

*Video hit* = the answer cites a segment of a gold video. *Segment hit* = it cites a segment whose start is inside a gold span (±60 s). They are different measures; neither is called task success. Rates are over completed runs; runs that failed because the free model stayed unavailable are shown in *runs ok/total*, not counted as misses.

## Headline, single architecture (needs_review records excluded)

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | existing | 22/22 | 82% | 82% (n=22) | 82% | 13% (30 sent.; <= 13% if unjudged were unsupported) | 0% of 30 sent., 0 of 18 ans. | 1.1 | 42.9 / 84.8 | 3840 / 1416 | 0.0000 |
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 7/7 | 71% | 57% (n=7) | 100% | 32% (22 sent.; <= 32% if unjudged were unsupported) | 0% of 22 sent., 0 of 7 ans. | 5.6 | 127.0 / 159.7 | 13455 / 3762 | 0.0000 |

Calls and guards, single: LLM calls are the agent's own (the judge is not included); HTTP attempts include retries after 429/5xx.

| model | source | LLM calls mean / max | HTTP attempts | final sentences / answer | no-coverage answers | verify dropped / failed / skipped | supervisor fallbacks | budget-forced runs |
|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | existing | 3.2 / 5 | 71 | 2.1 | 0 | 0 / 0 / 0 | 0 | 0 |
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 7.9 / 9 | 55 | 3.1 | 0 | 0 / 0 / 0 | 0 | 0 |

## Headline, multi architecture (needs_review records excluded)

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | existing | 22/22 | 91% | 86% (n=22) | 91% | 10% (30 sent.; <= 10% if unjudged were unsupported) | 0% of 30 sent., 0 of 20 ans. | 1.8 | 80.5 / 194.8 | 7824 / 2730 | 0.0000 |
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 7/7 | 57% | 57% (n=7) | 100% | 10% (20 sent.; <= 10% if unjudged were unsupported) | 0% of 20 sent., 0 of 6 ans. | 4.4 | 127.6 / 235.8 | 13922 / 5782 | 0.0000 |

Calls and guards, multi: LLM calls are the agent's own (the judge is not included); HTTP attempts include retries after 429/5xx.

| model | source | LLM calls mean / max | HTTP attempts | final sentences / answer | no-coverage answers | verify dropped / failed / skipped | supervisor fallbacks | budget-forced runs |
|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | existing | 6.5 / 10 | 142 | 2.1 | 0 | 0 / 0 / 0 | 0 | 1 |
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 10.6 / 13 | 74 | 3.0 | 1 | 1 / 1 / 0 | 0 | 6 |

## Unverified (needs_review: true), single, NOT part of the headline

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none |

## Unverified (needs_review: true), multi, NOT part of the headline

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none |

