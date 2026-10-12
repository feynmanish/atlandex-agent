# Eval run 20261012-062749

Command: `python -m eval.run --model both`

*Video hit* = the answer cites a segment of a gold video. *Segment hit* = it cites a segment whose start is inside a gold span (±60 s). They are different measures; neither is called task success. Rates are over completed runs; runs that failed because the free model stayed unavailable are shown in *runs ok/total*, not counted as misses.

## Headline (needs_review records excluded)

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | existing | 22/22 | 82% | 73% (n=22) | 86% | 15% (34 sent.; <= 15% if unjudged were unsupported) | 0% of 34 sent., 0 of 18 ans. | 1.2 | 26.4 / 78.0 | 3998 / 1374 | 0.0000 |
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 7/7 | 57% | 43% (n=7) | 100% | 33% (12 sent.; <= 33% if unjudged were unsupported) | 0% of 12 sent., 0 of 6 ans. | 3.9 | 58.8 / 115.2 | 12349 / 2645 | 0.0000 |
| poolside/laguna-xs-2.1:free | existing | 22/22 | 82% | 77% (n=22) | 82% | 23% (62 sent.; <= 23% if unjudged were unsupported) | 0% of 62 sent., 0 of 19 ans. | 1.9 | 83.4 / 193.2 | 5326 / 1022 | 0.0000 |
| poolside/laguna-xs-2.1:free | new_candidate | 7/7 | 86% | 86% (n=7) | 86% | 38% (21 sent.; <= 38% if unjudged were unsupported) | 0% of 21 sent., 0 of 7 ans. | 5.7 | 167.4 / 264.5 | 14399 / 1960 | 0.0000 |

## Unverified (needs_review: true), NOT part of the headline

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination (judged only) | unjudged | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none |
