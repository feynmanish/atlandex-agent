# Eval run 20261011-125600

Command: `python -m eval.run --model both`

*Video hit* = the answer cites a segment of a gold video. *Segment hit* = it cites a segment whose start is inside a gold span (±60 s). They are different measures; neither is called task success. Rates are over completed runs; runs that failed because the free model stayed unavailable are shown in *runs ok/total*, not counted as misses.

## Headline (needs_review records excluded)

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | existing | 22/22 | 82% | 82% (n=22) | 86% | 41% (27 sent.) | 1.5 | 33.5 / 62.9 | 5129 / 1785 | 0.0000 |
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 7/7 | 71% | 57% (n=7) | 100% | 78% (9 sent.) | 5.0 | 89.6 / 141.8 | 14157 / 4234 | 0.0000 |
| poolside/laguna-xs-2.1:free | existing | 22/22 | 82% | 77% (n=22) | 82% | 39% (57 sent.) | 1.7 | 53.4 / 132.9 | 4940 / 970 | 0.0000 |
| poolside/laguna-xs-2.1:free | new_candidate | 2/7 | 50% | 50% (n=2) | 100% | 40% (10 sent.) | 6.0 | 123.1 / 163.6 | 11039 / 1550 | 0.0000 |

## Unverified (needs_review: true), NOT part of the headline

| model | source | runs ok/total | video hit | segment hit (±60 s) | router acc. | hallucination | steps | latency p50/p95 (s) | tokens in/out | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b:free | new_candidate | 3/3 | 67% | n/a (n=0) | 100% | 50% (6 sent.) | 5.3 | 54.4 / 118.1 | 12852 / 4027 | 0.0000 |
| poolside/laguna-xs-2.1:free | new_candidate | 1/3 | 0% | n/a (n=0) | 100% | 67% (3 sent.) | 6.0 | 245.8 / 245.8 | 9843 / 2297 | 0.0000 |
