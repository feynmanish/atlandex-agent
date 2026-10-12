# Trajectory: baseline -> v1 -> v2

Query-runs completed in every run, per agent model. Latency is wall-clock seconds per query-run; tokens are the agent's own calls (the judge is excluded); hallucination counts judged sentences only (the first run uses its re-judged verdicts when available). `cap hits` = agent turns that used their whole token cap; `drafts` = turns with no tool call and more than 200 characters of visible text (recorded from v2 on, `n/a` before); `ending turn > 300 tok` is a proxy available for every run (it also counts hidden reasoning).

## Summary

| agent model | paired | run | steps | latency p50 / p95 (s) | tokens in / out | hallucination (judged sent.) | unjudged | segment hit | video hit | agent turns | cap hits (runs) | drafts | ending turn > 300 tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| large | 29 | baseline | 2.3 | 37.6 / 113.4 | 7308 / 2376 | 42% (50) | 0 | 76% | 79% | 89 | 5 (5) | n/a | 22 of 26 |
| large | 29 | v1 | 1.8 | 33.8 / 81.4 | 6014 / 1681 | 20% (46) | 0 | 66% | 76% | 60 | 17 (16) | n/a | 22 of 26 |
| large | 29 | v2 | 2.3 | 52.1 / 159.6 | 7552 / 2581 | 16% (64) | 0 | 86% | 86% | 71 | 2 (2) | 0 of 71 | 21 of 27 |
| small | 24 | baseline | 2.0 | 54.2 / 162.9 | 5448 / 1018 | 40% (80) | 0 | 75% | 79% | 67 | 0 (0) | n/a | 6 of 20 |
| small | 24 | v1 | 2.2 | 85.1 / 196.4 | 6298 / 1191 | 22% (68) | 0 | 79% | 83% | 52 | 3 (3) | n/a | 13 of 21 |
| small | 24 | v2 | 2.2 | 85.1 / 196.4 | 6298 / 1191 | 22% (68) | 0 | 79% | 83% | 52 | 3 (3) | n/a | 13 of 21 |

(`paired` = query-runs completed in all runs; completed per run: large: baseline 29, v1 29, v2 29; small: baseline 24, v1 29, v2 29)

## Per query

| query | model | steps | latency (s) | tokens in | hallucination (judged) | segment hit |
|---|---|---|---|---|---|---|
| ex01 | large | 1 -> 1 -> 1 | 46 -> 42 -> 63 | 4441 -> 4018 -> 3928 | 50% -> 0% -> 0% | yes -> no -> yes |
| ex02 | large | 1 -> 1 -> 1 | 46 -> 25 -> 34 | 4451 -> 3140 -> 3140 | 100% -> 100% -> 100% | yes -> yes -> yes |
| ex03 | large | 6 -> 2 -> 2 | 31 -> 79 -> 66 | 10460 -> 6637 -> 6134 | n/a -> 0% -> 0% | no -> yes -> yes |
| ex04 | large | 4 -> 1 -> 1 | 38 -> 26 -> 35 | 11671 -> 3134 -> 3134 | 50% -> 100% -> 100% | yes -> yes -> yes |
| ex05 | large | 2 -> 1 -> 1 | 46 -> 44 -> 35 | 6963 -> 3140 -> 3140 | 67% -> n/a -> 33% | yes -> no -> yes |
| ex06 | large | 3 -> 3 -> 3 | 72 -> 83 -> 120 | 10630 -> 10695 -> 10697 | 100% -> 100% -> 0% | yes -> yes -> yes |
| ex07 | large | 1 -> 2 -> 3 | 34 -> 47 -> 78 | 4329 -> 5954 -> 9316 | 100% -> 0% -> 0% | yes -> yes -> yes |
| ex08 | large | 1 -> 1 -> 2 | 30 -> 24 -> 52 | 4432 -> 3120 -> 6322 | 50% -> 0% -> 100% | yes -> yes -> yes |
| ex09 | large | 1 -> 1 -> 2 | 37 -> 26 -> 70 | 4426 -> 3116 -> 6192 | 50% -> 50% -> 0% | yes -> no -> yes |
| ex10 | large | 1 -> 1 -> 1 | 40 -> 23 -> 59 | 4435 -> 3128 -> 3922 | 33% -> 50% -> 20% | yes -> yes -> yes |
| ex11 | large | 1 -> 1 -> 1 | 50 -> 29 -> 32 | 5103 -> 3297 -> 3297 | 0% -> 0% -> 0% | yes -> yes -> yes |
| ex12 | large | 0 -> 0 -> 0 | 9 -> 8 -> 11 | 323 -> 323 -> 323 | n/a -> n/a -> n/a | no -> no -> no |
| ex13 | large | 0 -> 0 -> 0 | 6 -> 13 -> 7 | 325 -> 325 -> 325 | n/a -> n/a -> n/a | no -> no -> no |
| ex14 | large | 3 -> 2 -> 3 | 64 -> 55 -> 103 | 8904 -> 7487 -> 13803 | 0% -> 0% -> 0% | yes -> yes -> yes |
| ex15 | large | 1 -> 1 -> 1 | 31 -> 26 -> 40 | 4642 -> 3372 -> 3372 | 20% -> 0% -> 25% | yes -> yes -> yes |
| ex16 | large | 1 -> 1 -> 1 | 33 -> 36 -> 23 | 4624 -> 3344 -> 3344 | 0% -> 0% -> 0% | yes -> yes -> yes |
| ex17 | large | 1 -> 1 -> 1 | 28 -> 21 -> 30 | 4450 -> 3165 -> 3165 | 33% -> 0% -> 0% | yes -> yes -> yes |
| ex18 | large | 0 -> 0 -> 1 | 8 -> 7 -> 20 | 315 -> 315 -> 3319 | n/a -> n/a -> 0% | no -> no -> yes |
| ex19 | large | 1 -> 1 -> 1 | 41 -> 21 -> 32 | 4373 -> 3362 -> 3362 | 100% -> 0% -> 0% | yes -> yes -> yes |
| ex20 | large | 1 -> 2 -> 2 | 29 -> 58 -> 54 | 4472 -> 6990 -> 6571 | 0% -> 0% -> 0% | yes -> yes -> yes |
| ex21 | large | 1 -> 2 -> 2 | 31 -> 38 -> 45 | 4640 -> 6556 -> 6556 | 0% -> 0% -> 0% | yes -> yes -> yes |
| ex22 | large | 1 -> 1 -> 1 | 26 -> 23 -> 38 | 4427 -> 3338 -> 3338 | 0% -> 0% -> 0% | yes -> yes -> yes |
| nc02 | large | 4 -> 4 -> 4 | 81 -> 59 -> 82 | 11809 -> 12455 -> 13991 | 0% -> 0% -> 33% | yes -> yes -> yes |
| nc03 | large | 6 -> 1 -> 6 | 69 -> 14 -> 193 | 14396 -> 4467 -> 23001 | 50% -> 0% -> 50% | yes -> no -> yes |
| nc04 | large | 6 -> 5 -> 6 | 153 -> 132 -> 186 | 17300 -> 17854 -> 18657 | 100% -> 100% -> 33% | no -> no -> yes |
| nc05 | large | 4 -> 5 -> 6 | 111 -> 40 -> 113 | 16184 -> 11459 -> 17304 | n/a -> 100% -> 0% | no -> no -> no |
| nc06 | large | 5 -> 5 -> 6 | 36 -> 77 -> 38 | 11108 -> 18502 -> 12024 | 100% -> n/a -> 100% | no -> no -> no |
| nc09 | large | 6 -> 5 -> 5 | 115 -> 63 -> 93 | 14549 -> 14913 -> 14127 | 0% -> 0% -> 0% | yes -> yes -> yes |
| nc10 | large | 4 -> 2 -> 4 | 90 -> 34 -> 108 | 13752 -> 6791 -> 13207 | 67% -> 33% -> 0% | yes -> yes -> yes |
| ex01 | small | 1 -> 3 -> 3 | 68 -> 79 -> 79 | 3922 -> 8206 -> 8206 | 25% -> 20% -> 20% | yes -> yes -> yes |
| ex02 | small | 1 -> 1 -> 1 | 53 -> 112 -> 112 | 3898 -> 2857 -> 2857 | 25% -> 33% -> 33% | yes -> yes -> yes |
| ex03 | small | 5 -> 6 -> 6 | 119 -> 194 -> 194 | 11496 -> 16067 -> 16067 | 33% -> 0% -> 0% | yes -> yes -> yes |
| ex04 | small | 2 -> 1 -> 1 | 69 -> 53 -> 53 | 6118 -> 2857 -> 2857 | 0% -> 25% -> 25% | yes -> yes -> yes |
| ex05 | small | 4 -> 4 -> 4 | 241 -> 179 -> 179 | 8647 -> 11335 -> 11335 | 86% -> 0% -> 0% | yes -> yes -> yes |
| ex06 | small | 0 -> 0 -> 0 | 8 -> 16 -> 16 | 300 -> 300 -> 300 | n/a -> n/a -> n/a | no -> no -> no |
| ex07 | small | 0 -> 2 -> 2 | 6 -> 60 -> 60 | 296 -> 5519 -> 5519 | n/a -> 100% -> 100% | no -> no -> no |
| ex08 | small | 5 -> 2 -> 2 | 134 -> 85 -> 85 | 11446 -> 5348 -> 5348 | 50% -> 20% -> 20% | yes -> yes -> yes |
| ex09 | small | 4 -> 2 -> 2 | 112 -> 82 -> 82 | 9816 -> 5495 -> 5495 | 33% -> 20% -> 20% | yes -> yes -> yes |
| ex10 | small | 3 -> 1 -> 1 | 69 -> 66 -> 66 | 8059 -> 2852 -> 2852 | 17% -> 25% -> 25% | yes -> yes -> yes |
| ex11 | small | 0 -> 0 -> 0 | 11 -> 5 -> 5 | 318 -> 318 -> 318 | n/a -> n/a -> n/a | no -> no -> no |
| ex12 | small | 1 -> 1 -> 1 | 38 -> 132 -> 132 | 4169 -> 3122 -> 3122 | 25% -> 0% -> 0% | yes -> yes -> yes |
| ex13 | small | 2 -> 6 -> 6 | 65 -> 104 -> 104 | 5993 -> 13773 -> 13773 | 67% -> 50% -> 50% | no -> no -> no |
| ex14 | small | 1 -> 2 -> 2 | 60 -> 85 -> 85 | 4234 -> 6352 -> 6352 | 33% -> 0% -> 0% | yes -> yes -> yes |
| ex15 | small | 1 -> 1 -> 1 | 50 -> 29 -> 29 | 4129 -> 3131 -> 3131 | 0% -> 0% -> 0% | yes -> yes -> yes |
| ex16 | small | 2 -> 1 -> 1 | 47 -> 101 -> 101 | 5631 -> 3091 -> 3091 | 50% -> 0% -> 0% | yes -> yes -> yes |
| ex17 | small | 1 -> 3 -> 3 | 35 -> 197 -> 197 | 4088 -> 7989 -> 7989 | 100% -> 75% -> 75% | yes -> yes -> yes |
| ex18 | small | 0 -> 0 -> 0 | 4 -> 16 -> 16 | 310 -> 310 -> 310 | n/a -> n/a -> n/a | no -> no -> no |
| ex19 | small | 1 -> 1 -> 1 | 31 -> 31 -> 31 | 3861 -> 3121 -> 3121 | 33% -> 33% -> 33% | yes -> yes -> yes |
| ex20 | small | 1 -> 2 -> 2 | 54 -> 55 -> 55 | 3934 -> 6026 -> 6026 | 50% -> 0% -> 0% | yes -> yes -> yes |
| ex21 | small | 1 -> 2 -> 2 | 54 -> 109 -> 109 | 4110 -> 6026 -> 6026 | 33% -> 67% -> 67% | yes -> yes -> yes |
| ex22 | small | 1 -> 1 -> 1 | 31 -> 98 -> 98 | 3896 -> 3081 -> 3081 | 50% -> 0% -> 0% | yes -> yes -> yes |
| nc04 | small | 6 -> 6 -> 6 | 168 -> 167 -> 167 | 12654 -> 15589 -> 15589 | 38% -> 0% -> 0% | yes -> yes -> yes |
| nc06 | small | 6 -> 6 -> 6 | 78 -> 283 -> 283 | 9424 -> 18385 -> 18385 | 50% -> 100% -> 100% | no -> yes -> yes |

Runs: `baseline` = 20261011-125600; `v1` = 20261012-062749; `v2` = 20261012-fixed-v2
