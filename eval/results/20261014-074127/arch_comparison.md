# Before / after: `20261014-074127:single` (baseline) vs `20261014-074127:multi` (fixed version)

Paired query-runs only (completed in both). Latency is wall-clock seconds per query-run; tokens and LLM calls are the agent's own (the judge is not included). Hallucination = judged sentences marked unsupported; the baseline uses the re-judged verdicts when available, so both sides come from the same judge code.

## Summary

| agent model | paired | | steps | LLM calls mean / max | latency p50 / p95 (s) | tokens in / out | hallucination (judged sent.) | unjudged sent. | uncited flagged / dropped |
|---|---|---|---|---|---|---|---|---|---|
| small | 22 of 26/25 | before | 2.3 | 4.4 / 9 | 90.6 / 266.5 | 6332 / 1236 | 20% (55) | 0 | 4 / 0 |
| small | 22 of 26/25 | after | 2.3 | 6.5 / 11 | 132.7 / 293.6 | 8727 / 2229 | 27% (59) | 0 | 5 / 0 |
| small|existing | 19 of 26/25 | before | 1.7 | 3.8 / 7 | 80.3 / 197.1 | 4962 / 1109 | 21% (43) | 0 | 2 / 0 |
| small|existing | 19 of 26/25 | after | 1.8 | 5.9 / 11 | 116.0 / 296.8 | 7848 / 2182 | 20% (49) | 0 | 3 / 0 |
| small|new_candidate | 3 of 26/25 | before | 6.0 | 8.3 / 9 | 148.4 / 258.3 | 15008 / 2043 | 17% (12) | 0 | 2 / 0 |
| small|new_candidate | 3 of 26/25 | after | 5.0 | 10.3 / 11 | 150.7 / 199.1 | 14293 / 2527 | 60% (10) | 0 | 2 / 0 |
| all | 22 of 26/25 | before | 2.3 | 4.4 / 9 | 90.6 / 266.5 | 6332 / 1236 | 20% (55) | 0 | 4 / 0 |
| all | 22 of 26/25 | after | 2.3 | 6.5 / 11 | 132.7 / 293.6 | 8727 / 2229 | 27% (59) | 0 | 5 / 0 |

(*paired* = query-runs completed in both runs, then completed in baseline / completed in fixed run.)

## Quality and guards

Segment hit (±60 s) and video hit over the paired queries that have gold. *Discordant* = queries where only one side hit; *p* = exact two-sided sign test on them. *Hallucination diff* = after minus before, with a 95% bootstrap interval over queries (not sentences). The guards show whether a lower hallucination rate came from deleting content: final sentences per answer, no-coverage answers, sentences the verifier dropped.

| agent model | | segment hit | video hit | discordant before-only / after-only (p) | hallucination diff [95% CI] | final sent. / answer | no-coverage | verify dropped | supervisor fallbacks | budget-forced runs | route disagreements |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | before | 86% (n=22) | 86% | 8 / 0 (p=0.008) | +7 pts [-8, +25] | 2.6 | 0 | 0 | 0 | 0 | 1 (ex07) |
| small | after | 50% (n=22) | 82% |  |  | 2.9 | 0 | 0 | 1 | 5 |  |
| small|existing | before | 84% (n=19) | 84% | 7 / 0 (p=0.016) | -1 pts [-14, +18] | 2.4 | 0 | 0 | 0 | 0 | 1 (ex07) |
| small|existing | after | 47% (n=19) | 79% |  |  | 2.8 | 0 | 0 | 1 | 2 |  |
| small|new_candidate | before | 100% (n=3) | 100% | 1 / 0 (p=1.000) | +43 pts [+17, +100] | 4.0 | 0 | 0 | 0 | 0 | 0 |
| small|new_candidate | after | 67% (n=3) | 100% |  |  | 3.3 | 0 | 0 | 0 | 3 |  |
| all | before | 86% (n=22) | 86% | 8 / 0 (p=0.008) | +7 pts [-8, +25] | 2.6 | 0 | 0 | 0 | 0 | 1 (ex07) |
| all | after | 50% (n=22) | 82% |  |  | 2.9 | 0 | 0 | 1 | 5 |  |

## Per query

| query | model | steps | latency (s) | tokens in | tokens out | hallucination (judged) | unjudged | uncited dropped | LLM calls | segment hit | final sent. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ex01 | small | 3 -> 1 | 142 -> 132 | 7645 -> 2775 | 1967 -> 2981 | 33% -> 0% | 0 -> 0 | 0 | 5 -> 4 | yes -> yes | 3 -> 3 |
| ex02 | small | 1 -> 2 | 32 -> 99 | 2857 -> 9739 | 983 -> 2646 | 0% -> 100% | 0 -> 0 | 0 | 3 -> 8 | yes -> yes | 3 -> 1 |
| ex03 | small | 4 -> 4 | 136 -> 256 | 13857 -> 18595 | 2428 -> 4119 | 0% -> 0% | 0 -> 0 | 0 | 7 -> 11 | yes -> yes | 1 -> 1 |
| ex04 | small | 1 -> 1 | 50 -> 116 | 3731 -> 2891 | 1917 -> 2584 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 4 | yes -> yes | 1 -> 4 |
| ex05 | small | 3 -> 4 | 189 -> 188 | 8151 -> 13332 | 1599 -> 3094 | 0% -> 0% | 0 -> 0 | 0 | 5 -> 9 | yes -> no | 2 -> 1 |
| ex06 | small | 0 -> 0 | 10 -> 66 | 300 -> 300 | 426 -> 477 | n/a -> n/a | 0 -> 0 | 0 | 2 -> 2 | no -> no | 1 -> 1 |
| ex07 | small | 6 -> 0 | 158 -> 1 | 13177 -> 296 | 2597 -> 78 | 0% -> n/a | 0 -> 0 | 0 | 7 -> 2 | yes -> no | 3 -> 1 |
| ex08 | small | 2 -> 5 | 95 -> 250 | 5348 -> 18749 | 1825 -> 4368 | 20% -> 0% | 0 -> 0 | 0 | 4 -> 10 | yes -> yes | 5 -> 5 |
| ex09 | small | 2 -> 3 | 120 -> 296 | 5649 -> 15269 | 1078 -> 4213 | 50% -> 100% | 0 -> 0 | 0 | 4 -> 10 | yes -> no | 4 -> 4 |
| ex10 | small | 2 -> 1 | 126 -> 74 | 5360 -> 2850 | 1794 -> 2176 | 25% -> 25% | 0 -> 0 | 0 | 4 -> 4 | yes -> yes | 4 -> 4 |
| ex11 | small | 0 -> 0 | 80 -> 66 | 318 -> 318 | 76 -> 79 | n/a -> n/a | 0 -> 0 | 0 | 2 -> 2 | no -> no | 1 -> 1 |
| ex12 | small | 1 -> 5 | 65 -> 242 | 3122 -> 21459 | 504 -> 3094 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 10 | yes -> no | 1 -> 3 |
| ex15 | small | 1 -> 1 | 48 -> 50 | 3095 -> 3437 | 695 -> 1555 | 25% -> 0% | 0 -> 0 | 0 | 3 -> 4 | yes -> yes | 4 -> 5 |
| ex16 | small | 1 -> 1 | 80 -> 74 | 3100 -> 3250 | 659 -> 1793 | 100% -> 40% | 0 -> 0 | 0 | 3 -> 4 | yes -> no | 1 -> 5 |
| ex18 | small | 0 -> 0 | 2 -> 11 | 310 -> 310 | 81 -> 82 | n/a -> n/a | 0 -> 0 | 0 | 2 -> 2 | no -> no | 1 -> 1 |
| ex19 | small | 1 -> 1 | 37 -> 172 | 3121 -> 3451 | 208 -> 2414 | 33% -> 0% | 0 -> 0 | 0 | 3 -> 4 | yes -> no | 3 -> 6 |
| ex20 | small | 2 -> 2 | 272 -> 135 | 6028 -> 8539 | 858 -> 1127 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 7 | yes -> yes | 1 -> 1 |
| ex21 | small | 2 -> 3 | 78 -> 308 | 6026 -> 17817 | 1171 -> 3222 | 25% -> 67% | 0 -> 0 | 0 | 4 -> 10 | yes -> yes | 4 -> 3 |
| ex22 | small | 1 -> 1 | 87 -> 60 | 3081 -> 5737 | 205 -> 1362 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 6 | yes -> no | 3 -> 3 |
| nc02 | small | 6 -> 5 | 146 -> 133 | 16932 -> 13139 | 2270 -> 1772 | 0% -> 100% | 0 -> 0 | 0 | 9 -> 10 | yes -> no | 2 -> 2 |
| nc05 | small | 6 -> 5 | 148 -> 151 | 13553 -> 15690 | 1758 -> 2408 | 14% -> 50% | 0 -> 0 | 0 | 8 -> 11 | yes -> yes | 7 -> 2 |
| nc10 | small | 6 -> 5 | 271 -> 204 | 14538 -> 14050 | 2101 -> 3401 | 33% -> 50% | 0 -> 0 | 0 | 8 -> 10 | yes -> yes | 3 -> 6 |
