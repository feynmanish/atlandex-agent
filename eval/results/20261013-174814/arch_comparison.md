# Before / after: `20261013-174814:single` (baseline) vs `20261013-174814:multi` (fixed version)

Paired query-runs only (completed in both). Latency is wall-clock seconds per query-run; tokens and LLM calls are the agent's own (the judge is not included). Hallucination = judged sentences marked unsupported; the baseline uses the re-judged verdicts when available, so both sides come from the same judge code.

## Summary

| agent model | paired | | steps | LLM calls mean / max | latency p50 / p95 (s) | tokens in / out | hallucination (judged sent.) | unjudged sent. | uncited flagged / dropped |
|---|---|---|---|---|---|---|---|---|---|
| large | 29 of 29/29 | before | 2.2 | 4.3 / 9 | 64.3 / 138.4 | 6161 / 1982 | 21% (52) | 0 | 2 / 0 |
| large | 29 of 29/29 | after | 2.4 | 7.4 / 13 | 88.4 / 199.1 | 9296 / 3466 | 10% (50) | 0 | 1 / 1 |
| large|existing | 22 of 29/29 | before | 1.1 | 3.2 / 5 | 42.9 / 84.8 | 3840 / 1416 | 13% (30) | 0 | 2 / 0 |
| large|existing | 22 of 29/29 | after | 1.8 | 6.5 / 10 | 80.5 / 194.8 | 7824 / 2730 | 10% (30) | 0 | 0 / 0 |
| large|new_candidate | 7 of 29/29 | before | 5.6 | 7.9 / 9 | 127.0 / 159.7 | 13455 / 3762 | 32% (22) | 0 | 0 / 0 |
| large|new_candidate | 7 of 29/29 | after | 4.4 | 10.6 / 13 | 127.6 / 235.8 | 13922 / 5782 | 10% (20) | 0 | 1 / 1 |
| all | 29 of 29/29 | before | 2.2 | 4.3 / 9 | 64.3 / 138.4 | 6161 / 1982 | 21% (52) | 0 | 2 / 0 |
| all | 29 of 29/29 | after | 2.4 | 7.4 / 13 | 88.4 / 199.1 | 9296 / 3466 | 10% (50) | 0 | 1 / 1 |

(*paired* = query-runs completed in both runs, then completed in baseline / completed in fixed run.)

## Quality and guards

Segment hit (±60 s) and video hit over the paired queries that have gold. *Discordant* = queries where only one side hit; *p* = exact two-sided sign test on them. *Hallucination diff* = after minus before, with a 95% bootstrap interval over queries (not sentences). The guards show whether a lower hallucination rate came from deleting content: final sentences per answer, no-coverage answers, sentences the verifier dropped.

| agent model | | segment hit | video hit | discordant before-only / after-only (p) | hallucination diff [95% CI] | final sent. / answer | no-coverage | verify dropped | supervisor fallbacks | budget-forced runs | route disagreements |
|---|---|---|---|---|---|---|---|---|---|---|---|
| large | before | 76% (n=29) | 79% | 1 / 2 (p=1.000) | -11 pts [-28, +4] | 2.3 | 0 | 0 | 0 | 0 | 2 (ex06, ex18) |
| large | after | 79% (n=29) | 83% |  |  | 2.3 | 1 | 1 | 0 | 7 |  |
| large|existing | before | 82% (n=22) | 82% | 1 / 2 (p=1.000) | -3 pts [-20, +11] | 2.1 | 0 | 0 | 0 | 0 | 2 (ex06, ex18) |
| large|existing | after | 86% (n=22) | 91% |  |  | 2.1 | 0 | 0 | 0 | 1 |  |
| large|new_candidate | before | 57% (n=7) | 71% | 0 / 0 (p=1.000) | -22 pts [-55, +11] | 3.1 | 0 | 0 | 0 | 0 | 0 |
| large|new_candidate | after | 57% (n=7) | 57% |  |  | 3.0 | 1 | 1 | 0 | 6 |  |
| all | before | 76% (n=29) | 79% | 1 / 2 (p=1.000) | -11 pts [-28, +4] | 2.3 | 0 | 0 | 0 | 0 | 2 (ex06, ex18) |
| all | after | 79% (n=29) | 83% |  |  | 2.3 | 1 | 1 | 0 | 7 |  |

## Per query

| query | model | steps | latency (s) | tokens in | tokens out | hallucination (judged) | unjudged | uncited dropped | LLM calls | segment hit | final sent. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ex01 | large | 1 -> 1 | 64 -> 57 | 4050 -> 2732 | 3219 -> 2664 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 4 | yes -> yes | 1 -> 3 |
| ex02 | large | 1 -> 1 | 30 -> 64 | 3140 -> 3756 | 1152 -> 2384 | 100% -> 50% | 0 -> 0 | 0 | 3 -> 6 | yes -> yes | 1 -> 2 |
| ex03 | large | 2 -> 4 | 81 -> 196 | 6641 -> 17201 | 2652 -> 3660 | 100% -> 0% | 0 -> 0 | 0 | 4 -> 10 | yes -> yes | 1 -> 1 |
| ex04 | large | 1 -> 1 | 43 -> 55 | 3147 -> 2574 | 1111 -> 1485 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 4 | yes -> yes | 1 -> 2 |
| ex05 | large | 1 -> 3 | 40 -> 172 | 3140 -> 17140 | 1669 -> 4675 | 33% -> 0% | 0 -> 0 | 0 | 3 -> 10 | yes -> yes | 3 -> 2 |
| ex06 | large | 0 -> 4 | 19 -> 201 | 305 -> 15862 | 730 -> 5381 | n/a -> 0% | 0 -> 0 | 0 | 2 -> 10 | no -> yes | 1 -> 2 |
| ex07 | large | 3 -> 3 | 93 -> 135 | 9105 -> 13324 | 2113 -> 3023 | 0% -> 0% | 0 -> 0 | 0 | 5 -> 8 | yes -> yes | 1 -> 1 |
| ex08 | large | 1 -> 2 | 43 -> 82 | 3116 -> 9529 | 1190 -> 2483 | 0% -> 100% | 0 -> 0 | 0 | 3 -> 7 | yes -> yes | 1 -> 1 |
| ex09 | large | 2 -> 2 | 85 -> 128 | 6219 -> 11805 | 1530 -> 4116 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 9 | yes -> no | 1 -> 1 |
| ex10 | large | 1 -> 1 | 71 -> 88 | 4009 -> 5757 | 3515 -> 3949 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 6 | yes -> yes | 5 -> 3 |
| ex11 | large | 1 -> 1 | 36 -> 55 | 3297 -> 4259 | 1127 -> 2223 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 6 | yes -> yes | 3 -> 1 |
| ex12 | large | 0 -> 0 | 12 -> 14 | 323 -> 323 | 1012 -> 1027 | n/a -> n/a | 0 -> 0 | 0 | 2 -> 2 | no -> no | 11 -> 11 |
| ex13 | large | 0 -> 0 | 7 -> 11 | 325 -> 325 | 698 -> 1013 | n/a -> n/a | 0 -> 0 | 0 | 2 -> 2 | no -> no | 2 -> 6 |
| ex14 | large | 2 -> 2 | 76 -> 102 | 6857 -> 11144 | 1811 -> 3293 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 7 | yes -> yes | 1 -> 1 |
| ex15 | large | 1 -> 1 | 51 -> 48 | 3372 -> 2859 | 1505 -> 2323 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 4 | yes -> yes | 3 -> 2 |
| ex16 | large | 1 -> 1 | 29 -> 53 | 3344 -> 2610 | 854 -> 1371 | 100% -> 0% | 0 -> 0 | 0 | 3 -> 4 | yes -> yes | 1 -> 1 |
| ex17 | large | 1 -> 2 | 43 -> 127 | 3174 -> 9944 | 998 -> 3193 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 7 | yes -> yes | 1 -> 2 |
| ex18 | large | 0 -> 1 | 8 -> 55 | 315 -> 2578 | 612 -> 936 | n/a -> 0% | 0 -> 0 | 0 | 2 -> 4 | no -> yes | 2 -> 1 |
| ex19 | large | 1 -> 1 | 36 -> 61 | 3362 -> 4162 | 693 -> 2388 | 0% -> 100% | 0 -> 0 | 0 | 3 -> 6 | yes -> yes | 1 -> 1 |
| ex20 | large | 2 -> 2 | 68 -> 79 | 6792 -> 11359 | 919 -> 2945 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 9 | yes -> yes | 1 -> 1 |
| ex21 | large | 2 -> 2 | 66 -> 93 | 7104 -> 9770 | 1154 -> 2092 | 0% -> 0% | 0 -> 0 | 0 | 4 -> 7 | yes -> yes | 2 -> 1 |
| ex22 | large | 1 -> 5 | 36 -> 158 | 3338 -> 13107 | 884 -> 3425 | 0% -> 0% | 0 -> 0 | 0 | 3 -> 10 | yes -> yes | 2 -> 1 |
| nc02 | large | 6 -> 5 | 127 -> 124 | 15310 -> 11609 | 3166 -> 4407 | 25% -> 0% | 0 -> 0 | 0 | 8 -> 10 | yes -> yes | 4 -> 4 |
| nc03 | large | 6 -> 5 | 140 -> 128 | 11557 -> 15022 | 3623 -> 5235 | 0% -> 0% | 0 -> 0 | 0 | 8 -> 10 | yes -> yes | 2 -> 3 |
| nc04 | large | 6 -> 5 | 168 -> 152 | 14797 -> 15819 | 3814 -> 5562 | 50% -> n/a | 0 -> 0 | 1 | 8 -> 10 | no -> no | 2 -> 1 |
| nc05 | large | 6 -> 5 | 113 -> 257 | 15397 -> 16708 | 5424 -> 11098 | 20% -> 0% | 0 -> 0 | 0 | 9 -> 13 | no -> no | 5 -> 8 |
| nc06 | large | 6 -> 4 | 76 -> 61 | 12395 -> 10616 | 2898 -> 3853 | 100% -> 100% | 0 -> 0 | 0 | 8 -> 9 | no -> no | 1 -> 1 |
| nc09 | large | 6 -> 4 | 136 -> 67 | 14903 -> 13124 | 4008 -> 2953 | 0% -> 50% | 0 -> 0 | 0 | 8 -> 10 | yes -> yes | 5 -> 2 |
| nc10 | large | 3 -> 3 | 79 -> 186 | 9828 -> 14553 | 3402 -> 7366 | 100% -> 0% | 0 -> 0 | 0 | 6 -> 12 | yes -> yes | 3 -> 2 |
