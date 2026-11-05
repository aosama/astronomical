# Neural Engine experiments

Ideas drawn from reading oMLX's Neural Engine work (Apache 2.0, read only, nothing copied), tried one at a time on our own hardware, for issue #161.

## Rules for every A/B

- **Measurement:** 10,240 Shakespeare tokens of input and 500 output tokens, after a warm-up of 500 input and 50 output tokens.
- **Conditions:** optimized build, no SSD cache, and no model loaded in Stable or Development during the run.
- **Bias control:** A and B alternate on one loaded model for three rounds, and results are medians.
- **Lowest level first:** a single layer on real weights, then the whole model.
- **Quality:** logit cosine and greedy match are recorded, but neither proves quality. Anything touching recurrent state also needs discriminating tasks.
- **Constraints:** public Apple APIs only, and a failed experiment is still reported.
- **Hardware caveat:** oMLX's figures are mostly from an M3 Ultra with two Neural Engines. Ours is one M5 Pro with one Neural Engine, so only our numbers count here.

## Baseline

A is the GPU alone. B is today's opt-in path: whole MLP programs on 47% of the tokens.

## Experiments

| # | Idea | Question | Measure | Keep it if |
| --- | --- | --- | --- | --- |
| 1 | Gate and up only, split by channel | Does leaving `down` and the rotation on the GPU cut memory enough to pay for less offload? | Layer time, cosine, memory, then prefill and decode | Prefill gain close to the token split at far less memory |
| 2 | Wired memory after idle | Does the driver give the memory back after prefill? | Wired memory over time after the last prediction | Memory returns within seconds |
| 3 | Persistent compile cache | How much of a program's load is compilation? | Cold and warm load time, then repeated full load | Warm load well under cold load |
| 4 | Multifunction program | Can one public Core ML package hold many layers? | Compile, load, cycling time, memory, placement | Faster load, no speed loss, whole program on the Neural Engine |
| 5 | DeltaNet `z` rows only | How much do token-local rows add, leaving recurrent `qkv` on the GPU? | Projection time, prefill, cosine, discriminating tasks | A real gain and tasks unchanged |
| 6 | Tune the split per machine | Is 47% right for this Mac? | Prefill across split widths | A clear win over 47% |
| 7 | Padded tails | Is a padded tail faster than the GPU past the crossover? | Tail time, prefill on a matching prompt | A real gain above the crossover |
| 8 | CPU as a third engine | Does a CPU slice help on an M5 Pro? | Layer time, prefill | A gain beyond noise after tuning |
| 9 | 8-bit activations on GPU tensor units | Does this beat the Neural Engine on M5? | Prefill, decode, cosine | Compared directly with the best Neural Engine result |

Idea 9 is the one with no Neural Engine, and it can't run alongside the MLP split. It is last because comparing against it needs the best Neural Engine number first.

## Results

| # | Experiment | A | B | Prefill speedup | Decode change | Memory | Cosine | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | Baseline: whole MLP by token, 47% (9B) | 1,288 TPS | 1,597 TPS | 1.24× | none (about 1.00×) | About 11 GiB growth, 102 s load | 0.99963 | Current path |
| 1 | Gate and up by channel, 47% (9B) | 1,293 TPS | 1,387 TPS | 1.07× | none (0.993×) | 28 s load on the 9B; 18.4 GiB against 25 GiB on the 27B | 0.99889 | Rejected: less speed, a small memory saving, below the 0.999 cosine gate |
| 2 | Wired memory after idle (16 programs, 27B) | 6.0 GiB wired while working | About 0 GiB after 20 s idle | First pass after idle 1.30×, then normal | None; decode never uses the programs | Released within 10-20 s, retaken on resume | n/a | Kept: memory exists only during prefill, so admission can treat it as temporary |
| 3 | Compile cache at a stable path (9B) | 86 s cold load | 1.8-2.0 s in a new process | 1.25×, then 1.18× on a hot Mac, then 1.24× after idling, against 1.24× baseline | none (0.999×, 0.993×, 0.991×) | 9 GiB of disk | 0.99963 (identical) | Kept: about 45× faster load at the same speed. The cost is macOS's own compile, cached by path, so the first load must happen at the final path |
| 4 | One multifunction package, 4 functions against 4 separate programs (27B) | 40 s load, 0.55 GiB per layer, 30.1 ms | 41 s load, 0.67 GiB per layer, 30.1 ms | Not run on the whole model: same 30.1 ms per program, so no gain is possible | none (decode never uses the programs) | No saving, and slightly worse | n/a | Rejected. macOS still compiles each function separately. Above about 1-2 GiB of weights the whole package silently moves to the CPU (8 functions, 2.07 GiB: 153 ms, 68 operations off the Neural Engine) |
| 5 | DeltaNet `z` rows, 24 programs (9B) | 1,306 TPS GPU; 1,284 TPS GPU in the combined run | 1,360 TPS alone; 1,648 TPS with the MLP split | 1.04× alone; 1.28× with the MLP split, against 1.24× without | none (1.000×, 0.999×) | 8 s load for all 24 | 0.99965 alone, 0.99952 combined | Kept as an add-on. 6 of 6 answer tasks correct on every arm. On one code task the MLP arms ignored "output only the code", and `z` alone added only a code fence and a different return expression |
| 6 | Tune the split: 896 against 960 Neural Engine tokens (9B, prefill only, one loaded model) | 1.271× at 960 (GPU 1,323 TPS, both 1,642 TPS) | 1.271× at 896 (GPU 1,275 TPS, both 1,567 TPS) | No change: 1.271× for both | not measured; the split cannot affect decode | not measured | n/a | Rejected: keep 960. One layer on synthetic weights favoured 896 (17.9 ms against 20.5 ms), and the real model did not |

Ideas 7 (padded tails), 8 (the CPU as a third engine) and 9 (8-bit activations on GPU tensor units) have not been run.

## Reference point: Apple's on-device model

Apple's own 3B dense model runs on the Neural Engine alone, so it shows what that route reaches on the same M5 Pro. Timed through the `fm` command with the Shakespeare fixture, greedy, warm, each prompt on fresh text. The times include the command's start-up and Apple's safety models, so the model itself is somewhat faster than this.

| Measure | Apple 3B, Neural Engine alone | Ornith 9B, GPU alone | Ornith 9B, GPU and Neural Engine |
| --- | ---: | ---: | ---: |
| Prefill at about 2,100 tokens | about 2,200 TPS | not measured | not measured |
| Prefill at about 8,000 tokens | about 1,350 TPS | not measured | not measured |
| Prefill at 10,240 tokens | over its limit | 1,288 TPS | 1,648 TPS |
| Decode | 49 TPS | 45 TPS | 45 TPS |
| Context limit | about 8,100 prompt tokens | 10,240 tested | 10,240 tested |

- **Per parameter:** the 9B does about three times the work per token, so the GPU path is roughly three times faster per parameter than Apple's Neural Engine path, for both prefill and decode. This is a rough comparison across two architectures.
- **What Apple has that a conversion lacks:** the model was trained for 2-bit weights, it ships precompiled in a private format, and its shapes and context are fixed.

## Through the app

The kept results, wired into the serving path behind `experimental_neural_engine_prefill_enabled`: the MLP split, the `z` rows, and the program cache. Ornith 9B in the Development app, a 34 GB ceiling, the prompt cache on, about 10,300-token prompts each with a unique first line, median of three requests per run.

| Run | Prefill | Against off |
| --- | ---: | ---: |
| Off | 1,207 TPS | n/a |
| On, after a first load of 108 s | 1,444 TPS | 1.20× |
| On, after a reload of 3.9 s | 1,362 TPS | 1.13× |

Direct model runs of the same programs measured 1.27× to 1.28×. The first long prompt after a load was the slowest of each run (1,296 and 1,362 TPS). Off ran between the two on runs, so heat does not explain the gap between them, and it remains unexplained with three requests per run.
