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

## What programs cost in memory

Sampled every 50 ms with nothing else running on the Neural Engine. "Alone" cycles the programs with the GPU idle; "both minus GPU" compares peak wired memory across prefill with the programs running and paused. Both read system wired memory, because this memory appears in neither MLX's accounting nor the process footprint.

| Model | Programs' weights | Alone | Both minus GPU | Wired at load | Surfaces | Whole cost | Against weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 0.56 GiB | 1.52 GiB | 1.31 GiB | 1.52 GiB | 0.30 GiB | 1.82 GiB | 3.3× |
| 2B | 1.83 GiB | 3.13 GiB | 3.04 GiB | 3.11 GiB | 0.46 GiB | 3.57 GiB | 2.0× |
| 9B | 9.75 GiB | 13.83 GiB | 13.63 GiB | 13.65 GiB | 1.22 GiB | 14.87 GiB | 1.5× |

- **No ratio fits.** The cost against the weights changes with the model, which is why serving measures it rather than assuming one. A fixed 1.8× reserved too much for the 9B and about half of what the 0.8B needs.
- **Surfaces are wired at load, before any reading starts,** and appear in both arms of the comparison, so neither reading sees them. They are computed from each program's shape instead: 1,257 MiB of wired growth was read across a 9B load, against 1,249 MiB computed.
- **The driver wires with a delay.** Wired memory holds still for up to a second of running programs, then rises in steps and settles within about 2 seconds.
- **MLX's own wired memory falls after its last work,** by 2 GB or more. Measured sooner than 30 seconds after the GPU's last use, this hid most of a small model's programs: 0.37× and 0.59× of the true cost on the 0.8B and 2B.
- **The whole cost held up through the app.** On the 9B, measured at 15.02 GiB, wired memory grew 15.0 to 15.2 GiB under a 10,000-token prompt. A ceiling of 21 GB moved a long prompt to the GPU part way through with no further rise in wired memory, so the programs' 10 to 20 seconds of release did not carry it past the ceiling.
- **A reading taken after a run says nothing.** After a 10-second decode the programs had already been released, so a reading then shows only MLX's memory.

Prefill speed in the same runs, direct model forwards with the programs running against paused: 0.89× on the 0.8B, 1.12× on the 2B and 1.29× on the 9B.

## What a program's first load costs

macOS compiles a program into the chip's format the first time it is loaded from a path, and keeps the result in a system store that applications cannot read. Later loads reuse it. When the store has lost a program, the cost is paid again while the kept `.mlmodelc` is reused. Issue #2064 holds the full record.

One MLP program of the 9B (about 151 million weights, 960 tokens), three fresh copies of each form, loaded through Core ML. Chip compilations were counted in the system log (`ANECompilerService`, "Start of compilation of network"); the store's growth is the volume's used space before and after, with nothing added to any application-visible cache.

| | fp16 | 8-bit lookup tables |
| --- | ---: | ---: |
| First load | 1.14 to 1.16 s | 0.91 to 0.95 s |
| Chip compilation inside it | 0.76 s | 0.74 s |
| Added to the system store | 288 MB | 146 MB |
| Later loads | 0.04 s | 0.04 s |
| One 960-token prediction | 16.8 ms | 16.5 ms |

- **The compilation does not follow the stored bytes.** Halving the weight bytes halves the file and the system's copy, and shortens the first load by about a fifth.
- **A whole model pays it per program.** The 9B's 32 MLP programs took 32 s when all were compiled again, against about 2 s with the store intact.
- **A macOS update empties the store.** So does nothing an application binary does: a rebuilt binary with a new code hash and the same name loaded a kept program with no compilation, while a renamed binary compiled it again. Some programs were also compiled again between application updates on one macOS version, for a reason not yet found.
- **Read-only program files change nothing:** 1.04 to 1.18 s against 0.98 to 1.10 s.
- **The lookup-table builder is for ternary weights.** On one part of a real 9B MLP layer it moved the weights by 37 percent (relative RMS) at 16 rows per table, 32 percent at 4 and 1.4 percent at 1, and one row per table runs about 14 times slower. A dense 8-bit form needs its own quantizer before these byte savings can be used (issue #1878).

Core AI (macOS 27) can compile a model ahead of time for a named chip. The same layer, converted with `coreai-torch` and compiled with `coreai-build compile --preferred-compute neural-engine`:

| First load | Ahead-of-time bundle | Source model | Core ML |
| --- | ---: | ---: | ---: |
| One layer | 0.90 to 0.97 s | 1.65 to 1.9 s | 1.09 s |
| Four layers as functions of one model | 6.8 to 7.2 s | 6.0 to 6.1 s | about 4.4 s as four programs |

- **The on-device compilation shrinks, the load does not.** It fell from about 1.25 s to 0.14 s a layer, but the system still copies the bundle's weights into its store on first load.
- **Bundles must not be loaded one after another.** The large-model compile service exits after each compilation and is restarted no sooner than ten seconds later, so separate bundles took about 11 s each. Layers have to be functions of one model.
- **A bundle is tied to the runtime that reads it.** One built by the compiler from Xcode 27.2 did not load on macOS 27.0.1.

`cargo test-neural-engine-journeys --filter dense_weight_encoding` repeats the fp16 and lookup-table comparison on fixture weights at the same shape, and the table builder's error on dense weights.

## How much of a chunk to give the Neural Engine

The same loaded model at each share of a 2,048-token chunk, MLP programs and output gates running, against the GPU alone. One 0.8B row and one 9B row were disturbed by other work on the Mac and are left out.

| Tokens | 0.8B | 2B | 9B |
| ---: | ---: | ---: | ---: |
| 128 | 0.80× | | |
| 256 | 0.83× | | |
| 448 | 0.71× | 1.02× | |
| 704 | 0.95× | 1.12× | 1.18× |
| 960 | | 1.11× | about 1.3× |
| 1,216 | 0.80× | 0.98× | Core ML refused it |

- **The 47 percent share is right where a gain exists.** 960 tokens is at or near the best on both models that gain.
- **No share helps a 0.8B model.** A chunk split costs a hand-off and a forced GPU evaluation per layer, which outweighs the little work its small MLP moves. Serving now finds this out on each Mac and turns the programs off.
- **Timing a model at load did not work reliably.** A check that timed chunks of made-up tokens against the GPU read 1.31 on the 9B (1.20× through the app) and 0.85 on the 0.8B (0.89× direct), but 1.06, 0.68, 1.07 and 0.95 on repeats of the 2B, which gains 1.09× to 1.12×. So it was taken out in favour of comparing real chunks during real prompts, below.
- **The output gates added nothing measurable on the 2B, and at 704 tokens on the 9B,** and cost the 0.8B a further 3 to 15 percent. A later run at 960 tokens on the 9B, at the end of this file, found no gain there either.

## Through the app

The kept results, wired into the serving path behind `experimental_neural_engine_prefill_enabled`: the MLP split, the `z` rows, and the program cache. Ornith 9B in the Development app, a 34 GB ceiling, the prompt cache on, about 10,300-token prompts each with a unique first line, median of three requests per run.

| Run | Prefill | Against off |
| --- | ---: | ---: |
| Off | 1,207 TPS | n/a |
| On, after a first load of 108 s | 1,444 TPS | 1.20× |
| On, after a reload of 3.9 s | 1,362 TPS | 1.13× |

Direct model runs of the same programs measured 1.27× to 1.28×. The first long prompt after a load was the slowest of each run (1,296 and 1,362 TPS). Off ran between the two on runs, so heat does not explain the gap between them, and it remains unexplained with three requests per run.

## Comparing neighbouring chunks of real prompts

`should_compare_real_chunks_on_*` runs one prompt's chunks in turn on the GPU alone and on both engines, 12 chunks of 2,048 tokens per prompt. Each chunk is compared with the mean of its two neighbours, which cancels the steady growth in chunk time as the prompt's position advances. Figures are the median gain of one prompt's ten comparisons.

| Prompt | 0.8B | 2B | 9B |
| --- | ---: | ---: | ---: |
| Right after the programs load | 0.83 | 0.92 | 1.17 |
| Warm, starting on the GPU | 0.92, 0.92 | 1.03, 1.08 | 1.18, 1.19 |
| Warm, starting on both engines | 0.91, 0.91, 0.92 | 1.07, 1.06, 1.09 | 1.18, 1.19, 1.19 |
| After 25 s idle, starting on the GPU | 0.90 | 0.94 | 1.12 |

- **Real chunks separate the models where made-up ones did not.** Warm prompts vary by about 3 percent on one model, against about 40 percent for the load-time check.
- **Cold programs read low for several seconds,** enough to look like a 0.8B model on a 2B model, so one prompt never decides. Serving wants 15 comparisons across at least 3 runs of chunks.
- **A cold first chunk is the costly one.** Against a warm chunk of the same position, a cold chunk on both engines took about 380, 490 and 600 ms longer on the 0.8B, 2B and 9B when it was the prompt's first, and about 140, 180 and 170 ms longer when one chunk on the GPU came before it. So a prompt after 10 seconds of idle starts on the GPU.

Through the app with the rule in place, 20,000-token Shakespeare prompts, the median of three prompts per arm against the Neural Engine off:

| Model | The rule's reading | In quick succession | After 25 s idle |
| --- | ---: | ---: | ---: |
| 0.8B | 0.98, then 0.90 on a second comparison: turned off | 0.97× once turned off, the noise of this measurement | |
| 2B | 1.03 to 1.11, across four comparisons | 1.11× | 0.94× |
| 9B | 1.28 | 1.24× | 1.11× |

- **Waking cold programs still costs a small model its gain.** Leading with the GPU made the stall smaller, not zero. Waking the programs during that first GPU chunk is the untested next idea.
- **A 0.97 cutoff kept the 0.8B on.** It read 0.98 and really ran 0.98×, and one prompt after 25 s idle ran 0.62×, so the line moved to 1.02.

The output gates at the production share on the 9B, twice on one loaded model: the MLP programs alone ran 1.11× and 1.16× the GPU alone, and with the gates 1.13× and 1.13×. With the 2B and the 704-token 9B rows above, no model shows a gain from them.

The Overview's time saved, checked against one warm 10,200-token prompt on the 9B: the panel credited 1.4 to 1.5 s. That prompt ran at 1,355 to 1,433 tokens per second against 1,151 to 1,319 with the Neural Engine off, which works out to between 0.2 and 1.7 s. The off runs were earlier the same day, so this is a sanity check and not a measurement of the estimate's accuracy.
