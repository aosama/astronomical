# astronomical-quantization

Sensitivity-driven mixed-precision affine quantization for MLX artifacts: a Rust port of the quantization core of [OptiQ](https://mlx-optiq.com) (the `mlx-optiq` Python package, MIT License, Copyright (c) 2026 Thin Signal). The full attribution, including the oMLX lineage of the range-search encoder, is in `third-party/THIRD_PARTY_NOTICES`; the vendored calibration mix carries its license beside it in `calibration/LICENSE-OPTIQ`.

This document is the design record for the port. The crate currently holds the typed design, method stubs, and the complete test plan; every algorithm body is a `todo!()` whose message and preceding comment describe the work to port, and every test is `#[ignore]`d naming the pending function. Implementation proceeds module by module by removing the ignores.

## Why port rather than call Python

- Speed where it matters. The shard-wise requantizer and the per-group range search are bandwidth-bound and embarrassingly parallel; the Python tool runs them one tensor at a time with interpreter dispatch between every MLX call. The Rust driver can overlap the next tensor's reads with the current tensor's encode and write shards from another thread while keeping the same single-tensor peak memory.
- Control. Astronomical wants to modify the allocation policy and the encoder. The pure allocator is pinned to the Python reference by parity fixtures, so a modification is a visible, intended diff rather than drift.
- Calibration on the real graph. The exact sensitivity sweep runs through Astronomical's own Qwen3.5 engine, so the bit map is calibrated on the kernels the artifact will actually run through, not on an mlx-lm re-implementation.

## Pipeline and module map

| Stage | Module | OptiQ source | Needs MLX |
| --- | --- | --- | --- |
| List quantizable modules from safetensors headers | `module_inventory` | `sensitivity._index_bf16_layers`, `_quantizable_linears` | no |
| Classify modules by name | `module_taxonomy` | `optimizer._identify_*`, `shardwise.is_expert`, `is_router` | no |
| Build calibration sequences | `calibration` | `calibration/datasets.py` | no |
| Measure per-module KL sensitivity | `sensitivity::exact_sweep` through `SensitivityProbeModel` | `sensitivity.analyze_sensitivity_exact` | engine-side only |
| Score structurally without measurement | `sensitivity::static_scorer` | `sensitivity.analyze_sensitivity_static` | no |
| Allocate bits under a target | `allocation` | `optimizer.optimize_mixed_precision` | no |
| Encode with the range search | `encoder` | `lowbit.quantize_best` | `direct-mlx` (pure reference beside it) |
| Requantize tensor by tensor | `requantize` | `shardwise.requantize_shardwise` | `direct-mlx` |
| Write the artifact documents | `artifact` | `artifacts.py`, `mlx_backend` metadata | no |
| Roofline throughput model | `latency` | `latency.py` | no |
| Key-value cache allocation | `kv_cache` | `kv_cache.py` | engine-side only |
| Orchestrate | `pipeline` | `models/llm.run_llm_pipeline` | `direct-mlx` for the full path |
| Progress and ETA | `progress` | the per-layer ETA prints | no |

## Dependency direction

The crate is a leaf. Its `direct-mlx` feature pulls in the MLX bindings and the runtime integration for the encoder and the requantizer, and nothing else. The exact sweep needs a forward pass, which the crate obtains through the `SensitivityProbeModel` trait; model-serving implements that trait for the resident Qwen3.5 model in `crates/model-serving/src/quantization_probe/` behind its own `direct-mlx` feature, because the swap needs access to typed weights that are private to that crate. So model-serving depends on this crate's pure surface, and this crate never learns about an engine.

The CLI's `astronomical quantize` verb has three sub-verbs. `plan` (allocate from an existing `optiq/sensitivity.json`) and `inspect` (read an artifact's provenance) are pure and compiled into every CLI build. `convert` runs the sweep and the requantizer in-process and is compiled in only with the CLI's `quantize-convert` feature, so the default CLI binary stays free of MLX linkage; without the feature the verb names the feature and exits with failure rather than being silently absent.

## Output format

The output is OptiQ's artifact layout, which three independent readers accept: stock mlx-lm (`config.json`'s `quantization` document with global defaults plus per-module `{bits, group_size, mode}` entries), OptiQ's own tools (`optiq/metadata.json` with the `per_layer` bit map and `optiq/sensitivity.json` with the measured sweep), and Astronomical's model-serving crate, whose strict `OptiQQuantizationConfig` and `OptiQMetadata` readers validate the two documents against each other at load. The model-serving integration test `artifacts::optiq_artifact_round_trip` feeds this crate's writers straight into those readers.

Two rules from the field are encoded, not left to discipline: the metadata describes the weights that were written (a router passed through, a module dropped to group size 32), never the plan; and the sensitivity document ships with every quant so a re-target is a knapsack re-run, not a repeated ten-hour sweep.

## Calibration

The bundled mix is OptiQ's `optiq.jsonl`: 40 samples across six domains (prose, thinking traces, code reasoning, agent loops, tool calling, constraint-bearing instructions). Chat-style samples are rendered by role-prefixed concatenation, OptiQ's own fallback, because the Rust `tokenizers` crate has no chat-template engine; a caller who needs template-exact sequences supplies pre-tokenized samples through `CalibrationMix::from_pretokenized`. Tests use the Romeo and Juliet fixture exclusively.

## Progress reporting

Every long stage reports percentage completion and an estimated time to completion through `ProgressReporter` at a fixed five-second cadence, including the first unit and the final unit. A resumed sweep estimates from the units completed in the current run only. The CLI routes reports to standard error so standard output stays composable.

## Test plan

Test names start with `should_`; unit tests live beside their source as `*_tests.rs` modules; integration journeys live under `tests/integration/`; GPU contracts under `tests/direct_mlx/`; real-model journeys in the root acceptance package.

Unit tests (`cargo test -p astronomical-quantization --lib`) cover the taxonomy on names that bit real conversions, every allocator rule in isolation (efficiency ordering, the target as a floor, protected boosting, NaN handling, the uniform-reference sign detection, the attention slot reservation, the two-linear block cap, the run guard stride), the sweep driver's probe sequencing and the measured-majority guard against a scripted probe, checkpoint monotonicity and the Python checkpoint shape, calibration rendering and deterministic chunking, encoder packing and the never-worse-than-stock property, shard naming and the index, the artifact documents, the roofline arithmetic, the key-value cache allocator, and the progress tracker's cadence and ETA.

Integration journeys (`cargo test -p astronomical-quantization --test integration_tests`) pin the port to the Python reference through fixtures generated by `tests/fixtures/generate_optiq_parity_fixtures.py` with mlx-optiq 0.6.5: the allocator must reproduce five Python allocations bit for bit on a synthetic table that exercises every rule; the taxonomy must agree with OptiQ's four predicates on every fixture name; structural priorities must match to float rounding; the pure reference encoder must reproduce the Python range search's packed codes exactly at 2 and 3 bits and stock min/max at 4 and 8; the KL formula must match on a logits pair. The journeys also load the bundled mix end to end, inventory real safetensors files (2-D linears, fused gate-up splits, unfused expert stacking), build requantization plans with byte estimates, and round-trip the artifact documents through disk.

Direct-MLX contracts (`cargo test-quantization-direct-mlx`, in parallel) hold the Metal encoder to the pure reference (identical packed codes, scales within float rounding), require stock layout above 4 bits, require squared error never above stock min/max, require that MLX's own dequantize reads the GPU artifact back to the reference's values, and run the requantizer end to end on a synthetic two-shard source, asserting the written map, the index, and that peak MLX memory does not scale with tensor count.

Acceptance journeys (`cargo test-quantization-acceptance`, ignored, environment-discovered, serial) run lowest tier first: requantizer throughput with a warmup tensor; a static conversion validated and served back through the Qwen3.5 engine with a Romeo and Juliet continuation at a 2,048-token prefill chunk; the exact sweep with a simulated interruption and checkpoint resume; and tensor-by-tensor parity with an artifact the Python tool produced from the same source (identical packed words at 4 bits and above, no worse squared error below, identical bit maps from the Python sweep).

CLI journeys (`cargo test -p astronomical-cli --test integration_tests quantize`) cover argument parsing including the comma-separated ladder and its validation, `plan` against a document written to a temporary directory, `inspect` on a directory without metadata, and the `convert` refusal in a default build.

## Implementation order

1. `module_taxonomy`, `bit_width`, `progress`: small, pure, and everything else uses them.
2. `allocation`: drive it green against `tests/integration/allocation_parity.rs`; this is the first point where the port can be used, through `astronomical quantize plan` on a Python-produced `optiq/sensitivity.json`.
3. `artifact`, then the model-serving round-trip test.
4. `encoder::reference`, driven by `tests/integration/encoder_reference_parity.rs`; then `encoder::mlx` against the direct-MLX contracts.
5. `module_inventory` and `requantize`, driven by the integration journeys and then the direct-MLX requantizer contract. At this point `convert --method static` works end to end.
6. `calibration`, `sensitivity::exact_sweep`, and the Qwen3.5 probe in model-serving, driven by the scripted-probe unit tests and then the acceptance sweep journey.
7. `latency` and `kv_cache`, which have no consumer on the critical path.

## Open questions recorded for the implementer

- `forward_chunk` on the resident model returns the final chunk's logits; the probe needs logits at every position of the sample for KL. Confirm the returned shape and widen the path if it returns only the last row.
- The calibration shuffle uses a documented deterministic generator over sequence offsets; it does not reproduce numpy's `RandomState(42).shuffle`, so a Rust sweep and a Python sweep of the same mix see the same sequences in a different order and a different subset. Parity journeys therefore compare allocations from the same sensitivity document, not sweeps.
- The CLI's `quantize-convert` feature links the model-serving engine into the CLI binary. If that proves too heavy for the release bundle, the `convert` sub-verb can move to a separate `astronomical-quantize` binary under `apps/` with no change to this crate.
