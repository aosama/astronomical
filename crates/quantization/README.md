# astronomical-quantization

Sensitivity-driven mixed-precision affine quantization for MLX artifacts: a Rust port of the quantization core of [OptiQ](https://mlx-optiq.com) (the `mlx-optiq` Python package, MIT License, Copyright (c) 2026 Thin Signal). The full attribution, including the oMLX lineage of the range-search encoder, is in `third-party/THIRD_PARTY_NOTICES`; the vendored calibration mix carries its license beside it in `calibration/LICENSE-OPTIQ`.

This document is the design record for the port. The crate holds the typed design and the complete test plan. The taxonomy, bit widths, progress reporting, the allocator, the module inventory, the structural scorer, the KL formula, the pure reference encoder, the artifact writers, the Metal range-search encoder, the requantizer (plan, bit policy, shard writer, and the MLX driver with its source reader for fused, stacked, FP8, and packed layouts), calibration loading, the sweep driver with its checkpoint and report, the pipeline orchestration, the roofline latency model, and the key-value cache sweep and allocator are ported and tested. No algorithm body in the crate is a stub; what remains engine-side is the Qwen3.5 probe in model-serving.

## Why port rather than call Python

- Speed where it matters. The shard-wise requantizer and the per-group range search are bandwidth-bound and embarrassingly parallel; the Python tool runs them one tensor at a time with interpreter dispatch between every MLX call. The Rust driver can overlap the next tensor's reads with the current tensor's encode and write shards from another thread while keeping the same single-tensor peak memory.
- Control. Astronomical wants to modify the allocation policy and the encoder. The pure allocator is pinned to the Python reference by parity fixtures, so a modification is a visible, intended diff rather than drift.
- Calibration on the real graph. The exact sensitivity sweep runs through Astronomical's own Qwen3.5 engine, so the bit map is calibrated on the kernels the artifact will actually run through, not on an mlx-lm re-implementation.

## Pipeline and module map

Every `mod.rs` holds only module declarations and re-exports; each concern lives in a named file. OptiQ paths are relative to the `optiq` package of `mlx-optiq` 0.7.9.

| Stage | Module | OptiQ source | Needs MLX | Status |
| --- | --- | --- | --- | --- |
| List quantizable modules from safetensors headers | `module_inventory` | `core/sensitivity.py` `_index_bf16_layers`, `_quantizable_linears` | no | ported |
| Classify modules by name | `module_taxonomy` | `core/optimizer.py` `_identify_*`, `core/shardwise.py` `is_expert`, `is_router` | no | ported |
| Bit widths and group sizes | `bit_width` | the integer ladders throughout `core/` | no | ported |
| Build calibration sequences | `calibration::{sample, tokenizer, mix, seeded_shuffle}` | `calibration/datasets.py` | no | ported |
| Measure per-module KL sensitivity | `sensitivity::exact_sweep` through `sensitivity::probe::SensitivityProbeModel` | `core/sensitivity.py` `analyze_sensitivity_exact` | engine-side only | driver ported; Qwen3.5 probe in model-serving |
| The KL formula probes implement | `sensitivity::kl_divergence` | `core/sensitivity.py` `_kl_from_ref` | no | ported |
| The sweep output | `sensitivity::sensitivity_table` | `core/sensitivity.py` `SensitivityResult` | no | ported |
| Resume an interrupted sweep and report the ranking | `sensitivity::checkpoint`, `sensitivity::report` | `core/sensitivity.py` `_save_checkpoint`, `print_sensitivity_report` | no | ported |
| Score structurally without measurement | `sensitivity::static_scorer` | `core/sensitivity.py` `analyze_sensitivity_static` | no | ported |
| Allocate bits under a target | `allocation::optimizer` with `knapsack`, `block_floor`, `block_run_guard`, `protected_modules`, `bits_per_weight`, `predicate` | `core/optimizer.py` | no | ported |
| Encode with the range search | `encoder::range_search_policy`, `encoder::reference`, `encoder::mlx` | `core/lowbit.py` | `direct-mlx` (pure reference beside it) | ported |
| Requantize tensor by tensor | `requantize::{tensor_plan, bit_policy, shard_writer, driver}` | `core/shardwise.py` | `direct-mlx` | ported |
| Write the artifact documents | `artifact::{layout, quantization_config, metadata, sensitivity_document}` | `artifacts.py`, `backends/mlx_backend.py` metadata | no | ported |
| Roofline throughput model | `latency::{hardware_profile, roofline, calibrated_model}` | `core/latency.py` | no | ported |
| Key-value cache allocation | `kv_cache::{cache_sensitivity, cache_allocator, cache_config_document}` | `core/kv_cache.py` | engine-side only | sweep driver and allocator ported; an engine probe is not yet written |
| Orchestrate | `pipeline` with `artifact_files` and `conversion` | `models/llm.py` `run_llm_pipeline` | `direct-mlx` for the full path | ported |
| Progress and ETA | `progress` | the per-layer ETA prints | no | ported |

### Not yet represented in the crate

These parts of OptiQ 0.7.9's quantization path have no module here yet:

- The quantized-baseline reference sweep's layer mutation: `core/sensitivity.py` `_exact_with_quantized_baseline`, `_capture_quantized_layer_state`, `_restore_quantized_layer_state`, `_mutate_quantized_layer_to_bits`. The `ReferenceMode` enum names the mode, but no stage implements it.
- Multi-token-prediction head preservation at convert time: `runtime/mtp_convert.py`.
- Sidecar placement for vision towers and the MTP head: `sidecar_layout.py`.
- The post-conversion KL evaluation of a finished artifact against a reference: `eval/kl.py`.
- Expert pruning for MoE models (the `prune-experts` command): `mep/profiler.py`, `mep/prune.py`.

## Dependency direction

The crate is a leaf. Its `direct-mlx` feature pulls in the MLX bindings and the runtime integration for the encoder and the requantizer, and nothing else. The exact sweep needs a forward pass, which the crate obtains through the `SensitivityProbeModel` trait; model-serving implements that trait for the resident Qwen3.5 model in `crates/model-serving/src/quantization_probe/` behind its own `direct-mlx` feature, because the swap needs access to typed weights that are private to that crate. So model-serving depends on this crate's pure surface, and this crate never learns about an engine.

The CLI's `astronomical quantize` verb has three sub-verbs. `plan` (allocate from an existing `optiq/sensitivity.json`) and `inspect` (read an artifact's provenance) are pure and compiled into every CLI build. `convert` runs the sweep and the requantizer in-process and is compiled in only with the CLI's `quantize-convert` feature, so the default CLI binary stays free of MLX linkage; without the feature the verb names the feature and exits with failure rather than being silently absent.

## Output format

The output is OptiQ's artifact layout, which three independent readers accept: stock mlx-lm (`config.json`'s `quantization` document with global defaults plus per-module `{bits, group_size, mode}` entries), OptiQ's own tools (`optiq/metadata.json` with the `per_layer` bit map, in the published `optiq_mixed_precision` layout for measured conversions and the static allocator's layout for structural ones, listing only quantized modules; and `optiq/sensitivity.json` with the measured sweep), and Astronomical's model-serving crate, whose strict `MlxQuantizationDocument` and `OptiQMetadata` readers validate the two documents against each other at load. The model-serving integration test `artifacts::optiq_artifact_round_trip` feeds this crate's writers straight into those readers.

Two rules from the field are encoded, not left to discipline: the metadata describes the weights that were written (a router passed through, a module dropped to group size 32), never the plan; and the sensitivity document ships with every quant so a re-target is a knapsack re-run, not a repeated ten-hour sweep.

## Calibration

The bundled mix is OptiQ's `optiq.jsonl`: 40 samples across six domains (prose, thinking traces, code reasoning, agent loops, tool calling, constraint-bearing instructions). Chat-style samples are rendered by role-prefixed concatenation, OptiQ's own fallback, because the Rust `tokenizers` crate has no chat-template engine; a caller who needs template-exact sequences supplies pre-tokenized samples through `CalibrationMix::from_pretokenized`. Tests use the Romeo and Juliet fixture exclusively.

## Progress reporting

Every long stage reports percentage completion and an estimated time to completion through `ProgressReporter` at a fixed five-second cadence, including the first unit and the final unit. A resumed sweep estimates from the units completed in the current run only. The CLI routes reports to standard error so standard output stays composable.

## Test plan

Test names start with `should_`; unit tests live beside their source as `*_tests.rs` modules; integration journeys live under `tests/integration/`; GPU contracts under `tests/direct_mlx/`; real-model journeys in the root acceptance package.

Unit tests (`cargo test -p astronomical-quantization --lib`) cover the taxonomy on names that bit real conversions, every allocator rule in isolation (efficiency ordering, the target as a floor, protected boosting, NaN handling, the uniform-reference sign detection, the attention slot reservation, the two-linear block cap, the run guard stride), the sweep driver's probe sequencing and the measured-majority guard against a scripted probe, checkpoint monotonicity and the Python checkpoint shape, calibration rendering and deterministic chunking, encoder packing and the never-worse-than-stock property, shard naming and the index, the artifact documents, the roofline arithmetic, the key-value cache allocator, and the progress tracker's cadence and ETA.

Integration journeys (`cargo test -p astronomical-quantization --test integration_tests`) pin the port to the Python reference through fixtures generated by `tests/fixtures/generate_optiq_parity_fixtures.py` with mlx-optiq 0.6.5: the allocator must reproduce five Python allocations bit for bit on a synthetic table that exercises every rule; the taxonomy must agree with OptiQ's four predicates on every fixture name; structural priorities must match to float rounding; the pure reference encoder must reproduce the Python range search's packed codes exactly at 2 and 3 bits and stock min/max at 4 and 8; the KL formula must match on a logits pair. The journeys also load the bundled mix end to end, inventory real safetensors files (2-D linears, fused gate-up splits, unfused expert stacking), build requantization plans with byte estimates, and round-trip the artifact documents through disk.

Direct-MLX contracts (`cargo test-quantization-direct-mlx`, `#[parallel]` except the peak-memory contract, which is `#[serial]` because peak memory is process-global) hold the Metal encoder to the pure reference (identical packed codes at 2 bits and on a 3-bit expert stack, scales within float rounding), require stock layout above 4 bits and with the search disabled, refuse the keep-bfloat16 tier and an indivisible last axis, require squared error never above stock min/max, require that MLX's own dequantize reads the GPU artifact back to the reference's values, run the pipeline on a synthetic source (a complete static artifact, the overwrite refusal, and the exact path through a fake probe against both the bf16 and the uniform-baseline reference), and run the requantizer end to end on a synthetic two-shard source, asserting the written map, the index, packed words identical to the reference, a typed error for a missing shard, and that peak MLX memory does not scale with tensor count.

Acceptance journeys (`cargo test-quantization-acceptance`, ignored, environment-discovered, serial) run lowest tier first: requantizer throughput with a warmup tensor; a static conversion validated and served back through the Qwen3.5 engine with a Romeo and Juliet continuation at a 2,048-token prefill chunk; the exact sweep with a simulated interruption and checkpoint resume; and tensor-by-tensor parity with an artifact the Python tool produced from the same source (identical packed words at 4 bits and above, no worse squared error below, identical bit maps from the Python sweep).

CLI journeys (`cargo test -p astronomical-cli --test integration_tests quantize`) cover argument parsing including the comma-separated ladder and its validation, `plan` against a document written to a temporary directory, `inspect` on a directory without metadata, and the `convert` refusal in a default build.

## Implementation order

1. `module_taxonomy`, `bit_width`, `progress`: small, pure, and everything else uses them.
2. `allocation`: drive it green against `tests/integration/allocation_parity.rs`; this is the first point where the port can be used, through `astronomical quantize plan` on a Python-produced `optiq/sensitivity.json`.
3. `artifact`, then the model-serving round-trip test.
4. `encoder::reference` and `sensitivity::kl_divergence`, driven by `tests/integration/encoder_reference_parity.rs` and `tests/integration/kl_divergence_parity.rs`; then `encoder::mlx` against the direct-MLX contracts.
5. `module_inventory` and `requantize`, driven by the integration journeys and then the direct-MLX requantizer contract. At this point `convert --method static` works end to end.
6. `calibration`, `sensitivity::exact_sweep`, and the Qwen3.5 probe in model-serving, driven by the scripted-probe unit tests and then the acceptance sweep journey.
7. `latency` and `kv_cache`, which have no consumer on the critical path.

## Open questions recorded for the implementer

- `forward_chunk` on the resident model returns the final chunk's logits; the probe needs logits at every position of the sample for KL. Confirm the returned shape and widen the path if it returns only the last row.
- Resolved: the calibration shuffle reproduces numpy's `RandomState(seed).shuffle` exactly (MT19937 with `init_genrand` and the legacy masked-rejection interval), so a Rust sweep and a Python sweep of the same mix and tokenizer calibrate on the same sequences. Chat samples still render by role-prefixed concatenation rather than the tokenizer's chat template, so sequences match the Python tool only for models without one, or with pre-tokenized samples.
- The CLI's `quantize-convert` feature links the model-serving engine into the CLI binary. If that proves too heavy for the release bundle, the `convert` sub-verb can move to a separate `astronomical-quantize` binary under `apps/` with no change to this crate.

## Attribution

This crate is a translation, and its algorithms belong to the people who designed them:

- **OptiQ** (`mlx-optiq`, MIT License, Copyright (c) 2026 Thin Signal; [mlx-optiq.com](https://mlx-optiq.com), [PyPI](https://pypi.org/project/mlx-optiq/)) designed the following, which this crate ports:
  - the sensitivity sweep and the structural scorer
  - the knapsack allocator with its block rules
  - the range-search encoder
  - the shard-wise requantizer
  - the calibration mix and loader
  - the latency and key-value cache models
  - the artifact layout
- **oMLX** ([github.com/jundot/omlx](https://github.com/jundot/omlx), Apache-2.0) is where OptiQ's range-search encoder comes from: OptiQ adapted it from oMLX's oQ quantizer.
- **NumPy's legacy random stream** is reproduced by the calibration shuffle. It builds on:
  - MT19937, by Makoto Matsumoto and Takuji Nishimura
  - the interval draw, by Richard J. Wagner and Magnus Jonsson
  - Jean-Sebastien Roy's randomkit, which carries both into NumPy

The full license texts are in `third-party/THIRD_PARTY_NOTICES`. Each ported module names the OptiQ function it translates.
