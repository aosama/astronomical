# astronomical-quantization

Sensitivity-driven mixed-precision affine quantization for MLX artifacts: a Rust port of the quantization core of [OptiQ](https://mlx-optiq.com) (the `mlx-optiq` Python package, MIT License, Copyright (c) 2026 Thin Signal). The full attribution, including the oMLX lineage of the range-search encoder, is in `third-party/THIRD_PARTY_NOTICES`; the vendored calibration mix carries its license beside it in `calibration/LICENSE-OPTIQ`.

This document is the design record for the port. The crate holds the typed design and the complete test plan. The taxonomy, bit widths, progress reporting, the allocator, the module inventory with its output layout, the structural scorer, the KL formula, the pure reference encoder, the artifact writers, the Metal range-search encoder, the requantizer (plan, bit policy, shard writer, and the MLX driver with its source reader for fused, stacked, FP8, and packed layouts), calibration loading with chat-template rendering, the sweep driver with its checkpoint and report, the pipeline orchestration, the roofline latency model, and the key-value cache sweep and allocator are ported and tested. No algorithm body in the crate is a stub, and the Qwen3.5 probe in model-serving covers every module the allocator decides: dense layers, the mixture-of-experts router, the shared expert, and the routed experts.

## Parity with OptiQ

The port is pinned to mlx-optiq 0.8.2 (with mlx-lm 0.32.2) and proven against it on the original `Qwen/Qwen3.5-0.8B` release, by the acceptance journeys in `verification/acceptance/tests/quantization/python_parity.rs`:

| What | Result |
| --- | --- |
| `static` conversion, 4/8 ladder | every file byte for byte: the language shards, the vision sidecar, the multi-token-prediction sidecar, `config.json`, `optiq/metadata.json`, `optiq/sensitivity.json`, the shard index, the model card |
| Calibration sequences | token for token |
| Bit map from OptiQ's own sweep | identical for all 186 modules, same achieved bits per weight |
| Measured conversion resumed from OptiQ's sweep checkpoint | every file byte for byte |
| The sweep itself (a different engine on each side) | rank correlation 0.961 at 4 bits and 0.993 on the gain from 4 to 8 bits; 182 of 186 modules (97.8%) land on the same width, at the same achieved bits per weight |
| KL evaluation of OptiQ's measured artifact against the source | mean 0.1029 (OptiQ 0.1077), median 0.0350 (0.0358), 95th percentile 0.3451 (0.3506) |
| `static` conversion, 2/3/4/8 ladder | identical except 171 of 5.87 million range-searched groups (0.003%) |

The divergence at 8 bits alone is reported, not asserted: an 8-bit probe moves a bfloat16 base so little that OptiQ's reading (bfloat16 logits) sits on its own noise floor, about 45 times above this engine's float32 head, so that column ranks noise against signal (0.63). The allocator spends its budget on the gain, which agrees.

The low-bit row is not a defect on either side. The range search keeps a candidate only when it is strictly better, and OptiQ computes the squared error with MLX's Metal `pow` kernel, which does not round the same way in every MLX build. This crate computes an exact square, which is the same on every build and device; the two differ only where two candidates are within the last bit of each other.

Both source spellings are covered: the same weights spelled as an original Hugging Face release (`model.language_model.*`, `model.visual.*`, PyTorch convolution axes, RMSNorm `gamma - 1`) and as an mlx-lm conversion convert to the same bytes. Mixture-of-experts models are covered by a synthetic Qwen3.5 checkpoint, a few megabytes of random weights in an original release's layout. Its measured sweep against the uniform reference visits the same 102 modules as OptiQ's in the same order, ranks them with correlation 0.999 at 8 bits, and lands every module on the same width; each router and routed-expert swap is restored exactly (`verification/acceptance/tests/quantization/mixture_of_experts_sweep.rs`). Against a bf16 reference this crate also measures the routed experts, which OptiQ's bf16 sweep does not (it walks plain linear layers only), so there the two bit maps differ by design. The same checkpoint (fused expert tensors, a router, a shared expert, an untied head; `tests/fixtures/build_optiq_synthetic_moe_source.py`) converts byte for byte on the 4/8 ladder, checked with `tests/fixtures/compare_optiq_artifacts.py`.

To rebuild the reference: `crates/quantization/tests/fixtures/build_optiq_parity_sources.py` writes bf16 sources in both spellings from an installed quantized model, and `build_optiq_reference_artifacts.py` runs OptiQ on one of them. The second script exists for a reason worth knowing: mlx-optiq 0.8.2 passes its allocation to mlx-lm positionally, mlx-lm 0.32 inserted a parameter ahead of it, and the allocation is silently dropped, so an unassisted `optiq convert` writes a uniform quant under metadata that describes a mixed one. The script re-routes that one argument and changes nothing else.

## Why port rather than call Python

- Speed where it matters. The shard-wise requantizer and the per-group range search are bandwidth-bound and embarrassingly parallel; the Python tool runs them one tensor at a time with interpreter dispatch between every MLX call. The Rust driver can overlap the next tensor's reads with the current tensor's encode and write shards from another thread while keeping the same single-tensor peak memory.
- Control. Astronomical wants to modify the allocation policy and the encoder. The pure allocator is pinned to the Python reference by parity fixtures, so a modification is a visible, intended diff rather than drift.
- Calibration on the real graph. The exact sensitivity sweep runs through Astronomical's own Qwen3.5 engine, so the bit map is calibrated on the kernels the artifact will actually run through, not on an mlx-lm re-implementation.

## Pipeline and module map

Every `mod.rs` holds only module declarations and re-exports; each concern lives in a named file. OptiQ paths are relative to the `optiq` package of `mlx-optiq` 0.8.2.

| Stage | Module | OptiQ source | Needs MLX | Status |
| --- | --- | --- | --- | --- |
| List quantizable modules from safetensors headers | `module_inventory` | `core/sensitivity.py` `_index_bf16_layers`, `_quantizable_linears` | no | ported |
| Decide each tensor's output name, axis order and file | `module_inventory::output_layout` | mlx-lm `qwen3_5` and `qwen3_5_moe` `sanitize`; `vlm/sidecar.py` `_is_mm_key`, `_store_key`, `_sanitize_conv` | no | ported for the Qwen3.5 family; other families keep their source names |
| Order modules the way mlx-lm visits them | `module_inventory::traversal_order` | mlx `Module.named_modules` over the mlx-lm model classes | no | ported for the Qwen3.5 family |
| Classify modules by name | `module_taxonomy` | `core/optimizer.py` `_identify_*`, `core/shardwise.py` `is_expert`, `is_router` | no | ported |
| Bit widths and group sizes | `bit_width` | the integer ladders throughout `core/` | no | ported |
| Build calibration sequences | `calibration::{sample, tokenizer, chat_template, python_json, mix, seeded_shuffle}` | `calibration/datasets.py`; transformers' chat-template environment | no | ported |
| Measure per-module KL sensitivity | `sensitivity::exact_sweep` through `sensitivity::probe::SensitivityProbeModel` | `core/sensitivity.py` `analyze_sensitivity_exact` | engine-side only | ported; the Qwen3.5 probe in model-serving swaps dense modules, the router, the shared expert, and routed-expert projections (splitting the engine's fused gate-up array for the length of a probe) |
| The KL formula probes implement | `sensitivity::kl_divergence` | `core/sensitivity.py` `_kl_from_ref` | no | ported |
| The sweep output | `sensitivity::sensitivity_table` | `core/sensitivity.py` `SensitivityResult` | no | ported |
| Resume an interrupted sweep and report the ranking | `sensitivity::checkpoint`, `sensitivity::report` | `core/sensitivity.py` `_save_checkpoint`, `print_sensitivity_report` | no | ported |
| Score structurally without measurement | `sensitivity::static_scorer` | `core/sensitivity.py` `analyze_sensitivity_static` | no | ported |
| Allocate bits under a target | `allocation::optimizer` with `knapsack`, `block_floor`, `block_run_guard`, `protected_modules`, `bits_per_weight`, `predicate` | `core/optimizer.py` | no | ported |
| Encode with the range search | `encoder::range_search_policy`, `encoder::reference`, `encoder::mlx` | `core/lowbit.py` | `direct-mlx` (pure reference beside it) | ported |
| Requantize tensor by tensor | `requantize::{tensor_plan, bit_policy, shard_writer, driver}` | `core/shardwise.py` | `direct-mlx` | ported |
| Write the artifact documents | `artifact::{layout, quantization_config, metadata, sensitivity_document, converted_config, multimodal_config, generation_config, model_card}` | `artifacts.py`, `backends/mlx_backend.py`, `vlm/sidecar.py`, `sidecars.py`; mlx-lm `save_config`, `create_model_card` | no | ported |
| Write the multi-token-prediction head | `requantize::{mtp_sidecar, mtp_sidecar_writer}`, `artifact::mtp_config` | `runtime/mtp_convert.py` `preserve_mtp` | `direct-mlx` for the writer | ported |
| Evaluate a finished artifact against a reference | `evaluation::{prompts, kl_statistics, kl_evaluation}` through `evaluation::KlDivergenceEngine` | `eval/kl.py` `evaluate_kl` | engine-side only | ported; the Qwen3.5 engine in model-serving forwards both models; the reference is named explicitly rather than resolved from the Hub |
| Roofline throughput model | `latency::{hardware_profile, roofline, calibrated_model}` | `core/latency.py` | no | ported |
| Key-value cache allocation | `kv_cache::{cache_sensitivity, cache_allocator, cache_config_document}` | `core/kv_cache.py` | engine-side only | sweep driver and allocator ported; an engine probe is not yet written |
| Orchestrate | `pipeline` with `artifact_files`, `conversion` and `probe_source_view` | `models/llm.py` `run_llm_pipeline` | `direct-mlx` for the full path | ported |
| Progress and ETA | `progress` | the per-layer ETA prints | no | ported |

### Not yet represented in the crate

These parts of OptiQ 0.8.2's quantization path have no module here yet:

- Sweeping a mixture-of-experts model whose routed experts do not fit in memory. The probe swaps resident expert arrays; a model the engine has to page from disk has none, its experts are counted as skipped, and the measured-majority guard refuses the sweep, so such a model converts with the `static` method. OptiQ streams the reference from SSD for this case (`runtime/moe_stream.py`).
- A measured run on a real mixture-of-experts base. Routed-expert probing is proven on a synthetic model (below); the smallest real one is about 70 GB in bf16 and about 20 GB as a uniform baseline.
- Model-class bit overrides and the streaming convert's per-unit splitting (`optiq_quant_overrides`, `convert_llm_to_mlx_streaming`), which serve families OptiQ vendors its own model classes for.
- Requantizing sources already packed in a non-affine format (MXFP4, FP8 with E8M0 scales): `_requantize_prequantized`, `_relabel_e8m0_headers`.
- Expert pruning for MoE models (the `prune-experts` command): `mep/profiler.py`, `mep/prune.py`.
- An engine probe for the key-value cache sweep.
- The output layout and traversal order of families other than Qwen3.5. Their tensors keep the source's names and their allocation table is in a deterministic but not mlx-lm's order.

## Dependency direction

The crate is a leaf. Its `direct-mlx` feature pulls in the MLX bindings and the runtime integration for the encoder and the requantizer, and nothing else. The exact sweep needs a forward pass, which the crate obtains through the `SensitivityProbeModel` trait; model-serving implements that trait for the resident Qwen3.5 model in `crates/model-serving/src/quantization_probe/` behind its own `direct-mlx` feature, because the swap needs access to typed weights that are private to that crate. So model-serving depends on this crate (its pure surface, plus the requantizer's source reader under `direct-mlx`, so a probe reads a module's bf16 source exactly as the final conversion will), and this crate never learns about an engine.

The CLI's `astronomical quantize` verb has four sub-verbs. `plan` (allocate from an existing `optiq/sensitivity.json`) and `inspect` (read an artifact's provenance) are pure and compiled into every CLI build. `convert` runs the sweep and the requantizer in-process, and `evaluate` forwards a reference model and a finished artifact over the same prompts and reports how far the artifact's next-token distribution drifts (mean, median and 95th-percentile KL). Both are compiled in only with the CLI's `quantize-convert` feature, so the default CLI binary stays free of MLX linkage; without the feature each verb names the feature and exits with failure rather than being silently absent.

## Output format

The output is OptiQ's artifact layout, which three independent readers accept: stock mlx-lm, OptiQ's own tools, and Astronomical's model-serving crate, whose strict `MlxQuantizationDocument` and `OptiQMetadata` readers validate the two documents against each other at load. The model-serving integration test `artifacts::optiq_artifact_round_trip` feeds this crate's writers straight into those readers.

- `model*.safetensors` hold the language model under mlx-lm's names (`language_model.model.*`), with convolution weights in MLX axis order and RMSNorm weights as true gammas, whatever the source's spelling. OptiQ converts through mlx-lm, whose model class renames and reshapes on load; this crate walks shards without instantiating the model, so `module_inventory::output_layout` applies the same rules.
- `optiq_vision.safetensors` at the root holds the vision and audio towers at bfloat16 under the names the front-ends load, registered under `optiq_vision` in `config.json` and listed in the shard index so index-driven loaders find them. A config that declares a tower the source never shipped has its multimodal claims removed instead.
- `config.json` carries the `quantization` document (and its `quantization_config` twin) with global defaults and one `{bits, group_size}` entry per quantized module.
- `optiq/metadata.json` is OptiQ's pipeline layout for every method (reference, target and achieved bits per weight, counts at the ladder's ends, `per_layer`), and `optiq/sensitivity.json` is the sweep.
- `optiq/mtp.safetensors` holds the release's multi-token-prediction head, projections quantized at 4 bits with stock min/max, the fusion layer and norms as stored, fused expert tensors split per expert; `config.json` registers it (`mtp_file`, `mtplx_mtp_quantization`, `mlx_lm_extra_tensors`). It sits under `optiq/` so a root-level `*.safetensors` glob never loads it, and only the final conversion writes it: a uniform baseline or a bfloat16 probe copy has no use for it. A head with a projection the group size does not divide is left out, as OptiQ leaves it out.
- `generation_config.json` carries the family's recommended sampling under whatever the source published, and `README.md` is mlx-lm's minimal model card.

Rules from the field are encoded, not left to discipline. Tower tensors are never quantized. The token embedding table is quantized at the allocation's highest width and is never scored or allocated, because OptiQ's sensitivity methods walk linear layers only; it therefore appears in `config.json` and not in `per_layer`. A head tied to the embedding is dropped, as mlx-lm drops it. The metadata describes the weights that were written (a module dropped to a smaller group size, or left at bfloat16 and recorded at 16 bits), never the plan. The sensitivity document ships with every quant so a re-target is a knapsack re-run, not a repeated ten-hour sweep: a conversion started over an existing `sensitivity_checkpoint.json` re-measures nothing.

## Calibration

The bundled mix is OptiQ's `optiq.jsonl`: 40 samples across six domains (prose, thinking traces, code reasoning, agent loops, tool calling, constraint-bearing instructions). Chat-style samples are rendered through the model's own chat template (`chat_template.jinja`, else the `chat_template` of `tokenizer_config.json`) in an environment that mirrors the one transformers builds: block trimming, loop controls, `raise_exception`, a `tojson` backed by Python's `json.dumps(ensure_ascii=False)` so tool definitions enter the prompt in the same text, and the Python string and mapping methods templates call. A sample the template refuses is retried without its tools and then rendered by role-prefixed concatenation, OptiQ's order. With the numpy-exact shuffle this reproduces OptiQ's calibration sequences token for token. Tests use the Romeo and Juliet fixture exclusively.

## Progress reporting

Every long stage reports percentage completion and an estimated time to completion through `ProgressReporter` at a fixed five-second cadence, including the first unit and the final unit. A resumed sweep estimates from the units completed in the current run only. The CLI routes reports to standard error so standard output stays composable.

## Test plan

Test names start with `should_`; unit tests live beside their source as `*_tests.rs` modules; integration journeys live under `tests/integration/`; GPU contracts under `tests/direct_mlx/`; real-model journeys in the root acceptance package.

Unit tests (`cargo test -p astronomical-quantization --lib`) cover the taxonomy on names that bit real conversions, every allocator rule in isolation (efficiency ordering, the target as a floor, protected boosting, NaN handling, the uniform-reference sign detection, the attention slot reservation, the two-linear block cap, the run guard stride), the sweep driver's probe sequencing and the measured-majority guard against a scripted probe, checkpoint monotonicity and the Python checkpoint shape, calibration rendering and deterministic chunking, encoder packing and the never-worse-than-stock property, shard naming and the index, the artifact documents, the roofline arithmetic, the key-value cache allocator, and the progress tracker's cadence and ETA.

Integration journeys (`cargo test -p astronomical-quantization --test integration_tests`) pin the port to the Python reference through fixtures generated by `tests/fixtures/generate_optiq_parity_fixtures.py` with mlx-optiq 0.8.2: the allocator must reproduce five Python allocations bit for bit on a synthetic table that exercises every rule; the taxonomy must agree with OptiQ's four predicates on every fixture name; structural priorities must match to float rounding; the pure reference encoder must reproduce the Python range search's packed codes exactly at 2 and 3 bits and stock min/max at 4 and 8; the KL formula must match on a logits pair. The journeys also load the bundled mix end to end, inventory real safetensors files (2-D linears, fused gate-up splits, unfused expert stacking), build requantization plans with byte estimates, and round-trip the artifact documents through disk.

Direct-MLX contracts (`cargo test-quantization-direct-mlx`, `#[parallel]` except the peak-memory contract, which is `#[serial]` because peak memory is process-global) hold the Metal encoder to the pure reference (identical packed codes at 2 bits and on a 3-bit expert stack, scales within float rounding), require stock layout above 4 bits and with the search disabled, refuse the keep-bfloat16 tier and an indivisible last axis, require squared error never above stock min/max, require that MLX's own dequantize reads the GPU artifact back to the reference's values, run the pipeline on a synthetic source (a complete static artifact, the overwrite refusal, and the exact path through a fake probe against both the bf16 and the uniform-baseline reference), and run the requantizer end to end on a synthetic two-shard source, asserting the written map, the index, packed words identical to the reference, a typed error for a missing shard, and that peak MLX memory does not scale with tensor count.

Acceptance journeys (`cargo test-quantization-acceptance`, ignored, configured by an explicit test-run configuration, serial) run lowest tier first: requantizer throughput with a warmup tensor; a static conversion validated and served back through the Qwen3.5 engine with a Romeo and Juliet continuation at a 2,048-token prefill chunk; the exact sweep with a simulated interruption and checkpoint resume; and the parity journeys against artifacts OptiQ built from the same source, described under "Parity with OptiQ" above. The measured-sweep journey runs about three minutes on Qwen3.5 0.8B, so the parity lane is run with `--case-budget-seconds 900`.

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

- Resolved: `forward_chunk` returns only the last row, so the probe forwards through `forward_chunk_with_pre_final_normalization_hidden_states` and applies the final norm and head at every position (`all_position_vocabulary_logits`), with the head in Float32 and the logits stored as Float16.
- Measured on Qwen3.5 0.8B with a bf16 reference: the forward is bit-reproducible and a restore is exact, but every probe at 4 bits and above lands on a floor of about 1e-3 KL that is not monotone in width (one `up_proj`: 8-bit 1.3e-3, 6-bit 1.7e-3, 5-bit 3.1e-3, 4-bit 2.0e-3, 3-bit 4.5e-2, 2-bit 1.0e-1). The floor is the bf16-activation network's response to any weight perturbation; OptiQ's mlx-lm sweep runs bf16 activations too and shares it. On a small base the default 4/8 ladder therefore ranks modules mostly on noise; a ladder whose low rung clears the floor, more calibration sequences, or a Float32-activation probe forward would sharpen it.
- Resolved: the calibration shuffle reproduces numpy's `RandomState(seed).shuffle` exactly (MT19937 with `init_genrand` and the legacy masked-rejection interval), and chat samples render through the model's chat template, so a Rust sweep and a Python sweep of the same mix and tokenizer calibrate on the same sequences.
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
