# Cargo Commands for Testing

Concise catalog of the cargo commands that run this repository's Rust test
surface. Cargo commands are the interface, and every entry explains what it
runs and why it exists. Real-model journeys are invoked directly with
`--test-threads=1`; each test owns its timeout and cancellation/cleanup path.

## Cargo aliases (the interface for lanes)

Lanes are driven through `.cargo/config.toml` aliases so the packages,
features, targets, and the GPU serial rule (`--test-threads=1`) cannot be
forgotten:

- `cargo test-offline` / `cargo test-rest-api` / `cargo test-e2e` / `cargo test-offline-and-rest` — the CPU lanes (see below for what each covers).
- `cargo verify` — workspace-wide compile gate before every push.
- `cargo test-direct-mlx` — the model-serving direct-MLX contract suite plus
  the runtime-integration MLX-C coverage bridge contract (CPU-only, seconds).
- `cargo test-mlx-memory` — every native memory-contract phase (model-swap
  allocator, native probe, MLX-C transitions) in one bounded invocation. The
  probe binary must exist in the native build store first:
  `cargo test -p astronomical-runtime-integration --features mlx-memory-contract-probe --no-run`.
- `cargo test-acceptance` — the whole real-model acceptance package across
  every feature gate.

Real-model journeys are `#[ignore]`d; the memory and acceptance aliases pass
`--ignored` for you. A bare `cargo test -p astronomical-acceptance` without
features runs nothing — aliases are the interface.

## Offline and REST lanes (CPU only, safe to parallelize)

Lane vocabulary (issue #1208): **offline** is the property — sealed and
deterministic, no real network beyond localhost, no real model artifacts,
everything scripted. Scope words name which seam a test crosses: **unit**
(in-process, fakes), **integration** (stubs or fixture binaries over IPC,
filesystem), **end-to-end (E2E)** (the real `astronomicald` binary over real
HTTP), **acceptance** (env-gated real-model journeys).

- `cargo test-offline` — runs every package's `offline_tests` binary. The
  fast, machine-independent verification lane; safe to run on any host.
- `cargo test -p astronomical-test-harness` — the harness self-contract:
  pass propagation, panic propagation, and 120-second timeout abort for
  `astronomical_test_harness::offline_test!`, the macro every offline test
  is written with so the timeout rule is enforced by the harness rather
  than by discipline.
- `unshare -rn cargo test -p astronomical-config --test offline_tests`
  — Linux-only physics check (CI "Linux offline sandbox" job): runs the
  pilot crate's tests inside a user+network namespace where opening any
  socket fails, proving the offline property instead of promising it.
- `cargo test -p astronomical-cli` — the CLI verb surface: unit tests live
  beside their source as `*_tests.rs` modules inside `src/`, and the
  offline-lane target runs the stub-daemon/stub-server verb journeys
  under `tests/cli_journeys/`. No daemon, no GPU, no network.
- `cargo test-rest-api` — runs the REST contract and daemon
  `rest_api_tests` binaries, exercising the public HTTP surface in-process.
- `cargo build --bins -p astronomical-daemon && cargo test-e2e` — the root
  whole-product E2E suite (`tests/e2e`): every journey spawns a daemon
  binary over its real HTTP/IPC surface. By default the fixture
  `astronomical-daemon-test-daemon` serves those surfaces with the scripted
  worker, so the suite needs no model artifacts and runs on any machine;
  `ASTRONOMICAL_REAL_MODEL=1` flips the journeys to the real `astronomicald`
  plus the real inference worker and installed artifacts (see the
  whole-product daemon journeys section below). The build step is required
  because `CARGO_BIN_EXE_` covers only same-package binaries; the harness's
  `workspace_binary` fails loudly with this exact command when a binary is
  missing.
- `cargo test-offline-and-rest` — both lanes together. This is the
  pre-commit verification gate alongside `cargo fmt --all -- --check`.
- `cargo xtask tier-pyramid` — workstream 5 of #1198: count tests per
  tier (unit / integration / e2e / acceptance), report the pyramid
  distribution, and fail when a tier drifts more than 5 percentage points from
  the committed `scripts/ci/tier-pyramid-baseline.json`. Pass
  `--update-baseline` only inside a reviewed pull request that intends the
  migration; `scripts/ci/tests/test-report-tier-pyramid-contract.sh` pins the
  contract.
- `cargo test -p astronomical-model-serving --test offline_tests qwen3_5_moe_offline -- --test-threads=1`
  — the focused Qwen3.5 sparse-artifact lane for quantization configuration,
  native/affine single- and multi-layer planning, and tensor-profile contracts;
  it uses no model runtime or installed model files.
- `cargo test -p astronomical-model-serving --test offline_tests qwen3_5_import_direction -- --test-threads=1`
  — enforces the Qwen3.5 module dependency direction: both engines depend on
  shared core, never on each other; resident owns no pager; the transitional
  family re-export module is absent; and attribution contains no streaming-only
  route dependency.
- `cargo test -p astronomical-model-serving --lib performance_attribution::performance_attribution_tests::measurement::should_record_an_external_operation_interval_only_when_attribution_is_enabled -- --exact`
  — verifies that an operation measured outside the request-owning thread is
  recorded against the request report's monotonic start time.
- `cargo test -p astronomical-model-serving --features direct-mlx --test offline_tests prompt_processing_chunk_sizer -- --test-threads=1`
  — verifies that resident and streaming execution use separate chunk-sizer
  policies: resident uses fixed chunks, while streaming owns SSD chunk sizing
  and short-tail folding.
- `cargo test -p astronomical-model-serving --features direct-mlx --test offline_tests engine_backed_worker::chat::resident_streaming_retry -- --test-threads=1`
  — verifies the private resident-to-streaming request retry, including
  one-shot behavior, pre-output eligibility, preservation of request cache
  identity, and release of the replay request after the first generated token
  without loading a model.

## Real-model acceptance journeys (Apple-Silicon host, serial only)

All commands below load large artifacts into wired GPU memory. They are
`#[ignore]`d, use Rust's `#[serial]` test annotation, and must run with
`--test-threads=1`. SSD-paging journeys have a 60-second timeout inside each
test; do not wrap them in a longer external timeout or run them concurrently.

### Memory-management acceptance (SSD paging journeys)

All journey commands resolve the real worker binary from the shared target
directory, so build it once first:
`cargo build --release -p astronomical-inference-worker`.

- `cargo test --release -p astronomical-acceptance --test memory_management_acceptance --features memory-management-acceptance -- --ignored --nocapture --test-threads=1`
  — the full memory-management journey suite: SSD-paged decode expert reuse,
  expert eviction, complete-residency control, prefill memory progress,
  reverse model swap, and the paging memory-shape experiment.
- `cargo test --release -p astronomical-acceptance --test memory_management_acceptance --features memory-management-acceptance model_ssd_streaming::follow_up_admission_paged_moe_rest_journey::should_admit_the_paged_moe_follow_up_turn_after_a_long_prefill -- --ignored --nocapture --exact --test-threads=1`
  — the paged-MoE follow-up admission regression journey: a long first turn
  teaches the RAM budget chunk-shaped activation evidence, and the follow-up
  turn in the same conversation must be admitted, never rejected with
  `generation context exceeds available GPU wired memory`. Resident and SSD
  streaming prefill both use the fixed 2,048-token test span; the journey has a
  60-second timeout and a 36 GB maximum configured memory ceiling.
- `cargo test --release -p astronomical-acceptance --test memory_management_acceptance --features memory-management-acceptance should_measure_paging_memory_shape_and_serving_rates_under_the_configured_ceiling -- --ignored --nocapture --exact --test-threads=1`
  — the paging memory-shape experiment (default 32 GB cell): samples the
  memory timeline during a 5,000-token Romeo-and-Juliet prefill plus 500
  decode tokens, prints the full performance-attribution segment tables, and
  preserves evidence under `target/acceptance-evidence/`. Environment cell
  knobs: `PAGING_EXPERIMENT_MLX_MEMORY_GB` (ceiling), `PAGING_EXPERIMENT_SSD_STREAMING_CHUNK_TOKENS`,
  `PAGING_EXPERIMENT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL`,
  `PAGING_EXPERIMENT_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL`, and
  `ASTRONOMICAL_POSITIONAL_READ_PARALLELISM` (bounded-read pool size).

### Memory-ceiling sweep cells (issue #1120 evidence lane)

One command per ceiling cell; run each as its own process (MLX memory limits
are process-global). Each cell has a 60-second test timeout with a 45-second
serving budget, appends one `memory_ceiling_sweep` record with its `memory_cell`
evidence to
`tests/acceptance/tests/performance_throughput/throughput-history.jsonl`, and
fails
when the serving budget is exhausted so partial evidence remains available.
Set `MEMORY_SWEEP_CELL_ORDER` to the cell's position in your sweep so records
carry it; the recommended protocol brackets the sweep with the 32 GB cell
first and last and interleaves the rest so page-cache warmth cannot masquerade
as a ceiling effect.

- `cargo test --release -p astronomical-acceptance --features performance_throughput --test performance_throughput performance_throughput::memory_ceiling_sweep::should_serve_the_large_sparse_moe_under_a_23gb_ceiling_and_record_the_memory_cell -- --ignored --nocapture --exact --test-threads=1`
- The same command with `..._28gb_...`, `..._32gb_...`, `..._35gb_...`,
  `..._38gb_...`, and `..._40gb_...` runs the remaining cells. Each journey
  uses a 2,048-token prefill chunk, a 500-token output target, and a
  500-input/50-output warmup. Before loading the model, the worker-reported
  effective ceiling must be no greater than 36 GB; otherwise the cell fails
  closed without loading a model. Completed 38 GB and 40 GB cells must report
  hybrid expert residency. Partial rates and load/generation I/O evidence are
  recorded.

### Serving throughput (production-faithful rates)

- `cargo test --release -p astronomical-acceptance --features performance_throughput --test performance_throughput performance_throughput::qwen3_5_moe::should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput -- --ignored --nocapture --exact --test-threads=1`
  — resident-model text throughput: 9,000–11,000 Romeo-and-Juliet input
  tokens, 450–550 output tokens, 500-input/50-output warmup, persistent prompt
  cache disabled, 2,048-token prefill chunks, and a 36 GB maximum configured
  active-memory ceiling. Server-attributed rates are appended to the durable
  history log. The measured journey lives in a nested module, so `--exact` needs the full
  `performance_throughput::qwen3_5_moe::` module path.
- The same command with the module path
  `performance_throughput::qwen3_5_moe_vision::should_measure_resident_sparse_moe_vision_prompt_processing_and_decode_throughput`
  runs the vision variant under the same input/output, cache, chunk, memory,
  and warmup constraints.

### Model-serving installed-artifact journeys

- `cargo test -p astronomical-acceptance --features direct-mlx --test installed_model_contracts <journey_filter> -- --ignored --test-threads=1`
  — the installed-artifact acceptance binaries (for example the Qwen-Image-2.1
  journeys gated on `ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY`).

### Inference-worker serving acceptance (dense-model journeys)

- `cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving serving_acceptance::chat::follow_up_turn_admission_rest::should_admit_the_follow_up_turn_when_the_ssd_streaming_chunk_is_wider -- --ignored --nocapture --exact --test-threads=1`
  — the small dense Qwen3.5 variant of the follow-up admission regression: a
  wider SSD-streaming chunk is configured while the dense model never pages,
  so the resident mode's own operation scope must govern the activation
  reserve and both conversation turns must complete. The journey caps
  configured model memory at 36 GB.

### Qwen3.5 artifact-derived engine selection and retry journeys

Each command below launches the production worker and makes a real public REST
request using a configured sparse artifact. The resident journey is bounded to
115 seconds; each streaming/retry SSD journey is bounded to 60 seconds. All
three tests are `#[serial]`, run with one test thread, and enforce a 36 GB
maximum configured active-memory ceiling.

- `cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving serving_acceptance::qwen3_5::automatic_engine_selection_rest::should_select_resident_engine_when_artifact_geometry_and_prompt_fit -- --ignored --nocapture --exact --test-threads=1`
  — verifies a request selects resident execution when validated artifact
  geometry plus prompt context fit the configured ceiling.
- `cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving serving_acceptance::qwen3_5::automatic_engine_selection_rest::should_select_streaming_engine_when_complete_residency_headroom_does_not_fit -- --ignored --nocapture --exact --test-threads=1`
  — verifies a request selects paging when the artifact's complete-residency
  headroom does not fit.
- `cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving serving_acceptance::qwen3_5::automatic_engine_selection_rest::should_restore_cached_prefix_after_an_invisible_resident_to_streaming_retry -- --ignored --nocapture --exact --test-threads=1`
  — verifies one invisible resident-to-streaming retry retains persistent
  prefix reuse and emits a nonzero retry-attribution interval.

### Live ceiling demotion journeys (stop-first worker restart)

Both journeys lower the public MLX ceiling below the loaded artifact size. The
resident engine cannot shrink in place, so the daemon stops and reaps the
old worker before it launches the replacement, and the streaming engine then
serves the same conversation. Each journey is `#[serial]`, runs with one test
thread, uses the Romeo and Juliet fixture with a fixed 2,048-token prefill
chunk, stays within a 36 GB configured ceiling, and is bounded to 60 seconds
because it exercises solid-state-drive streaming.

- `cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving serving_acceptance::qwen3_5::live_ceiling_lowering_rest::should_serve_after_lowering_the_live_ceiling_below_the_artifact_size -- --ignored --nocapture --exact --test-threads=1`
  — lowers the ceiling on an idle serving session and requires a follow-up
  request to return generated output under the streaming engine.
- `cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving serving_acceptance::qwen3_5::queued_live_ceiling_demotion_rest::should_demote_the_live_ceiling_queued_behind_an_active_generation -- --ignored --nocapture --exact --test-threads=1`
  — lowers the ceiling while a generation is in flight; the update must queue,
  the active request must finish normally, and the restart must not begin until
  that request completes.

The offline daemon tests that pin the stop-first ordering and the four
failure branches need no real model and no optimized build:

- `cargo test -p astronomical-daemon --test offline_tests worker_memory_demotion -- --test-threads=1`

### Whole-product daemon journeys (tests/e2e, fidelity-switchable)

Both journeys run in two fidelities selected by `ASTRONOMICAL_REAL_MODEL=1`
and live in the `cargo test-e2e` suite: by default they drive the fixture
daemon and scripted worker (no artifacts), and with the variable set they
spawn the real `astronomicald`, which discovers and spawns the real inference
worker from its own directory — build both binaries first (`cargo build
--release -p astronomical-daemon -p astronomical-inference-worker`), run one
test thread, and respect each journey's 115-second budget.

- `ASTRONOMICAL_REAL_MODEL=1 cargo test --release -p astronomical-e2e --test e2e should_serve_one_schema_matching_json_object_over_the_cli_ipc_path -- --nocapture --exact --test-threads=1`
  — the CLI `respond --schema` wire contract end to end: the journey sends the
  exact `ChatGenerate` request the CLI verb sends (raw schema JSON in
  `schema_json`) over the daemon's IPC socket, through the daemon's schema
  validation into the worker's logits-masked structured generation, and
  requires a real model's reply to be one JSON object matching the schema.
  Needs the installed Qwen3.5-2B-4bit artifact under the configured Development
  model directories.
- `ASTRONOMICAL_REAL_MODEL=1 ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY=<installed snapshots directory> cargo test --release -p astronomical-e2e --test e2e should_serve_one_qwen_image_over_http_end_to_end -- --nocapture --exact --test-threads=1`
  — one 256×256 Qwen-Image-2.1 render over real HTTP through the whole live
  process tree: discovery advertisement, worker startup, model swap, the native
  engine, and the REST response contract.

### Prompt-cache acceptance journeys (resident sparse MoE role)

- `cargo test -p astronomical-acceptance --features direct-mlx --test engine_prompt_cache -- --ignored --nocapture --test-threads=1`
  — the whole prompt-cache acceptance surface (cache disabled, cache-miss
  memory, restore peak, parity, block reporting, partial tail reuse, startup
  cleanup attribution, and the three vision journeys). Every one of these
  journeys runs the resident sparse MoE role (`resident_sparse_moe`) because
  they measure cache behaviour, not expert paging; the whole binary takes about
  three minutes. The one exception is the interaction matrix journey, which
  fails closed unless a cell is selected.
- `ASTRONOMICAL_PROMPT_CACHE_INTERACTION_ACCEPTANCE_CELL=<cell> cargo test --release -p astronomical-acceptance --features direct-mlx --test engine_prompt_cache prompt_cache::cache_interaction_matrix::should_run_selected_pinned_ornith_cache_interaction_matrix_cell -- --ignored --nocapture --exact --test-threads=1`
  — runs one storage-transition cell; the cells are `fixed-live-reuse`,
  `fixed-worker-restart`, and `fixed-deleted-while-live`. Run all three.
- `ASTRONOMICAL_PROMPT_CACHE_ACCEPTANCE_PREFILL_CHUNCK_TOKENS=4096 cargo test ... prompt_cache_acceptance::engine_prompt_cache::should_restore_exact_cache_parity_for_one_selected_large_prefill_size -- --ignored --nocapture --exact --test-threads=1`
  — the same parity journey at the wider chunk; the cell accepts only 2048 or
  4096 and defaults to 2048.

## Building test binaries without running them

- `cargo test --release -p astronomical-acceptance --test memory_management_acceptance --features memory-management-acceptance --no-run`
  — compiles the memory acceptance binary and the production worker binary it
  spawns; useful before iterating on one journey.
