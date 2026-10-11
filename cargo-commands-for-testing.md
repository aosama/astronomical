# Cargo Commands for Testing

Concise catalog of the cargo commands that run this repository's Rust test
surface. Cargo commands are the interface, and every entry explains what it
runs and why it exists. Real-model journeys are invoked directly with
`--test-threads=1`; each test owns its timeout and cancellation/cleanup path.

## Hermetic and REST lanes (CPU only, safe to parallelize)

- `cargo test-hermetic` — runs every package's `hermetic_tests` binary. The
  fast, machine-independent verification lane; safe to run on any host.
- `cargo test-rest-api` — runs the REST contract and supervisor
  `rest_api_tests` binaries, exercising the public HTTP surface in-process.
- `cargo test-hermetic-and-rest` — both lanes together. This is the
  pre-commit verification gate alongside `cargo fmt --all -- --check`.
- `cargo test -p astronomical-model-serving --test hermetic_tests qwen3_5_moe_hermetic -- --test-threads=1`
  — the focused Qwen3.5 sparse-artifact lane for quantization configuration,
  native/affine single- and multi-layer planning, and tensor-profile contracts;
  it uses no model runtime or installed model files.
- `cargo test -p astronomical-model-serving --test hermetic_tests qwen3_5_import_direction -- --test-threads=1`
  — enforces the Qwen3.5 module dependency direction: both engines depend on
  shared core, never on each other, and attribution plus the family facade do
  not depend on engine internals.
- `cargo test -p astronomical-model-serving --test hermetic_tests hermetic::performance_attribution::measurement::should_record_an_external_operation_interval_only_when_attribution_is_enabled -- --exact --test-threads=1`
  — verifies that an operation measured outside the request-owning thread is
  recorded against the request report's monotonic start time.
- `cargo test -p astronomical-model-serving --features direct-mlx --test hermetic_tests prompt_processing_chunk_sizer -- --test-threads=1`
  — verifies that resident and streaming execution use separate chunk-sizer
  policies: resident uses fixed chunks, while streaming owns SSD chunk sizing
  and short-tail folding.
- `cargo test -p astronomical-model-serving --features direct-mlx --test hermetic_tests engine_backed_worker::chat::resident_streaming_retry -- --test-threads=1`
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

- `cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance -- --ignored --nocapture --test-threads=1`
  — the full memory-management journey suite: SSD-paged decode expert reuse,
  expert eviction, complete-residency control, prefill memory progress,
  reverse model swap, and the paging memory-shape experiment.
- `cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_admit_the_paged_moe_follow_up_turn_after_a_long_prefill -- --ignored --nocapture --exact --test-threads=1`
  — the paged-MoE follow-up admission regression journey: a long first turn
  teaches the RAM budget chunk-shaped activation evidence, and the follow-up
  turn in the same conversation must be admitted, never rejected with
  `generation context exceeds available GPU wired memory`.
- `cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance should_measure_paging_memory_shape_and_serving_rates_under_the_configured_ceiling -- --ignored --nocapture --exact --test-threads=1`
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
evidence to `tests/performance_throughput/throughput-history.jsonl`, and fails
when the serving budget is exhausted so partial evidence remains available.
Set `MEMORY_SWEEP_CELL_ORDER` to the cell's position in your sweep so records
carry it; the recommended protocol brackets the sweep with the 32 GB cell
first and last and interleaves the rest so page-cache warmth cannot masquerade
as a ceiling effect.

- `cargo test --release -p astronomical-inference-worker --features performance_throughput --test performance_throughput_tests performance_throughput::memory_ceiling_sweep::should_serve_the_large_sparse_moe_under_a_23gb_ceiling_and_record_the_memory_cell -- --ignored --nocapture --exact --test-threads=1`
- The same command with `..._28gb_...`, `..._32gb_...`, and `..._35gb_...`
  runs the remaining cells. Each journey uses a 2,048-token prefill chunk,
  a 500-token output target, a 500-input/50-output warmup, and stays within
  the 36 GB test ceiling. Partial rates and load/generation I/O evidence are
  recorded.

### Serving throughput (production-faithful rates)

- `cargo test --release -p astronomical-inference-worker --features performance_throughput --test performance_throughput_tests performance_throughput::qwen3_5_moe::should_measure_resident_sparse_moe_prompt_processing_and_decode_throughput -- --ignored --nocapture --exact --test-threads=1`
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

- `cargo test -p astronomical-model-serving --features direct-mlx --test serving_acceptance_tests <journey_filter> -- --ignored --test-threads=1`
  — the installed-artifact acceptance binaries (for example the Qwen-Image-2.1
  journeys gated on `ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY`).

### Inference-worker serving acceptance (dense-model journeys)

- `cargo test -p astronomical-inference-worker --features serving-acceptance --test serving_acceptance_tests should_admit_the_follow_up_turn_when_the_ssd_streaming_chunk_is_wider -- --ignored --nocapture --exact --test-threads=1`
  — the small dense Qwen3.5 variant of the follow-up admission regression: a
  wider SSD-streaming chunk is configured while the dense model never pages,
  so the resident mode's own operation scope must govern the activation
  reserve and both conversation turns must complete.

### Qwen3.5 artifact-derived engine selection and retry journeys

Each command below launches the production worker and makes a real public REST
request using a configured sparse artifact. The resident journey is bounded to
115 seconds; each streaming/retry SSD journey is bounded to 60 seconds. All
three tests are `#[serial]`, run with one test thread, and enforce a 36 GB
maximum configured active-memory ceiling.

- `cargo test --release -p astronomical-inference-worker --features serving-acceptance --test serving_acceptance_tests serving_acceptance::qwen3_5::automatic_engine_selection_rest::should_select_resident_engine_when_artifact_geometry_and_prompt_fit -- --ignored --nocapture --exact --test-threads=1`
  — verifies a request selects resident execution when validated artifact
  geometry plus prompt context fit the configured ceiling.
- `cargo test --release -p astronomical-inference-worker --features serving-acceptance --test serving_acceptance_tests serving_acceptance::qwen3_5::automatic_engine_selection_rest::should_select_streaming_engine_when_complete_residency_headroom_does_not_fit -- --ignored --nocapture --exact --test-threads=1`
  — verifies a request selects paging when the artifact's complete-residency
  headroom does not fit.
- `cargo test --release -p astronomical-inference-worker --features serving-acceptance --test serving_acceptance_tests serving_acceptance::qwen3_5::automatic_engine_selection_rest::should_restore_cached_prefix_after_an_invisible_resident_to_streaming_retry -- --ignored --nocapture --exact --test-threads=1`
  — verifies one invisible resident-to-streaming retry retains persistent
  prefix reuse and emits a nonzero retry-attribution interval.

### Prompt-cache acceptance journeys (resident sparse MoE role)

- `cargo test -p astronomical-model-serving --features direct-mlx --test prompt_cache_acceptance_tests -- --ignored --nocapture --test-threads=1`
  — the whole prompt-cache acceptance surface (cache disabled, cache-miss
  memory, restore peak, parity, block reporting, partial tail reuse, startup
  cleanup attribution, and the three vision journeys). Every one of these
  journeys runs the resident sparse MoE role (`resident_sparse_moe`) because
  they measure cache behaviour, not expert paging; the whole binary takes about
  three minutes. The one exception is the interaction matrix journey, which
  fails closed unless a cell is selected.
- `ASTRONOMICAL_PROMPT_CACHE_INTERACTION_ACCEPTANCE_CELL=<cell> cargo test -p astronomical-model-serving --features direct-mlx --test prompt_cache_acceptance_tests prompt_cache_acceptance::cache_interaction_matrix::should_run_selected_pinned_ornith_cache_interaction_matrix_cell -- --ignored --nocapture --exact --test-threads=1`
  — runs one storage-transition cell; the cells are `fixed-live-reuse`,
  `fixed-worker-restart`, and `fixed-deleted-while-live`. Run all three.
- `ASTRONOMICAL_PROMPT_CACHE_ACCEPTANCE_PREFILL_CHUNCK_TOKENS=4096 cargo test ... prompt_cache_acceptance::engine_prompt_cache::should_restore_exact_cache_parity_for_one_selected_large_prefill_size -- --ignored --nocapture --exact --test-threads=1`
  — the same parity journey at the wider chunk; the cell accepts only 2048 or
  4096 and defaults to 2048.

## Building test binaries without running them

- `cargo test --release -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance --no-run`
  — compiles the memory acceptance binary and the production worker binary it
  spawns; useful before iterating on one journey.
