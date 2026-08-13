# Expert streaming execution flow

## Ownership

- The model-serving `memory` package owns cross-component fit, reclamation, residency, retry, and ceiling decisions from typed requirements and Machine Learning framework for Apple silicon observations.
- Rust paging owns expert selection, bounded SafeTensors manifests, source reads, temporary layer owners, and execution of typed memory decisions.
- Runtime integration exposes ordinary Machine Learning framework for Apple silicon arrays and operations through Machine Learning framework for Apple silicon C.
- Production expert streaming has no custom C++ page store, page table, paged-buffer slot, snapshot, or retirement protocol.

Dependency direction:

`model-serving Rust streamer -> runtime-integration Machine Learning framework for Apple silicon C wrapper -> Machine Learning framework for Apple silicon arrays and gathered matrix multiplication`

## Resident fast path

When every sparse expert fits with activation headroom, the model owns complete contiguous expert arrays. Gate, up, and down projections use global expert identifiers directly with `gather_qmm` or `gather_mm`.

## Multi-token prefill

1. The router produces global expert identifiers and scores.
2. Rust selects every expert in the current sparse layer.
3. Rust builds exact bounded SafeTensors intervals from startup-validated tensor geometry.
4. The bounded reader creates ordinary compact Machine Learning framework for Apple silicon arrays for that complete layer.
5. Gate, up, and down use ordinary gathered matrix multiplication.
6. The temporary Rust owner drops after the layer output has entered the forward graph. Machine Learning framework for Apple silicon retains only dependencies required by lazy execution.
7. The next decoder layer repeats the same process.

Each layer executes once. There is no missing-route replay, cross-layer expert cache, or reusable native streaming slot.

## One-token decode

1. After multi-token prefill finishes, `MlxRamBudget` may admit the largest exact decode-warm complete-layer prefix under `retained_expert_budget_bytes`; no second fixed byte cap is applied.
2. If the current layer is warm, decode reuses that complete layer with ordinary gathered matrix multiplication.
3. Otherwise Rust evaluates and deduplicates the top-K routed expert identifiers, builds bounded SafeTensors intervals only for that route, and executes the same gathered matrix operations.
4. Multi-token prefill reuses warm complete layers that still fit `retained_expert_budget_bytes`. Only live budget pressure shrinks the warm prefix; prefill start must not zero the retention ceiling.

## Memory behavior

- `maximum_mlx_memory_gb` remains the only user memory limit.
- `MlxRamBudget` is the single source of truth for the split among `model_core_payload_bytes`, `context_window_reserve_bytes`, `activation_headroom_bytes`, `complete_layer_stream_slot_bytes`, and `retained_expert_budget_bytes`.
- Allocation, context, speculative-prefill, complete-residency, recovery, and live-ceiling policies live beside `MlxRamBudget` under `model-serving/src/memory`; Qwen and paging modules do not recompute those decisions.
- `context_window_reserve_bytes` starts at 1 GB SI and rises from live measurements; multi-token prefill keeps `complete_layer_stream_slot_bytes` and does not grow retained complete layers while activations are large.
- Initial admission reserves the model-derived largest complete expert layer because prefill streams one complete layer.
- Bounded loading reports the exact pending page to the memory package before construction, then executes admit, allocator-cleanup, or reject advice.
- Per-prefill-chunk synchronization and allocator cleanup release temporary layer storage before the next configured chunk.
- Request pressure can demote the complete resident owner. Rust-streamed pages need no explicit retention reclamation because they are operation-local.

## Performance attribution

When enabled, record:

- `rust_expert_streaming_layer_preparation` elapsed time and occurrences;
- `rust_expert_streaming_payload_byte_count`;
- generic positional file-read calls, bytes, elapsed time, concurrency, and failures;
- `rust_streamed_expert_projection_graph_count`;
- router, multilayer perceptron, graphics-processor completion, and allocator-cleanup spans;
- active, cache, and peak Machine Learning framework for Apple silicon memory.

The prior custom C++ page-store counters are retired because that implementation is no longer built or linked.
