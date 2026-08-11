# Expert Paging CPU, GPU, and SSD Execution Flow

This records the verified demand-only Qwen3.5 and Qwen3.6 affine and native bfloat16 expert path.

## Labels

- CPU means central processing unit host work in Rust or native C++.
- GPU means graphics processing unit work encoded through Metal.
- SSD means solid-state-drive storage.
- MLX means Machine Learning framework for Apple silicon.
- I/O means input/output work.
- ID means identifier.
- NAX is MLX's internal name for its Metal 4 matrix kernels.
- Graph construction creates lazy MLX arrays; it does not prove execution.

## Production flow

1. **CPU: build routing graph.** Rust builds lazy router projection, softmax, top-K selection, score gathering, and normalization operations.
2. **CPU: reserve the possible route.** Before lazy route synchronization, Rust bounds distinct experts by route assignments and layer capacity, projects that payload against live MLX memory, and lowers optional retention when needed. The separate initial request reserve remains one model-derived top-K page.
3. **Rust to C++: pass the route unchanged.** Rust gives the lazy selected-index MLX array to the native cache. Original indices remain available for projection and score alignment.
4. **GPU: reduce route evidence.** Native C++ builds a fixed bitmap. One thread per assignment atomically marks one expert bit and records an out-of-range value in a guard word.
5. **CPU and GPU: synchronize when needed.** An incomplete layer evaluates and copies the bitmap once. Evaluation executes required router dependencies. A fully resident layer defers that synchronization and retains the lazy bitmap as pending recency evidence.
6. **CPU: apply cache policy.** Native C++ looks up `(layer index, expert ID)` entries, updates recency, and protects the routed set. It first evicts the current layer's oldest unprotected slots until that layer fits its proportional share, then prefers globally oldest entries from layers above their shares if the global ceiling still requires space. Pending complete-layer evidence is reconciled before eviction.
7. **CPU and SSD: fill missing slots.** Each missing expert receives one aligned MLX `PagedBufferSlot`. One batched `read_paged_buffer_ranges` call reads its gate, up, and down tensor ranges directly into the slot. The slot is committed before immutable typed views are created.
8. **CPU: publish immutable snapshots.** Changed layers rebuild metadata snapshots over retained slots. A snapshot contains no copied expert payload and keeps every referenced slot alive through lazy GPU execution.
9. **GPU: execute gathered projections.** Native affine or bfloat16 gathered matrix multiplication consumes original global expert IDs. Prompt processing uses sorted assignments; decode can use unsorted assignments. Affine execution applies MLX's common activation, scale, and bias output type without widening retained page storage.
10. **GPU: select the projection kernel.** Large sorted affine work uses NAX matrix kernels when available and all operands share its supported type. Other affine work uses generic matrix or gathered matrix-vector kernels that promote independently typed scale and bias values while loading compute tiles. Native bfloat16 uses paged gathered matrix kernels.
11. **GPU: combine outputs.** Existing activation, weighted-sum, shared-expert, and sparse/shared combination operations produce the next hidden state.

## Route larger than retention capacity

The cache reuses routed hits, loads only missing routed experts, and returns an immutable ephemeral snapshot without evicting retained pages. Ephemeral slots remain alive through lazy GPU execution but never enter recency or retained-payload accounting.

## Cache policy

- Cache identity is `(layer index, expert ID)`.
- Only router-requested experts can enter the cache.
- Hits and insertions advance one monotonic access sequence.
- The in-flight route is protected during admission.
- One byte ceiling covers all layers and is divided into proportional layer shares rounded to complete expert pages.
- Per-layer eviction protects each layer's decode working set. Global fallback prefers layers above their proportional shares, then uses access sequence, layer index, and expert ID.
- A route larger than its layer share executes ephemerally and cannot displace another layer's retained working set.
- Request pressure can freeze growth, lower the ceiling, reclaim exact bytes, and later resume retention. While frozen, a lower route-specific ceiling evicts immediately; only a higher ceiling is deferred. Resume installs the newest configured ceiling, and resident payload never remains above the active ceiling.
- Startup, cleanup, and memory-limit changes never prewarm experts.

## Storage and validation

- MLX core owns paged-buffer allocation, direct range reads, commit state, immutable typed views, and shared-buffer lifetime.
- MLX-C exposes product-neutral opaque paged-slot and file-reader handles for direct Rust boundary tests.
- Astronomical C++ owns model tensor geometry, layer-balanced recency, retention, pressure policy, snapshot publication, and gathered projection execution.
- Construction validates source files, intervals, data types, shapes, packed geometry, affine widths, group sizes, and layer compatibility.
- Source reads fail on truncation or any incomplete positional read.
- Supported affine widths are 2, 3, 4, 5, 6, and 8 bits. Supported group sizes are 32, 64, and 128. Gate, up, and down projections retain independent affine profiles. Scale and bias types may independently be float16, bfloat16, or float32.
- Native bfloat16 experts contain uncompressed gate, up, and down weights without affine companions.

## Attribution

- `native_expert_cache_route_preparation` covers route evaluation, cache policy, source reads, and snapshot preparation.
- Native request reports count hits, misses, disk pages, disk batches, successful source reads, source bytes, optional source-read elapsed time, publications, payload copies, and complete-layer synchronization elisions. Cumulative native statistics also count evictions.
- Native request reports separately count and time route-dependency synchronization. This wait can execute the preceding lazy projection and router graph, so it must not be labeled as cache-policy or solid-state-drive time.
- Source-read elapsed time is collected only when performance attribution is enabled.
- `native_paged_expert_projection_graph_count` reports native paged gathered projection graphs.
- `paged_moe_graph_construction` measures paged projection and combination graph construction, not GPU completion.
- Positional-read timing measures host file-service work; the operating-system page cache may satisfy it.

## Source map

- Routing and native preparation: `crates/model-serving/src/qwen3_5_moe/model/paged_forward.rs`.
- Native projection orchestration: `crates/model-serving/src/qwen3_5_moe/model/paged_execution.rs`.
- Pager ownership and pressure integration: `crates/model-serving/src/qwen3_5_moe/expert_paging/expert_pager/native_cache.rs`.
- Source descriptor construction: `crates/model-serving/src/qwen3_5_moe/expert_paging/expert_pager_construction.rs`.
- Rust native ownership: `crates/runtime-integration/src/mlx_native_expert_cache.rs`.
- Product-neutral MLX-C wrappers: `crates/runtime-integration/src/mlx_paged_buffer_store.rs`.
- Native cache and policy: `crates/runtime-integration/native/expert_cache/`.
- Native projection kernels: `crates/runtime-integration/native/paged_expert_execution/`.
- MLX core paged storage: `third-party/patches/mlx-0.32.0-paged-buffer-store.patch`.
- MLX positional short-read fix: `third-party/patches/mlx-0.32.0-paged-buffer-short-read.patch`.
- MLX-C handles: `third-party/patches/mlx-c-0.6.0-paged-buffer-store.patch`.
