# Expert Residency and Paging CPU, GPU, and SSD Execution Flow

This records the verified automatic Qwen3.5 and Qwen3.6 affine and native bfloat16 expert paths.

## Labels

- CPU means central processing unit host work in Rust or native C++.
- GPU means graphics processing unit work encoded through Metal.
- SSD means solid-state-drive storage.
- MLX means Machine Learning framework for Apple silicon.
- MTP means multi-token prediction.
- I/O means input/output work.
- ID means identifier.
- NAX is MLX's internal name for its Metal 4 matrix kernels.
- Graph construction creates lazy MLX arrays; it does not prove execution.

## Automatic mode selection

- Every sparse model retains one validated native pager as its fallback.
- After core model loading and allocator cleanup, admission compares fresh MLX active bytes plus the exact complete expert payload with the stable memory ceiling.
- `Resident` installs every target and optional multi-token-prediction expert layer as complete contiguous MLX arrays. `Paged` leaves the native cache demand-only.
- Selection uses artifact geometry and live memory, never model names or quantization labels.

## Ownership invariants

- Every sparse model keeps the validated pager for fallback, including while complete arrays are resident.
- `resident_expert_weights = Some` means every target and optional MTP sparse layer is resident. `None` means every sparse layer uses paging.
- Pager plan order is target decoder layers followed by the optional MTP layer. Both owners use those indices and the router's global expert IDs unchanged.
- The complete owner contains packed or native weights plus every required scale and bias. Its payload count excludes MLX allocator metadata and temporary graph work.
- Source descriptors remain open with the pager. Each promotion clones descriptors for one attempt, so failure cannot consume paging fallback ownership.

## Admission arithmetic

- Idle promotion projects `fresh active MLX bytes + exact complete expert payload bytes` against the stable ceiling.
- Resident request admission projects current active bytes, exact context reservation, direct cache-publication workspace when enabled, and any draft page reserve. Target page reserve is zero because the complete owner is already active.
- If that projection fails, admission demotes the complete owner and repeats from fresh active bytes with one largest target top-K expert page reserved.
- Persistent prompt-cache restoration repeats this mutable admission with one loaded-prefix key/value overlap before reading arrays, then re-admits only the remaining context after dropping loaded owners.
- Later prefill or decode growth uses exact persistent-state growth, exact temporary workspace, and a target page reserve only in paged mode. A multi-token resident prefill retries a smaller chunk before demotion; one-token work demotes and reprojects before page-level reclamation or rejection.
- No formula contains a model-name threshold or laptop-specific memory constant.

## Resident production flow

1. **CPU: materialize complete layers.** Rust clones validated source descriptors into transition-scoped safetensors maps, binds exact source data types and affine profiles, evaluates one complete layer at a time, and publishes the owner only after every layer succeeds.
2. **CPU: build common routing.** Router projection, softmax, top-K selection, sorting, score normalization, activation, weighted sum, and shared-expert combination remain common with paging.
3. **GPU: execute contiguous projections.** Standard MLX gathered dense or affine matrix multiplication consumes original global expert identifiers against complete gate, up, and down arrays.
4. **CPU and SSD: remain dormant.** Resident inference performs no native cache preparation, page publication, or expert source read.

## Paged production flow

1. **CPU: build routing graph.** Rust builds lazy router projection, softmax, top-K selection, score gathering, and normalization operations.
2. **Rust to C++: pass the route unchanged.** Rust gives the lazy selected-index MLX array to native analysis. Original indices remain available for projection and score alignment.
3. **GPU: reduce route evidence.** Native C++ builds a fixed bitmap. One thread per assignment atomically marks one expert bit and records an out-of-range value in a guard word.
4. **CPU and GPU: synchronize when needed.** An incomplete layer evaluates the bitmap once and reports exact distinct, missing, and payload counts before disk reads. A complete layer defers synchronization and retains the bitmap as pending recency evidence.
5. **CPU: admit exact misses.** Rust samples MLX memory after route evaluation, reserves exact missing payload plus the model-derived future-page reserve, and derives the route ceiling.
6. **CPU: commit cache policy.** Native C++ atomically installs the ceiling with route protection, updates recency, and evicts only the global deficit. It prefers entries above proportional layer floors, then global least-recently-used entries. Unused layer capacity is borrowable.
7. **CPU and SSD: fill missing slots.** Each missing expert receives one aligned MLX `PagedBufferSlot`. One batched `read_paged_buffer_ranges` call reads its gate, up, and down tensor ranges directly into the slot. The slot is committed before immutable typed views are created.
8. **CPU: publish immutable snapshots.** Changed layers rebuild metadata snapshots over retained slots. A snapshot contains no copied expert payload and keeps every referenced slot alive through lazy GPU execution.
9. **GPU: execute gathered projections.** Native affine or bfloat16 gathered matrix multiplication consumes original global expert IDs. Prompt processing uses sorted assignments; decode can use unsorted assignments. Affine execution applies MLX's common activation, scale, and bias output type without widening retained page storage.
10. **GPU: select the projection kernel.** Large sorted affine work uses NAX matrix kernels when available and all operands share its supported type. Other affine work uses generic matrix or gathered matrix-vector kernels that promote independently typed scale and bias values while loading compute tiles. Native bfloat16 uses paged gathered matrix kernels.
11. **GPU: combine outputs.** Existing activation, weighted-sum, shared-expert, and sparse/shared combination operations produce the next hidden state.

## Whole-model transitions

- Promotion synchronizes the model stream, freezes and empties native retention, clears reclaimable allocator storage, checks the exact complete payload, materializes a local candidate, then publishes `Resident` atomically.
- A non-fitting or source-validation-failed promotion resumes native retention and keeps mode `Paged`. Runtime cleanup failures remain fatal.
- Demotion synchronizes the model stream, drops the complete resident owner, clears allocator storage, then resumes native retention.
- Startup, request admission, prompt-cache restoration, later request pressure, request finalization, speculative-prefill draft loading, and live ceiling changes use these same boundaries. No partial resident model exists. The first finalization after request-driven demotion defers promotion to avoid same-request unload/reload churn.

| Trigger | Transition rule | Published state |
| --- | --- | --- |
| Startup | Clean allocator storage, then attempt exact complete-payload admission. | `Resident` when the candidate fits; otherwise `Paged`. |
| Initial request admission | Keep residency when the exact request fits; otherwise demote before request arrays are allocated. | Mode used by the first sparse forward. |
| Prompt-cache restoration | Demote before reading cached arrays when exact reconstruction overlap does not fit, then re-admit remaining context after reconstruction cleanup. | `Paged` during the pressured request. |
| Later request pressure | Retry smaller multi-token prefill work, then demote only when minimum work still requires it. | `Paged` for the remaining pressured request. |
| Request finalization | Drop request arrays, synchronize, and clear allocator storage. Skip the first promotion after request-driven demotion. | `Paged` without immediate full-payload reload, or the prior idle mode. |
| Ceiling decrease | Demote first when current resident active bytes exceed the requested ceiling, then reduce native retention before changing MLX limits. | Safe mode beneath the accepted ceiling. |
| Ceiling increase | Increase MLX and native capacity first, then attempt promotion. | `Resident` only after complete publication. |
| Request-scoped draft loading | Project current target active memory plus exact draft artifact payload. Preserve complete target residency when both fit; otherwise demote or reclaim only the required target payload. Recheck scoring workspace after loading. | After draft release, restore complete target residency before target execution whenever the live request state fits. |

Readiness, model-swap, memory-limit, generation-finalization, and idle telemetry publish the selected state directly. Consumers do not infer mode from byte totals.

## Route larger than retention capacity

The cache reuses routed hits, loads only missing routed experts, and returns an immutable ephemeral snapshot without admitting misses or displacing another layer’s floor. Ephemeral slots remain alive through lazy graphics-processor execution but never enter recency or retained-payload accounting.

## Cache policy

- Cache identity is `(layer index, expert ID)`.
- Only router-requested experts can enter the cache.
- Hits and insertions advance one monotonic access sequence.
- The in-flight route is protected during admission.
- One byte ceiling covers all layers. Proportional shares rounded to complete pages are eviction floors and preferences, not independent hard ceilings.
- A layer borrows unused global capacity. Global eviction prefers extras above layer floors, then uses access sequence, layer index, and expert ID.
- A route larger than its borrowable capacity executes ephemerally and cannot displace another layer's retained floor.
- Request pressure can freeze growth, lower the ceiling, reclaim exact bytes, and later resume retention. While frozen, a lower route-specific ceiling evicts immediately; only a higher ceiling is deferred. Resume installs the newest configured ceiling, and resident payload never remains above the active ceiling.
- The paged native cache never prewarms experts; only router-requested pages enter it.

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
- Native request reports count assignments, exact distinct and missing experts, selected and missing payload bytes, ceiling before and after commit, evicted payload bytes, hits, misses, disk pages, disk batches, successful source reads, source bytes, optional source-read elapsed time, publications, payload copies, and complete-layer synchronization elisions. A complete-layer elision reports zero missing bytes without synchronizing distinct-route evidence. Cumulative native statistics also count evictions.
- Native request reports separately count and time route-dependency synchronization. This wait can execute the preceding lazy projection and router graph, so it must not be labeled as cache-policy or solid-state-drive time.
- Source-read elapsed time is collected only when performance attribution is enabled.
- `native_paged_expert_projection_graph_count` reports native paged gathered projection graphs.
- `paged_moe_graph_construction` measures paged projection and combination graph construction, not GPU completion.
- `resident_moe_graph_construction` measures contiguous resident projection and combination graph construction.
- `resident_weight_materialization_synchronization_wait` covers attributed complete-weight materialization.
- Positional-read timing measures host file-service work; the operating-system page cache may satisfy it.

## Source map

- Common routing and mode selection: `crates/model-serving/src/qwen3_5_moe/model/forward.rs`.
- Resident projection execution: `crates/model-serving/src/qwen3_5_moe/model/resident_execution.rs`.
- Native projection orchestration: `crates/model-serving/src/qwen3_5_moe/model/paged_execution.rs`.
- Whole-model transitions: `crates/model-serving/src/qwen3_5_moe/model/expert_residency_transition.rs`.
- Resident ownership and loading: `crates/model-serving/src/qwen3_5_moe/expert_residency/`.
- Pager ownership and pressure integration: `crates/model-serving/src/qwen3_5_moe/expert_paging/expert_pager/native_cache.rs`.
- Source descriptor construction: `crates/model-serving/src/qwen3_5_moe/expert_paging/expert_pager_construction.rs`.
- Rust native ownership: `crates/runtime-integration/src/mlx_native_expert_cache.rs`.
- Product-neutral MLX-C wrappers: `crates/runtime-integration/src/mlx_paged_buffer_store.rs`.
- Native cache and policy: `crates/runtime-integration/native/expert_cache/`.
- Native projection kernels: `crates/runtime-integration/native/paged_expert_execution/`.
- MLX core paged storage: `third-party/patches/mlx-0.32.0-paged-buffer-store.patch`.
- MLX positional short-read fix: `third-party/patches/mlx-0.32.0-paged-buffer-short-read.patch`.
- MLX-C handles: `third-party/patches/mlx-c-0.6.0-paged-buffer-store.patch`.
