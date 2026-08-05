# Expert Paging CPU, GPU, and SSD Execution Flow

This document records the verified Qwen3.5 and Qwen3.6 mixture-of-experts paging path. It distinguishes host control work, graphics-processor execution, solid-state-drive reads, and expert-weight memory-cache least-recently-used policy. Every placement below is derived from the cited implementation; timing attribution alone is not used to infer where work executes.

## Label contract

- CPU means Rust or C++ code executing on a host thread.
- GPU means an MLX primitive encoded for the Metal graphics-processor stream and executed by the Apple graphics processor.
- SSD and Metal I/O means a Metal I/O command reads file ranges into Metal buffers. The CPU constructs and submits the command; the CPU does not copy the payload bytes through a Rust buffer.
- SSD and MLX I/O worker means an MLX lazy Load primitive performs bounded positional file reads through MLX input/output workers when the graph is evaluated.
- Graph construction on the CPU creates lazy MLX arrays. It does not prove that the represented arithmetic has executed.
- A synchronous MLX evaluation wait includes host dependency traversal and submission plus waiting for required device work. It does not identify one graphics-processor kernel as the elapsed-time owner.

## Verified flow

    CPU: Astronomical inference engine thread
        |
        | Constructs a lazy router graph on the MLX GPU stream:
        | - quantized linear or matrix multiplication
        | - softmax
        | - argpartition
        | - selected-index slice
        | - selected-score gather and optional normalization
        |
        | No router arithmetic is guaranteed to have executed here.
        v
    CPU: Complete-layer expert-weight memory-cache lookup
        |
        | Looks up the current layer by layer index.
        | A hit increments the access sequence and updates that layer's
        | least-recently-used position.
        |
        +---- COMPLETE-LAYER HIT -----------------------------------------+
        |                                                                 |
        | No selected-expert ID copy to CPU is required.                  |
        | No expert-weight SSD read is required.                          |
        | Lazy selected indices remain MLX arrays and are passed          |
        | directly into complete-layer mixture-of-experts graph building. |
        |                                                                 v
        |                                                GPU: Later MLX evaluation
        |                                                    |
        |                                                    | Executes router and
        |                                                    | complete-layer gathered
        |                                                    | expert calculations.
        |                                                    v
        |                                                GPU: Next hidden state
        |
        +---- COMPLETE-LAYER MISS ----------------------------------------+
        |
        | Constructs a lazy contiguous selected-expert ID array.
        | Calls the synchronous MLX-C API mlx_array_eval.
        v
    CPU: MLX-C and MLX core scheduler
        |
        | mlx_array_eval calls mlx::core::array::eval.
        | MLX traverses every unscheduled dependency needed to produce
        | the contiguous selected-expert ID array.
        | MLX encodes GPU primitives and commits the relevant Metal work.
        v
    GPU: Apple graphics processor
        |
        | Executes the required dependency chain for the current hidden state.
        | Executes router projection, softmax, argpartition, slicing,
        | selected-score operations, and the contiguous selected-ID copy.
        | Signals the MLX Metal shared event after required work completes.
        v
    CPU: Astronomical inference engine thread
        |
        | Waits through MLX on the Metal shared event.
        | Resumes only when the selected IDs are readable.
        | Reads the evaluated UInt32 values from MLX unified memory.
        | Copies those small values into a Rust vector.
        | Sorts and deduplicates the selected global expert IDs.
        v
    CPU: Expert page policy
        |
        | Builds the exact selected-page manifest.
        | Chooses one of the verified branches below according to execution
        | mode, retained capacity, live memory budget, and storage availability.
        |
        +---- RETAINED ONE-EXPERT MEMORY-CACHE PATH ----------------------+
        |                                                                 |
        | CPU: Looks up each selected expert by layer and expert ID.       |
        |      A hit increments the access sequence and updates that       |
        |      expert's least-recently-used position.                      |
        |                                                                 |
        |      Before admitting missing experts, protects the currently   |
        |      selected IDs and removes oldest unselected experts until   |
        |      the incoming retained payload fits.                        |
        |                                                                 |
        |      Cache hit: reuses retained MLX expert arrays; no SSD read.  |
        |                                                                 |
        |      Cache miss: constructs lazy bounded safetensors Load arrays |
        |      and separates those lazy arrays into independently owned    |
        |      one-expert page owners retained by the memory cache.        |
        |                                                                 v
        | SSD and MLX I/O workers: During later MLX evaluation, perform    |
        |      bounded positional reads for missing expert tensor ranges. |
        |                                                                 |
        +---- TEMPORARY OR DIRECT PAGE PATH ------------------------------+
        |                                                                 |
        | CPU: Builds the selected-page manifest and samples the live      |
        |      MLX memory budget. If necessary, reconciles retained        |
        |      expert pages before admitting the temporary page.          |
        |                                                                 |
        |      If the layer has a validated aligned expert pack:          |
        |      - computes source file offsets                              |
        |      - computes destination tensor offsets                       |
        |      - computes exact byte counts                                |
        |      - merges adjacent sorted expert IDs into one range per tensor |
        |      - allocates MLX-owned Metal destination buffers              |
        |      - calls astronomical_metal_expert_loader_start              |
        |                                                                 v
        | CPU: Native Metal I/O command encoding                           |
        |      - opens an MTL::IOFileHandle                                |
        |      - obtains an MTL::IOCommandBuffer                           |
        |      - encodes MTL::IOCommandBuffer::loadBuffer for each range   |
        |      - encodes a completion-event signal                         |
        |      - commits the Metal I/O command buffer                      |
        |      - makes the target GPU stream wait on that event            |
        |                                                                 v
        | File storage and Metal I/O subsystem                             |
        |      Reads exact ranges from the aligned expert-pack file.       |
        |      Transfers those bytes into MLX-owned MTL::Buffer objects.   |
        |      Signals the completion event.                               |
        |                                                                 |
        |      If no aligned pack is active, the direct path instead       |
        |      constructs lazy bounded safetensors Load arrays, whose      |
        |      positional file reads occur on MLX I/O workers during       |
        |      later graph evaluation.                                     |
        |                                                                 |
        +-----------------------------------------------------------------+
        |
        v
    CPU: Paged mixture-of-experts graph construction
        |
        | Retains only the sorted unique host IDs required to select files
        | and retained expert owners. The assignment-sized selected-index
        | array remains in MLX.
        | Verifies that those sorted unique routed IDs exactly match the
        | compact page manifest before graph construction.
        | The manifest stores a dense fixed-capacity lookup from every global
        | expert ID to its compact page slot. IDs absent from the page contain
        | a UInt32 maximum sentinel that validated production routing cannot use.
        | Creates an MLX array from that lookup for the current page execution.
        | Builds a lazy MLX take_axis operation from the lookup array and the
        | existing device-side selected indices. The output preserves the
        | selected-index shape while replacing global IDs with compact slots.
        | Builds gathered quantized matrix multiplication, expert activation,
        | routing-score weighting, and sparse/shared combination operations.
        | These remain lazy MLX operations at construction time.
        v
    GPU: Apple graphics processor
        |
        | The Metal compute stream waits for any required Metal I/O event.
        | Executes take_axis to remap each global expert assignment to its
        | compact page slot without constructing remapped assignments in Rust.
        | No second selected-index synchronization or assignment-sized host
        | upload occurs for page-slot remapping.
        | Executes gathered expert matrix operations and combination graphs.
        v
    GPU: Next hidden state

## Least-recently-used policy location

The expert-weight memory-cache least-recently-used policy executes on the CPU. It uses monotonically increasing access sequence numbers rather than clock reads.

- A complete-layer hit updates the complete layer's last access sequence.
- A retained one-expert hit updates that expert's last access sequence.
- Insertion counts as an access.
- Selected experts are protected while making room for an incoming page.
- Global partial-page eviction selects the unprotected expert with the smallest last access sequence, then uses layer index and expert ID as deterministic tie-breakers.
- Same-layer selected-page admission selects the oldest unselected expert in that layer, using expert ID as the deterministic tie-breaker.
- Complete-layer prewarm preserves the exact one-token route payload for every layer that remains paged when those route floors fit the live retention ceiling.
- Decode reconciles existing complete layers against those route floors before selected-route admission. After a demotion, it synchronizes the graphics-processor stream, clears released allocator storage once, and resamples the live budget.
- Direct multi-token prompt processing admits complete layers against physical retained capacity without reserving decode-route payload.
- Temporary direct-page admission can reconcile retained pages when the live memory budget is insufficient.

The policy does not read expert payload bytes and does not execute mixture-of-experts arithmetic.

## File-read locations

There are two distinct file-read mechanisms.

### Aligned expert-pack Metal I/O

The CPU calculates exact ranges and submits MTL::IOCommandBuffer::loadBuffer commands. Adjacent sorted expert IDs share one range per tensor; scattered IDs remain separate so no unselected gap is read. Metal I/O transfers file bytes into MLX-owned Metal buffers. The target GPU compute stream waits on the Metal I/O completion event before consuming those buffers. There is no intermediate Rust payload-byte copy.

### Bounded safetensors MLX loading

The CPU constructs lazy MLX Load arrays backed by bounded positional readers. Host positional reads occur later on MLX input/output workers when evaluation requires the arrays. The operating-system page cache may satisfy them, so these timings do not prove physical solid-state-drive service. This is the fallback when no aligned pack is active and is also used by the retained one-expert population path.

## Attribution interpretation

The selected_expert_id_evaluation_synchronization_wait operation wraps the synchronous evaluation call that makes selected IDs readable to the CPU. Its elapsed time can include:

- CPU dependency-graph traversal and Metal command encoding.
- Required graphics-processor work inherited through the current hidden state.
- Router and selected-ID graphics-processor operations.
- Waiting for the MLX Metal shared event.

It does not include the separately measured host copy of evaluated selected IDs. It also does not prove how much of the wait belongs to any individual graphics-processor kernel.

The aligned_expert_pack_metal_io_page_load operation wraps aligned-page construction and Metal I/O submission. The native loader records completion separately, and the GPU dependency is enforced by a shared event. Do not interpret the operation name alone as pure physical SSD service time.

## Verified source map

- Router graph and complete-layer branch: crates/model-serving/src/qwen3_5_moe/model/paged_moe_forward.rs, lines 47-94.
- Selected-ID evaluation, host copy, sorting, and deduplication: crates/model-serving/src/qwen3_5_moe/model/paged_moe_execution.rs, lines 217-243.
- Router operations: crates/model-serving/src/qwen3_5_moe/model/moe.rs, lines 35-67.
- Complete-layer and one-expert access sequence updates: crates/model-serving/src/qwen3_5_moe/expert_paging/expert_cache.rs, lines 90-117.
- Complete-layer admission and partial route-floor allocation: crates/model-serving/src/qwen3_5_moe/expert_paging/expert_cache_capacity.rs.
- One-expert lookup, protection, eviction, loading, and assembly: crates/model-serving/src/qwen3_5_moe/expert_paging/expert_pager/memory_cache.rs, lines 39-262.
- Global partial-page eviction: crates/model-serving/src/qwen3_5_moe/expert_paging/expert_cache_eviction.rs, lines 4-64.
- Same-layer deterministic oldest-unselected eviction: crates/model-serving/src/qwen3_5_moe/expert_paging/expert_cache.rs, lines 258-287.
- Direct-page memory admission and aligned-versus-safetensors branch: crates/model-serving/src/qwen3_5_moe/expert_paging/expert_pager/direct_page.rs, lines 38-138.
- Aligned range calculation, adjacent-run coalescing, and runtime call: crates/model-serving/src/qwen3_5_moe/expert_paging/aligned_expert_pack_loader.rs, lines 100-247.
- Rust-to-native Metal I/O boundary: crates/runtime-integration/src/mlx_metal_expert_pack_loader.rs, lines 153-253.
- Native Metal I/O file handle, loadBuffer commands, completion event, commit, and GPU-stream wait: crates/runtime-integration/native/astronomical_metal_expert_loader.cpp, lines 238-339.
- Rust synchronous MLX-C evaluation call: crates/runtime-integration/src/mlx_array.rs, lines 217-223.
- MLX-C array evaluation boundary: mlx-c/mlx/c/array.cpp, lines 348-355 in the pinned upstream source.
- MLX dependency traversal and synchronous wait: mlx/mlx/transforms.cpp, lines 80-350 in the pinned upstream source.
- MLX Metal shared-event host wait: mlx/mlx/backend/metal/event.cpp, lines 28-31 and 55-72 in the pinned upstream source.
- Dense global-expert-ID to page-slot manifest construction: crates/model-serving/src/qwen3_5_moe/expert_paging/quantized_expert_manifest.rs, lines 239-323.
- Complete-layer and compact-page mixture-of-experts graph construction plus device-side page-slot remapping: crates/model-serving/src/qwen3_5_moe/model/paged_moe_execution.rs, lines 20-253.
