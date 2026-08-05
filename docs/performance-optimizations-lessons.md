# Performance Optimization Lessons

## First principles

- Rust and C++ are not automatically faster than Python. Python only submits graphs to Apple’s MLX array framework; graph shape and selected Metal kernels dominate.
- Compare the exact model implementation selected by model_type. Qwen3.5-MoE uses qwen3_5_moe, not the similar qwen3_next model.
- Match Machine Learning framework for Apple silicon (MLX) version, model files, prompt, tokenizer, sampling, prefill_chunck_tokens, cache state, and build profile before comparing.
- Measure long-context parity after the runtime chat template is rendered, and pass the same thinking-mode setting to both the prompt renderer and output parser; counting only source text or disabling thinking in metadata produces unlike workloads.
- Source-matched sanitize behavior is part of model arithmetic. Stock MLX-LM 0.31.3 double-shifts already-converted Qwen3.5 and Qwen3.6 normalization weights when MTP tensors are present, corrupting target-only output; apply the checkpoint producer's sanitize correction while keeping MTP construction disabled.
- Use release builds for performance. Debug builds remain useful for correctness only.

## Mixture-of-experts routing

- Unsorted expert assignments cause scattered expert-weight reads during gathered quantized matrix multiplication.
- For at least 64 assignments, flatten assignments, sort by expert, gather token rows in that order, and pass sorted_indices=true to all three expert projections.
- Reuse one sort for the gate, up, and down projections.
- Compute the inverse sort once, then use a Metal kernel to apply router scores directly to sorted outputs. Restoring [batch, tokens, top-k, hidden] first creates an expensive transient tensor.
- Keep small decode batches unsorted; sorting eight assignments costs more than it saves.
- Native BF16 paged experts should retain their one-weight source layout through bounded reads, page retention, and GPU selection. Converting them to affine parameters or copying selected weights to the host defeats paging and changes precision.
- Do not implement selected native BF16 experts as take_axis(weights, assignment_indices) followed by batched matrix multiplication. That materializes one complete expert matrix per token assignment; a moderate prompt can request a tens-of-gigabytes temporary Metal buffer. Use MLX gather_mm, which reads selected matrix batches inside the multiplication kernel without expanding weights.
- Retain the sorted-expert weighted-sum Metal kernel with the loaded model. Recreating its owner for every layer repeats avoidable graph-configuration work even when the system pipeline cache is warm.
- Size request admission for the largest complete expert-layer page derived from every tensor source and its own shard capacity. A decode-only experts-per-token estimate under-reserves prompt processing, causing late layer failures after context allocation; a layer-wide multiplier over shard-sliced sources over-reserves by counting experts repeatedly.
- Validate this with a controlled local mutation. Disabling only sorting reduced measured prefill from about 979 to 444 tokens per second.

## Custom recurrent Metal kernel

- A token-by-token recurrent loop built from generic MLX operations creates excessive graph nodes and kernel launches.
- Fuse the complete gated-delta sequence recurrence into one Metal kernel while keeping recurrent state in 32-bit floating point.
- Construct and retain the kernel object once with the loaded model. Never recreate it per layer, chunk, or token.
- Match upstream grid, threadgroup, pointer arithmetic, data types, state layout, and grouped-head mapping exactly.
- Keep an operation-based implementation as a numerical reference test, not as the production path.

## Runtime T versus template constants

- T is the current sequence token count.
- Pass T as a zero-dimensional MLX integer array at runtime.
- Do not make T a Metal template argument. That compiles a separate kernel variant for every chunk length, including the final partial chunk.
- Template only stable specialization values: input type, state type, head counts, and head dimensions.
- A runtime scalar preserves one compiled kernel across configured prefill chunks, partial chunks, and one-token decode.

## Metal compilation and dispatch

- Ahead-of-time and just-in-time Metal builds can contain different kernel families; this is not merely a startup-versus-binary-size choice.
- The prior ahead-of-time build targeted macOS 14. MLX therefore defined MLX_METAL_NO_NAX and excluded its Metal 4 matrix kernels.
- MLX source uses nax as an internal label for those Metal 4 kernels.
- Enabling just-in-time Metal compilation lets MLX inspect the operating system and Apple graphics-processor generation, then select the fast kernels or a compatible fallback.
- Read CMake branches and preprocessor definitions. A build flag can silently change runtime dispatch while preserving correct output.
- First-use compilation can make a correct fast path appear slow. MLX stores compiled Metal pipelines in the system cache and reuses them across processes and reboots.
- Kernel name, source, template values, compile options, and target architecture affect cache identity.
- Report cold and warm kernel-cache results separately.
- For 256-wide attention heads on Metal 4 matrix hardware, split each head across two single-instruction multiple-data groups and use block matrix operations.
- Maintain online-softmax state per block instead of materializing the quadratic attention-score tensor.
- Gate specialized attention by hardware, data type, mask, and measured shape. Retain the unfused graph for unsupported shapes.
- Before adding a custom decode split, inspect Machine Learning framework for Apple silicon (MLX) fused scaled dot-product attention (SDPA): for one to eight query tokens and sufficiently long grouped-query key/value state, it already selects a two-pass partitioned Metal kernel. Measure the active dispatch before duplicating it with a paged-attention layer.
- Preserve MLX automatic key/value partitioning for long grouped-query decode. A one-partition control removes useful graphics-processor parallelism, so a custom paged-attention layer must prove an additional end-to-end benefit rather than replicate MLX splitting.
- Quantized matrix multiplication split-K dispatch can change bfloat16 reduction order when output row count changes. Once row work already exposes enough threadgroups, cap the row-count contribution to split-K selection so equivalent prompt partitions retain one reduction topology and exact recurrent state.

## Compiled graph composites

- Python decorators in the MLX language-model toolkit still change native graph execution. The equivalent C path is an owned mlx_closure passed through mlx_compile.
- Compile only composites isolated by controlled reference mutations. For Qwen3.5-MoE decode, shapeless native-dtype SwiGLU fusion matters; compiling sampling, precise gated normalization, or decay arithmetic did not.
- Retain one shapeless compiled SwiGLU with the model and reuse it across expert layers and prompt or decode shapes.
- Keep mixture-of-experts SwiGLU in the activation dtype. Reserve 32-bit floating-point casts for the gated normalization that requires them; unnecessary precision adds conversion work and can inhibit the intended fused path.
- mlx_closure_new intentionally returns a null-context output placeholder. Accept it only before mlx_compile, then require compilation to populate it before application.
- A Rust callback invoked through C must own every temporary MLX handle, return status codes, and never unwind across the foreign-function boundary.
- Benchmark compiled composites separately for multi-token prefill and one-token decode. Full-attention output gating, gated-delta decay arithmetic, and sparse/shared expert combination can favor prompt processing, while the shapeless precise SwiGLU graph also improves one-token decode.
- Prefer one stable MLX primitive such as logaddexp(x, 0) over manually composing equivalent softplus arithmetic from several operations. This reduces graph construction and kernel launches while retaining the upstream numerical path.
- Materialize model-invariant scalar arrays once at model load instead of recreating and casting them in every forward pass.
- Compare independently assembled bfloat16 inference paths with one representable logit-step tolerance plus identical greedy-token checks. Requiring bit-identical logits can force unnecessary 32-bit floating-point work despite equivalent model decisions.
- Shapeless compilation does not make input-derived static constants dynamic. Use rank-zero broadcast constants; an input-shaped zero froze gated-delta decay to the first prefill length and broke later tail chunks.
- Submit dependent forward graphs asynchronously in bounded layer groups for prompt processing and one-token generation. This overlaps Rust graph construction with graphics-processor evaluation without synchronizing every layer; measure the interval because excessive submissions add scheduler overhead.
- Give MLX evaluation only independently branching graph roots. State arrays already reachable from the requested output, including sibling outputs of one primitive, are evaluated and detached during the same dependency traversal; listing them again adds host traversal work. Keep explicit roots for state that branches away from the output graph.
- Treat mamba_ssm_dtype: float32 as a decay-computation contract, not a disk-storage contract. Qwen3.5-family artifacts can store A_log and other model parameters as FP16, BF16, or FP32 independently from activation dtype. Retain source storage without conversion and promote only the decay operation that requires FP32 arithmetic; eager whole-model promotion increases residency and changes the artifact’s precision contract.
- Keep OptiQ metadata validation aligned with MLX affine quantization: bit widths 2, 3, 4, 5, 6, and 8 and group sizes 32, 64, and 128 are supported. Metadata may include embedding and output-head measurements; when present, verify them against config instead of rejecting them as unexpected.

## Cache types are different

- Do not confuse model state, the MLX allocator cache, and the system Metal pipeline cache.
- Model state contains full-attention key-value tensors, linear-attention recurrent state, and convolution state. Clearing it changes generation.
- The MLX allocator cache holds reusable memory. Clearing it does not clear model context, but retaining too much can trigger a long-context throughput cliff.
- The macOS process wired-memory limit is separate from MLX active-memory guidance and allocator-cache limits. Set it to the machine-reported recommended working set before inference; otherwise long-context evaluation can throttle sharply while MLX active and cached bytes still look healthy.
- Upstream MLX documents its active-memory limit as graph-evaluation guidance. Astronomical’s pinned allocator must enforce the configured ceiling plus a one-percent transient allowance before converting any new, reused, or host-backed Metal buffer to active ownership; application admission must still keep completed stable memory at or below the configured ceiling.
- The system Metal pipeline cache holds compiled kernels. It explained a large cold-versus-warm difference after enabling fast Metal dispatch.
- Evaluating model cache arrays after each prefill chunk severs lazy graphs and prevents prior chunks from remaining live.
- Intermediate prefill chunks need only materialized decoder state. Do not force final logits until the final prompt token; doing so evaluates output work that cannot affect the next chunk.
- Clear reclaimable MLX allocator memory after every evaluated prefill chunk. Retaining intermediates caused allocator cache growth from zero to 20 gigabytes and a sharp long-context throughput collapse despite stable live-model memory.
- After dropping one model, initialize the replacement runtime and clear reclaimable MLX allocator memory before loading replacement weights. MLX recycles freed model buffers process-wide; leaving them resident can suppress first-request expert retention and contaminate mode comparisons.
- For the pinned MLX v0.32.0 Metal allocator, active excludes reclaimable allocator-cache buffers and clear_cache() frees only those unused buffers. A live array remains active after cache clearing; assert relational transitions rather than laptop-specific byte totals.
- MLX graph construction can allocate tiny bookkeeping buffers before evaluation. Evaluation is the boundary that creates the model-scale active residency, so allocator contracts must compare the final payload transition rather than require every construction counter to remain literally unchanged.
- MLX active memory has no allocator-level ownership tags. Attribute model core, retained experts, and request context from their array owners, then assign the remainder of the measured active total to runtime work. Logical array views can share physical backing storage, so reconcile each category against the remaining measured active bytes to prevent double-counting and guarantee that the breakdown sums to the allocator total.
- set_cache_limit(0) prevents freed-buffer retention. Lowering a nonzero cache limit does not reclaim immediately; the pinned allocator reclaims excess cache during a later allocation. A same-shaped allocation can reuse cached capacity instead of creating another Metal buffer.
- The explicit scripts/test-mlx-memory-contracts.sh qualification runs a native C++ probe against the archived patched MLX source, Rust MLX-C boundary checks, and the real Qwen3.6-35B-A3B-8bit to Qwen3.6-35B-A3B-oQ4e-mtp lifecycle. It is serial because MLX allocator policy is process-global.
- An uncached request cleans the MLX allocator after each prompt prefill chunk and again at request finalization; attribute both phases instead of assuming one cleanup per request.
- Keep direct-runtime, real-model, and persistent-cache test roots separate so a local edit does not relink unrelated native test suites.
- A completed model load can retain tiny graph bookkeeping after materialization. Compare replacement cache with prior stale cache and live model residency, not literal zero.
- Request finalization samples immediately after synchronized allocator cleanup. Require zero allocator-cache bytes while live model residency remains nonzero.
- Generation attribution is request-scoped. Enabling model-loading attribution on the engine does not emit a generation report; attach enabled attribution to each Qwen3_5InferenceRequest. On maximum-token completion, the engine synchronously finalizes and clears request memory while returning the final TokenId; a further decode is invalid.
- Lazy MLX execution can charge graphics-processor work to a later synchronization boundary rather than the graph-building call that submitted it. Record non-overlapping request-phase spans alongside attention, multilayer-perceptron, input/output, and synchronization leaf timings; exclude overlapping spans from the leaf-duration sum so a slow phase is located without double counting.
- A slice may retain a large backing allocation. Copy small long-lived convolution state into contiguous storage, as upstream MLX does.
- Full-attention key-value storage should grow by a visible configured step, defaulting to 256 tokens, update only the new slice, and expose only the used prefix to attention. Per-token concatenation causes quadratic copying and device starvation.
- Persistent decoder-cache sequence-state files must save only tensors with a declared token axis; fixed or circular state must store the exact block-boundary snapshot separately. Do not fake a one-token rollback for recurrent state.
- Enforce a global SSD prompt-cache quota with one recursive metadata scan and one oldest-written sort per enforcement, then subtract each successful deletion from the scanned totals. Rescanning after every eviction turns cleanup into quadratic filesystem work.
- Serialize persistent prompt-cache tensors while they remain on the Machine Learning framework for Apple silicon owner thread, then transfer only bounded byte vectors to a filesystem thread. The default writer should use the device and operating-system throughput rather than an arbitrary fixed rate; retain an explicit user cap for machines that need one. A bounded queue must drop opportunistic captures when full rather than retain model arrays, grow memory without limit, or block generation on filesystem I/O.
- Keep sampled tokens as MLX arrays. Submit one token asynchronously, build and submit its dependent successor, then read the first token on the host.
- Before request-finalization allocator cleanup, synchronize the model-owned MLX graphics-processor stream. One-token-ahead decode work can still be in flight after request arrays are dropped, so clearing allocator memory without that boundary can race buffer completion.
- Apply the same synchronization before per-prefill-chunk allocator cleanup. MLX Metal allocator cleanup releases cached buffers without synchronizing its graphics-processor stream, while prompt processing can still have dependent Metal input/output and evaluation work submitted on that stream.

## Spectral attention experiments

- Do not qualify a spectral attention replacement with a conventional checkpoint that lacks its trained projections and gates. An identity or passthrough output bypasses the proposed mixing operation, so its output quality and throughput do not represent that architecture.
- Preserve one-shot prompt processing when evaluating a Prefix Fast Fourier transform design. Repeating a maximum-window transform for ordinary prompt chunks multiplies work and defeats the intended prefill scaling.
- Real Fast Fourier transform workspace scales with transform length and independent feature count. A smaller logical feature batch does not bound peak memory when every batch remains lazy until concatenation; evaluate at a deliberate bounded boundary and measure the synchronization cost against retained parallelism.
- Include every persistent spectral tensor in an explicit evaluation boundary. Disconnected lazy graphs can otherwise survive across layers or forwards and move their memory and latency into a later operation.
- Incremental Prefix Fast Fourier transform updates still touch every retained frequency bin, and token-space output still requires an inverse transform. Treat decode as linear plus log-linear work in the configured window rather than constant-time work.

## Model lifecycle

- Start the worker without constructing a processor or engine. Lazy first-request loading removes avoidable application-launch input/output, memory allocation, and graphics-processor residency.
- Bound on-demand model loading and report a recoverable failure over inter-process communication. A bad discovered artifact must not stall the request queue or force a healthy idle worker to exit.
- Preserve the model-load cause chain across the worker boundary and return a dedicated local application programming interface error. A generic worker-unavailable response hides correctable artifact or quantization failures and encourages unnecessary worker restarts.
- Keep disabled diagnostic state pointer-sized. Inline fixed arrays enlarge every variant of a command enum and make each decode command pay for diagnostics; allocate the bounded accumulator only when attribution is enabled.

## Expert paging

- Cache complete one-expert MLX pages, not slices from larger expert pages. MLX slices can retain lazy backing graphs and accidentally pin oversized storage.
- Keep router-selected expert IDs separate from page-cache addresses. Preserve original selected indices for router-score alignment, and use sorted unique expert IDs only for manifests and cache lookup.
- Batch cold missing experts per layer and source shard before inserting one-expert cache entries. One safetensors reader setup per missing expert repeats avoidable work.
- Automatic sparse-expert residency must omit permanently bound selected-expert tensors and route both prompt processing and decode through ExpertPager. Retained complete layers remain an automatic reusable outcome, not a separate execution mode.
- When an optional MTP layer shares the expert pager, construct its page plan from the validated MTP tensor-to-shard map as well as the language-trunk map. Keeping those inventories separate for artifact validation does not make the trunk map sufficient for MTP page loading.
- Prefer MLX safetensors loading for bounded expert pages unless evidence proves otherwise. Rebuilding direct byte-to-array loading in Rust can add host copies and lose MLX reader optimizations.
- MLX-C managed-data constructors can let MLX wrap page-aligned memory with a shared Metal buffer, but a memory map is not automatically a valid typed tensor. Safetensors payload starts need not be aligned to the tensor dtype.

- A no-copy file mapping joins the Machine Learning framework for Apple silicon (MLX) Metal residency set. Measure mapped-page residency before and after GPU access and reclaimability advice; a virtual mapping alone is not evidence of bounded physical residency.
- Test mapped tensor offsets through an actual graphics-processor operation before using them. A dtype-unaligned packed UInt32 expert view can read from the wrong byte offset; use the bounded reader to materialize aligned MLX storage unless the artifact layout is aligned or the consuming kernel is explicitly byte-offset-aware.
- Router-selected expert indices can be a strided final-axis slice. Materialize them contiguously once, then reuse that host copy for page manifests and page-slot remapping; direct pointer copies silently read wrong multi-token IDs and can falsely implicate gather_qmm.
- Reading router-selected expert IDs on the host forces MLX array evaluation. Measure contiguous graph construction, evaluation synchronization, and the evaluated host-memory copy separately; the synchronization can dominate even when expert pages are warm and the copied byte count is tiny.
- Do not replace the bounded selected-ID host copy and central-processing-unit sort with a presence kernel that gives one graphics-processor thread to every expert and scans every assignment. Its work grows with expert count multiplied by assignment count; reducing host-transfer bytes does not compensate for that repeated scan.
- Keep sorted router assignments lazy until their dependent mixture-of-experts graph needs them. Eager evaluation before page selection can disrupt Machine Learning framework for Apple silicon scheduling, reduce retained complete layers, and amplify solid-state-drive reads.
- Do not move complete-layer admission ahead of router-selected host evidence solely to remove a synchronization boundary. Changing the memory-admission order can evict hot retained layers and increase storage traffic even when output parity is preserved.
- When complete layers consume nearly all retained expert memory, reserve each remaining paged layer's exact one-token route payload before admitting another complete layer. Apply this hybrid policy only to decode; direct multi-token prompt processing benefits from physical complete-layer admission and should not fund sparse decode routes.
- After intentionally demoting a complete layer to fund partial decode routes, synchronize the graphics-processor stream, clear released allocator storage once, and resample the live memory budget. Repeated cleanup or demotion without allocator reclamation can erase the storage-read gain.
- MLX safetensors arrays are lazy: page construction records Load primitives, while positional file reads run later on MLX input/output workers during evaluation. Measure exact reader-callback call count, bytes, total worker read time, maximum concurrency, maximum read latency, and failures without adding an evaluation boundary. These are host file-read timings, not proof of physical solid-state-drive service because the operating-system page cache may satisfy them. Keep the synchronization wait as wall time and do not subtract summed worker time from it because reads from separate readers can overlap and the same wait can include queued computation and thread-pool delay.
- Multi-token prefill should load one direct expert page per layer and keep the whole assignment batch in gather_qmm. Use a sequential full-layer page at high routing density, or at lower density when a live preflight proves the complete layer can remain retained; otherwise use the compact page.
- A retained complete layer preserves global expert order. Feed graphics-processor routing indices directly to gather_qmm; do not copy IDs to the host, concatenate one-expert pages, or remap global IDs to page slots.
- A compact expert page requires page-slot remapping after host page selection. Build one fixed-capacity global-expert-ID to page-slot MLX lookup array for the current page execution, then apply take_axis to the existing device-side selected indices; do not iterate every assignment through a Rust map and upload another assignment-sized array.
- Report resident expert-cache payload bytes, not only entry counts. A cache policy that works on one graphics-processor memory budget can fail on smaller laptops.
- A live MLX memory-limit change must update the allocator, expert-page budget, and adaptive growth guard from the same accepted byte ceiling. If one owner retains the old limit, the correct deficit formula evicts experts against stale capacity and creates avoidable SSD reloads.
- Measure cold-cache and warm-cache decode separately. Warm expert pages can prove the compute path is healthy while cold misses still dominate first-request latency.
- Warm storage-path comparisons must prewarm both graphics-processor execution paths and alternate their measured order. Otherwise whichever path runs first absorbs Machine Learning framework for Apple silicon graph and kernel first-use work, producing a false storage conclusion.
- Cold storage comparisons on macOS require an actual disk-buffer purge before each measured order. Compare only the first path from each purged process when the candidates read different files, because the second path inherits graphics-processor first-use work even if its own file pages remain cold.
- Keep alternative expert-storage layouts experimental until representative end-to-end serving improves with identical model arithmetic, output parity, residency, prompt workload, and generated-token workload. A faster isolated loader is not product evidence.
- Within an isolated storage experiment, make the graphics processor consume every loaded output, warm both paths, alternate order, time file load through immediate gather quantized matrix multiplication, and copy outputs to the host only after timing for exact parity.
- Preserve experimental repacking as an explicit offline command. Validate source identity, geometry, data type, quantization, and payload bytes before atomic publication so format mechanics remain reproducible without coupling them to model loading.
- When an experimental storage path regresses representative serving, remove its production discovery, configuration, status, attribution, packaging, and marketing together. Keeping dormant activation branches adds states and compile cost without user value.
- Bound retained expert pages by payload bytes and live Machine Learning framework for Apple silicon residency under the one configured MLX ceiling. Do not add a separate user retention maximum.
- Resolve the MLX wired-memory ceiling once at worker startup. Use that configured ceiling with MLX active and allocator-cache counters for every numeric request and expert-paging admission decision; do not add live driver or system-wide allocation caps.
- Keep ordinary partial one-expert eviction and recency layer-local. During a live retention-budget reduction, reclaim globally least-recently-used unprotected partial pages before dropping a complete layer; complete layers are coarse, original-order MLX pages and are the fallback only when finer-grained payload cannot satisfy the required reclamation bytes.
- Dropping retained expert arrays moves their buffers into the allocator cache only after in-flight graphics-processor work releases them. If a page would fit without those buffers, synchronize the model-owned stream, clear allocator memory once, and resample local MLX counters before rejecting.
- Credit a pending expert page against reclaimable Machine Learning framework for Apple silicon allocator bytes already included in system allocation. The pinned Metal allocator tries to reuse a cached buffer first and otherwise releases cached buffers under pressure; adding the full page again can trigger destructive full-cache clears before nearly every layer.
- Retained experts are lower priority than request context, but full eviction creates a disk-paging throughput cliff. If stable persistent growth exceeds the configured ceiling or the exact-context transient projection exceeds the allowed peak, lower retention by only the required reclamation bytes, clear allocator buffers, resample active memory, and retry before rejecting. If only the recovery reserve is short, freeze further retention growth without evicting hot pages. Preserve an adaptive-growth ceiling until request finalization so page-level budget updates cannot repopulate reclaimed space; release a start-admission ceiling before later request setup that can still fail.
- Reuse the forward-admission memory snapshot when full expert-cache hits need no page allocation. Do not add a second page-budget sample on that path.
- Do not subtract a universal expert-paging reserve from every laptop and model. Expert-page admission should use local MLX residency, reusable allocator bytes, the exact pending page, and the configured ceiling; initial request admission separately projects exact context growth, while per-forward admission applies exact-context transient evidence.
- When an in-flight expert page will become retained, include it once in the post-load retained target and reserve one separate normal decode page. A temporary page that will not be retained instead reserves only the larger current or future page. Counting an incoming complete layer as both in-flight and unavailable future capacity blocks safe complete-layer residency early.
- Before a temporary direct expert page is checked, use its fresh local MLX snapshot to reconcile retained experts. Rejecting before targeted eviction turns safe warm residency into a later-request failure.
- Automatic expert paging must use independently owned complete-layer pages, not permanently bound sparse tensors. Fill reversible idle memory under fresh local MLX and maximum-page checks. Request admission can reclaim the exact context, key-value, or learned-transient deficit later. Repeat the same per-layer admission after request cleanup when layers are missing, regardless of whether adaptive growth admission is enabled.
- Load-time expert residency and post-request expert recovery require different evidence. Load-time growth can use idle live capacity because no request graph exists. Post-request growth must require an enabled adaptive RAM growth guard and a completed prefill observation; otherwise it can refill memory above the safe load-time baseline without reserving transient graph work, making the next large prefill fail while synchronizing the Metal stream.
- Do not combine the configured MLX ceiling with a macOS pressure veto. A transient Warning sample can permanently strand large nominal headroom and force avoidable solid-state-drive paging. Keep one numeric authority and let request admission reclaim experts when real request growth needs capacity.
- Persistent recurrent snapshots preserve model execution state, including state produced by an invalid expert-residency layout. Quarantine snapshots created during a known correctness failure before validating its fix; otherwise restored state can reproduce output degeneration after the execution path is corrected.
- Keep persistent prompt-cache reuse enabled for MTP model artifacts. The current cache format persists target decoder state but not MTP's shifted request history, so select target-only execution whenever the persistent cache is available rather than silently disabling cache lookup or restoring incompatible MTP state.
- Model ID and revision isolate persistent prompt-cache files from foreign checkpoints, but they cannot identify changed execution semantics in the serving binary. Increase the prompt-cache format version whenever a model-state correctness fix changes what a recurrent snapshot represents, then rebuild the deployed app before allowing new snapshots to populate.
- Request-memory telemetry must sample each user-visible prefill and decode boundary independently of adaptive admission. Internal forwards whose snapshots are not published should retain the record-only path so disabling adaptive learning avoids needless fallible system queries. Always publish one post-cleanup final snapshot so idle status cannot retain released request context.
- A soft transient-recovery reserve must freeze optional expert retention for the active request, including when some layers are already paged, because asynchronous prefill pages can otherwise consume the only recovery room. Request finalization must release that ceiling before the existing live per-layer recovery checks run; retaining it across finalization would make a paged state permanent despite ample memory.
- Do not rely on an in-flight prefill to restore a reclaimed complete expert layer: request-owned tensors can keep the full page above the live Metal budget even for a short prompt. When finalization lifts a request pressure barrier, first drop request arrays and clear allocator memory, then retry only the missing complete layers under fresh per-layer budget snapshots. Exercise this transition with a synthetic cache ceiling rather than a giant live prompt.
- Scripted REST executors prove HTTP serialization, not deployability. Keep one ignored release-model REST litmus that sends consecutive long Chat and Responses requests through the public TCP surface to the same worker, includes representative tool schemas and prompt-cache reuse, and requires the worker to remain ready afterward. A soft learned-transient shortfall must freeze optional retention growth for that request even when some layers are already paged; finalization releases the request ceiling before normal live recovery.
- Expert-memory recovery logs must join the complete state transition: barrier release, allocator cleanup success, each missing layer's payload and live-budget decision, load-time budget drift, retained-layer counts before and after recovery, internal mode after recovery, elapsed recovery time, and every mode event emitted to the worker with its request phase. Logging only eviction or successful prewarm leaves no evidence to distinguish a denied reload from stale externally reported mode.
- When finalization itself changes expert residency, sample mode after finalization and attach it to the existing terminal token boundary. Reporting only the pre-finalization token mode leaves idle supervisor and menu state stale until another request starts, even though internal complete-layer recovery already succeeded.
- Report resident/paged mode from retained complete-layer ownership rather than taking another MLX or system-memory sample. Carrying that cheap state on existing generation boundaries lets the worker emit transitions without repeating model work, synchronizing the graphics processor, or contaminating public token streams.
- Automatic expert recovery happens after the last prefill-progress measurement. Emit one final post-cleanup MLX snapshot with the final residency before completion; otherwise a menu can show obsolete SSD paging and unused capacity despite a fully resident model.
- Cancellation also finalizes engine state. Preserve and publish its final residency and memory snapshot rather than discarding it, and attribute the snapshot timing independently when performance attribution is enabled.
- Avoid the “1978 Volvo” policy: do not sacrifice hot retained experts merely to maximize a generic safety reserve. Reclaim only measured allocation shortfall after reclaimable allocator memory is cleared, keep the reason and exact bytes observable, and preserve performance whenever the next operation can fit beneath the machine-derived cap.
- A virtual partial-page reserve does not create physical capacity after live memory pressure freezes retention growth. To fund partial reuse, explicitly demote retained complete layers and accept the change only when fewer reads outweigh the additional host-routing barriers.
- Do not interpret zero partial expert-page hits as zero expert reuse. Complete-layer hits use a separate counter and avoid host routing plus page assembly; assess both counters before demoting complete layers to increase partial reuse.
- Safetensors loading creates lazy Machine Learning framework for Apple silicon arrays; reader construction does not prove physical input/output or materialization. Attribute lazy page construction separately and report logical payload bytes without claiming they were read at that boundary.
- Before serializing an opportunistic prompt-cache capture, compare its conservative tensor-byte estimate with the bounded write queue's remaining byte capacity. Retain the exact post-serialization reservation because the estimate is not the serialized format's authority. This avoids Machine Learning framework evaluation, host allocation, and encoding for a capture that cannot enter the queue.
- Treat a dependency's fixed input/output worker count as a profiling hypothesis, not a universal default. After proving the readers are concurrent and the workers are saturated, size the pool from logical processor count with conservative lower and upper bounds; this adapts across laptops without creating one thread per processor.
- Do not asynchronously evaluate raw quantized expert-page tensors before their dependent mixture-of-experts graph. Eager submission can destroy beneficial lazy scheduling and serialize loads before useful dependent work.
- Do not infer useful file-read parallelism from input/output worker count. Generic same-reader positional concurrency increased memory and file-service contention and regressed representative warm model loading; retain serialization unless an end-to-end model-load benchmark proves a benefit on the active path.
- Gate previous-token expert prefetch by measured useful versus wasted payload. Partial route overlap can still amplify reads substantially even when adjacent tokens share several experts.
- Distinguish fixed layer-order weight prefetch from route-aware expert paging. Layer order can stage a complete sparse-expert layer while earlier layers compute, but it cannot reveal later router selections. At low routing density, staging the complete layer transfers mostly unused expert payload; keep direct route-driven reads unless a benchmark proves that the transfer is hidden and the larger retained payload is safe.
- Keep experimental data-plane reports explicitly scoped to their measured boundary. Do not infer user-visible latency or throughput from command count, host encoding, queue time, or one-layer projection timing.
- Measure storage-path user value only through representative production serving with prompt reuse disabled, equivalent residency, exact output parity, and enough prompt and generated tokens to expose both workloads. Exclude lazy model loading from request throughput without excluding real request orchestration.
- Compare prompt-processing throughput only at equivalent expert residency. A compact mixed-bit checkpoint can retain every sparse expert layer under the live MLX ceiling while an 8-bit variant pages missing layers from storage during each prefill; that is a different execution path, not an architecture-level prefill advantage.
- Native multi-token prediction prepares predictor history during prefill and accelerates only accepted greedy decode drafts. Do not treat its presence as evidence of faster prompt processing.

## Sampling

- Keep softmax, top-k, top-p, temperature scaling, and categorical sampling on the graphics processor.
- For descending top-p probabilities, use an exclusive cumulative sum. Keep the candidate that crosses the threshold.
- Test top-p independently with a dominant token and a threshold-crossing token before testing a full model.
- Qualify structured tool generation with the model family's documented sampling policy. Greedy decoding can loop over narrated intent and waste the output budget even when the prompt template and parser are correct.
- Multi-token prediction (MTP) verification is not automatically a throughput win. Warm both target-only and MTP paths before comparing, preserve target output as the source of truth, and keep the production path only when accepted drafts remove target work, hide latency, or otherwise improve an end-to-end measurement.
- A depth-one MTP path only becomes faster when the target model verifies a two-token window, accepts the matching draft prefix, and emits the accepted draft without another target forward. Merely drafting one token and comparing it against the next ordinary target step adds work without acceleration.
- Admit MTP before its draft mutates request state, charging the exact rounded physical growth of its independent key/value slab together with target verification growth. Do not add a fixed MTP reserve: exact model-derived growth preserves aggressive memory use without hiding an allocation from the adaptive guard.
- Project predictor history updates in their real forward boundaries. Combining two sequential one-token updates into one two-token estimate can undercount slab growth when the first update fills the current capacity and the second allocates a complete growth step.
- Keep the MTP history entry created from the confirmed current token when its draft is rejected. Only the draft is unconfirmed; rolling back the predictor entry discards committed history and misaligns later proposals.
- Qualify MTP with representative output lengths. A short all-accepted sample can hide later rejection-state defects, two-token target-forward numerical drift, and acceptance rates too low to repay proposal and verification work.
- Use approximately 1,000 input tokens and 1,000 generated tokens for model-generation throughput claims. Shorter probes may diagnose mechanics but are not performance evidence.
- Treat exact greedy parity and throughput as independent release gates. Mathematically equivalent two-token verification can use different multi-token kernels and accumulate a different decoder state from repeated one-token decode, so verification alone does not prove exact production parity.
- Prefill MTP history with each target hidden row paired to the following prompt token. Starting prediction with empty history discards the prompt context that trained the prediction head.
- Roll append-only attention state back by moving its logical frontier. Retaining complete key/value slabs for every MTP checkpoint creates context-sized copy-on-write work; only overwritten recurrent and convolution state needs tensor snapshots.
- Build greedy token selection on the graphics processor before synchronizing prediction and verification graphs. Synchronizing vocabulary logits first adds a second graphics-processor barrier without improving token selection.
- Keep raw throughput gates free of performance attribution and adaptive memory-admission sampling. Qualify attribution and memory safety separately; clock reads, allocator snapshots, and peak resets can change the result being measured.

## Chunking

- Chunk size is not an independent tuning knob. Re-tune it after changing kernels, routing, or cache layout.
- Before fast expert and Metal kernels, 512-token chunks beat 2,048-token chunks. After both fixes, 2,048 became much faster.
- Larger chunks reduce graph and launch overhead but increase attention work and peak memory.
- Re-test power-of-two chunk boundaries after graph or kernel changes. Quantized matrix-multiplication dispatch can make 8,192 tokens materially faster than nearby non-power-of-two sizes; do not infer intermediate-size performance by interpolation.
- Treat any large fixed chunk as a benchmark-only upper bound until it passes the same output and memory qualifications. The production optimizer derives candidates from the validated model context maximum, while live admission can run a smaller chunk.
- Keep the final partial chunk on the same runtime-T kernel path.
- Measure peak physical footprint together with throughput; a faster chunk is invalid if it makes the laptop unsafe.

## Profiling

- Separate central-processing-unit graph construction from graphics-processing-unit evaluation.
- Rust graph construction was already fast. Synchronizing each layer showed that excess time was inside Metal evaluation.
- Per-layer synchronization changes scheduling. Use it only to locate a subsystem, then remove it before final measurement.
- A long Machine Learning framework for Apple silicon synchronization wait identifies a host blocking boundary, not a slow graphics-processor kernel. Use Xcode Metal profiling at that boundary before changing model code.
- A long-lived attribution writer must compare the open file identity with the configured path before each report. If log rotation removes or replaces the path, reopen it so profiling continues without restarting inference.
- Start steady-state generation timing after the first output token. A cumulative clock that includes prompt processing creates artificial acceleration as more output tokens amortize prefill.
- The first layer pass mixes page faults, kernel compilation, and execution. Repeat with warm weights and pipelines before judging steady state.
- Mutate one reference behavior at a time. This identified expert sorting and Metal dispatch without guessing.
- Use 1,024 input tokens and 1,024 output tokens for representative generation measurements. Keep shorter probes only as development diagnostics.

## C++ and Rust build boundary

- A Rust toolchain change invalidates compiled artifacts and compiler-cache entries by design. Measure commit-verification steady state only after that compulsory cold rebuild, and keep the cold path bounded rather than weakening cache correctness.
- Benchmark compiler caches separately on Rust artifacts and the complete native C++ graph. Zero-copy restoration does not guarantee lower wall time when cache-key preprocessing dominates, and a wrapper that performs well for Rust can regress a large MLX C++ rebuild.
- Keep Cargo profiles, features, compiler flags, and target directories stable during Machine Learning framework for Apple silicon (MLX) iteration. Changing them invalidates native CMake paths; zero sccache hits across distinct artifact graphs does not indicate a broken wrapper.
- Turning a feature-gated CMake target off does not delete an archive produced by an earlier configuration in the same output directory. Remove stale optional archives and test executables explicitly when the feature is disabled so production artifact audits reflect the active graph.
- Native MLX and the final Rust binary must agree on the minimum macOS version.
- Static MLX release objects use Clang availability checks. Link the active Xcode libclang_rt.osx.a runtime, discovered through xcrun, or release linking can fail on ___isPlatformVersionAtLeast.
- Bind only required C functions, but update the allowlist when adopting upstream operations such as argument sorting, floor division, compiled closures, and custom Metal kernels.
- Apply native dependency patches to the pinned source archive during configuration and make Cargo track each patch as a rebuild input.
- MLX C application programming interface (MLX-C) v0.6.0 needs one release-specific patch for the Machine Learning framework for Apple silicon (MLX) v0.32.0 Fast Fourier transform parameter change. Keep it visible and fail-closed so a later upstream interface change cannot silently alter native behavior.
- MLX Fast Fourier transform output along a non-final axis can be strided. Materialize a row-major contiguous MLX array before reading a host pointer; physical storage order can differ from logical tensor order.
- The MLX C application programming interface (MLX-C) v0.6.0 exposes real Fast Fourier transform and inverse real Fast Fourier transform without a public normalization argument. Bind those functions directly; the compatibility patch supplies backward normalization internally.
- Mark each successfully applied native patch by its content hash inside the generated source tree. A patch file containing multiple sequential commits cannot be classified reliably with one reverse dry run, while unconditional error suppression can hide partial application.
- Represent scalar Metal inputs as zero-dimensional MLX arrays; do not copy device values back to Rust merely to control a kernel loop.
- Preserve ownership around C vectors, arrays, kernel handles, and output handles. Performance work must not weaken memory safety.

## Graphics-processor kernel-panic evidence

- A full-resident Qwen3.6-35B-A3B-8bit run coincided with "completeMemory() prepare count underflow" @IOGPUMemory.cpp:550 on macOS 26.5.2. The inference worker held 38,426 MiB resident at the panic.
- This panic proves an IOGPU memory-bookkeeping failure, not its initiating cause. Do not attribute it to paging, cache eviction, a particular MLX operation, or memory exhaustion without a controlled reproduction.
- Metal's recommended maximum working-set size is guidance, not a safe request-admission boundary. Before each growth operation, project current active memory, exact persistent growth, and the observed transient high-water mark as the hard requirement; native allocation enforcement remains the final boundary and a soft recovery window cannot independently trigger eviction.
- Learn transient headroom from operation-local MLX peak samples instead of withholding a fixed multi-gigabyte block. Project rounded physical key-value slab growth rather than logical token growth.
- A completed prefill observation and a positive transient byte count are different facts. Record observation completion separately for adaptive request-growth projections; do not use either fact as permission to occupy reversible idle memory with expert weights.
- Carry the adaptive guard's measured transient high-water window into later request-start and injected-context admission. Post-request expert recovery may use idle capacity because the next similar request reclaims only the exact context and learned-transient deficit before forward execution.
- Track prefill and decode transient high-water windows independently. Direct paged prefill can temporarily materialize several complete expert layers and produce a multi-gigabyte peak that correctly protects later prefill admission, but applying that same peak before every one-token decode operation evicts safe retained layers, multiplies router synchronization points, and turns the request into SSD-bound expert thrashing.
- Persistent prompt-cache restoration temporarily owns both every loaded full-attention key/value block and the concatenated live decoder state. After the allocation-free prefix lookup, reserve one additional restored-prefix key/value payload before reading MLX arrays; the final context reservation alone undercounts this reconstruction peak on a cold worker.
- Drop the loaded prompt-cache block owners before clearing the MLX allocator, then re-admit only the context tokens that remain after the restored prefix. Clearing while block owners are live cannot reclaim their allocations, while retaining the reconstruction-only expert ceiling makes the faster cache-hit path needlessly paged for the rest of generation.
- Require direct Metal input/output ranges to cover every destination tensor byte exactly. Bounds and non-overlap checks alone can leave uninitialized graphics-processor memory inside an otherwise valid MLX array and silently corrupt later matrix operations.
- Record execution mode, process footprint, wired memory, free memory, MLX memory counters, and the final submitted operation when testing large full-resident models.
- Treat throughput as invalid when the workload leaves insufficient operating-system headroom or destabilizes the graphics-processor driver.
- Never disable adaptive RAM growth protection in production. Raising the live MLX ceiling from 35 GB to 40 GB filled all 40 expert layers, left 36.83 GB active before a 41,725-token cached request, and was followed by an IOGPUMemory prepare-count-underflow kernel panic. Controlled benchmarks may disable protection only when they cannot expose the laptop to sustained high-memory inference.

## Durable rule

- Reuse the decode allocator observation already required for adaptive memory admission as live telemetry; a second MLX query or synchronization on the token path adds overhead without improving the measurement.

- A user-selected MLX memory ceiling must remain one authority: update the MLX wired, active, and allocator-cache process limits together, then update request admission, adaptive growth, and expert paging from that same exact byte cap. Decimal configuration gigabytes use 1,000,000,000 bytes.

- When lowering a live ceiling, evict only retained pageable expert payload first, synchronize the model graphics-processor stream, clear reclaimable allocator memory, and verify active bytes before lowering native MLX limits. This intentionally trades complete-layer residency for bounded SSD expert streaming instead of failing an otherwise usable model.

- When raising a live ceiling, change native MLX limits before restoring admission capacity. Automatic expert residency may prewarm missing complete layers once under the newly expanded cap; do not add polling, a second controller, or a pressure-monitor layer.

When native MLX is slower than the Python reference, assume graph or kernel-path divergence first. Compare source, disable one reference optimization, inspect compiled feature gates, and profile evaluation before redesigning architecture.
