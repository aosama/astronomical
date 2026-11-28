---
name: developing_and_optimizing_gpu_kernels_and_operations
description: The lessons that really moved Astronomical's GPU speed, and the ideas that measured as dead ends. Use before writing or changing a Metal kernel, fusing MLX operations, changing how a model's graph is submitted, tuning tile sizes or prompt chunk sizes, or proposing any GPU speed-up for prompt reading or replies.
---

# Developing and optimizing GPU kernels and operations

Follow `AGENTS.md` (the "Metal Kernels Are First-Class Files" rules), `docs/glossary.md` and the `performance_measurement` skill. This skill records what actually changed speed on Apple GPUs in this codebase, with the numbers, so the next kernel starts from what is known instead of from a guess. Every lesson below was measured; several ideas that sounded certain were measured as losses, and those are recorded too.

## First: is there anything to win?

Before writing a kernel, find out how far the operation is from its ceiling. A kernel can only remove time that the hardware is not already using.

- **Compare the kernel with both ceilings**: arithmetic (TFLOP/s) and memory bandwidth (bytes moved per second). `cargo measure-prefill-chunk-breakdown` reports which one a kernel is near.
- **At a ceiling, stop.** Bonsai 2.5 27B's 2-bit prompt projections run at 22-25 TFLOP/s, and the attention projections of Ornith 35B-A3B at about 95% of the arithmetic ceiling. A one-token 4-bit expert multiply reads at 240-255 GB/s, the Mac's bandwidth. No shader beats those numbers.
- **Below both ceilings is headroom.** Ornith 35B-A3B's gathered expert multiplies were about 80% of a prompt layer and reached only 30-60% of either ceiling. That is where the tiled kernels below won.
- **Decode on a held model is bandwidth-bound.** A reply token is GPU-busy 96-98% of its time and reads its weights near the bandwidth limit. The multiplier left for replies is drafting, which shares one pass's weight reads across several kept tokens, not a faster kernel.

## The lessons that moved the needle

### 1. Fuse so intermediates never reach memory

The biggest win came from a kernel that computes gate, up and SwiGLU in one pass and writes only the activated row. The two projection outputs never touch device memory. Together with lesson 2, a prompt layer's routed experts took 17.2% less GPU time than MLX's route (12 of 12 rounds, same bits), and a served 2,048-token chunk ran 3.8% faster.

Fusion pays when the fused-away intermediate is large and is written then read straight back. It does not pay when the GPU is already full (see "Ideas that measured as losses").

### 2. Read through an index instead of copying to reorder

MLX's sorted gather route first copies every token row into expert order, then multiplies. Our gate/up kernel reads the token rows in place through the sort order (`row = order[i] / TOPK`). The copy is gone, and with it most of the transient memory: about 40 KB a prompt token instead of about 88 KB. On a large prompt that saves time and `mlx_working_buffers` together.

Whenever an operation exists only to put data in a convenient order for the next operation, try making the next operation read through the order.

### 3. Do the bookkeeping on the GPU, inside the graph

- The tiled kernels need a list of (first row, expert, row count) tiles. One threadgroup of 1,024 threads builds it on the GPU (`astro_primitive_sorted_expert_tile_list.metal`), so the host never reads routing back mid-layer.
- In-graph route lookups (`crates/engine/graph-host-step`, a host step that runs as an MLX primitive on MLX's CPU stream) removed per-layer host round trips on SSD streaming: Flash-Next 34 GB rose from 17.8 to 19.3 tokens a second, Ornith OptiQ 16 GB from 48.6 to 56.0.
- A blocking evaluation in the middle of a token costs about 150 µs. Under Metal shared events, a hand-off costs about twenty times what it costs under fast synch.

Count hand-offs before writing arithmetic. A wait between the host and the GPU is often worth more than a faster kernel.

### 4. Tile size and launch order are tuning knobs, so race them

The same arithmetic ran at different speeds depending on how the grid handed out work:

| Change | Result |
|---|---|
| Gate/up row tiles of 96 rows | Kept |
| Row tiles of 128 rows | Slower (6,127-6,335 ms against 5,995 ms for a prompt), reverted |
| Down projection: 64-row tiles in groups of 32 on the grid's x axis | One layer's routed experts 6.16 ms against 6.43 ms at 2,048 tokens; 19.31 against 20.02 ms at 8,192 |
| Double-buffered 64-deep weight steps, one barrier a step | Part of the 17.2% kernel win |

Never pick a tile shape by reasoning alone. Race the candidates at the model's real shapes and at more than one token count.

### 5. The prompt chunk size is a kernel parameter

Wider spans give each expert more rows, so the tiles fill. At real 35B-A3B shapes, our routed-expert cost fell from 3.0 µs to 2.4 µs a token between 2,048 and 8,192 tokens (oMLX's kernels showed the same curve).

Served on an M5 Pro, 4,096-token chunks read a 15K prompt 7.6% faster than 2,048 (30K: 2.3%) for 0.35 GB more peak memory. A dense 9B showed no difference beyond run-to-run drift.

The engine now picks this itself: the in-memory chunk's Automatic size is 4,096 for models with experts on an M5 or newer and 2,048 elsewhere (`prompt_processing_chunk_choice`). The 6-bit 35B-A3B, which runs MLX's own expert kernels, gained 6.4% too, so the gain comes from filling expert tiles, not from one kernel.

When a kernel's efficiency depends on rows per expert, measure the chunk size with it. Memory, macOS's roughly 10-second limit on a GPU command buffer, and how often a request can be cancelled set the upper bound.

### 6. Let decode's fused kernels serve prompt spans too

The gated-delta prework kernel (rolling window, conv1d, SiLU, split, both RMS norms and scales in one launch) had been capped at 4 tokens, so prompt spans ran the composed operations. Raising the cap to 16,384 (`MAX_FUSED_TOKEN_COUNT`) and proving the bits at 2,048 rows and an uneven 37-row tail gave about 50 ms on a 15K prompt.

That was less than predicted: estimates of a fusion's gain are routinely too high. Look for fused kernels that already exist before writing new ones.

### 7. Learn from the fastest implementation available, then race it in one process

The tiled schedule follows oMLX's tensor-unit sorted `gather_qmm` (Apache-2.0, credited in the commit). Reading the fastest competing code saved days. The verdict came from a race in one process:

- both kernels built from their own sources;
- the same inputs;
- bits compared first;
- alternating which one runs first each round;
- medians over 30 rounds.

Two separate runs in two separate sittings can never answer "which is faster".

### 8. Look outside the kernels before blaming them

In the oMLX comparison, the largest single gap was not a kernel. The `astro_ssd_prompt_cache` wrote its blocks synchronously after the last layer, about 450 ms on a 15K prompt, and it was fixed by publishing on a background thread. The next was block sizes that cut prompt spans into odd pieces, fixed by aligning blocks to the chunk.

Split the whole prompt or token budget first (kernels, hand-offs, host bookkeeping, storage, policy) and fix the largest part.

## Ideas that measured as losses

Do not rebuild these without new evidence.

| Idea | What happened | Why |
|---|---|---|
| Joining gate and up weights into one gathered multiply | Kernel alone: 23.6% less GPU time. Served engine: within spread, at 11 GB extra memory. Joining inside the graph: +33% | MLX already runs the gate and up gathers concurrently in a real layer, so the GPU was already full. A single-kernel verdict does not see that overlap. |
| Fusing Flash-Next's lane mixers | 9,678 → 8,135 primitives a token for +1% (bit for bit) | A primitive costs about 0.3 µs in a real token, not the 3-4 µs that dividing unexplained time by primitive count suggested. Small proven gains are still kept. |
| Raising MLX's ops or megabytes per command buffer | Slower | MLX's own split points are tuned; leave its limits. |
| One graph per reply token instead of submitting every 3 layers | +3% on 9B, +1.5% on 27B (slower) | MLX then splits on its own limits and the GPU waits more. |
| A custom ternary matmul for Bonsai | No headroom | MLX's 2-bit kernel is already at the arithmetic ceiling and 3-20% faster than float16. |
| Zero-copy mapped buffers for expert pages | 2.8× slower cold, 2.9× warm | Faulting pages in during the kernel costs more than the copy it saves. |
| Computing missed experts on the CPU | 0.38 ms an expert on a core against 0.012 ms on the GPU | The read still has to happen; the CPU only adds compute. |

The pattern: a kernel or primitive measured alone overstates what serving gains. Confirm any fusion through the served engine (`load_paging_experiment_engine` with the profiler's `compare_procedures`) before shipping it.

## Exactness: a faster kernel with different bits is a different model

- **Match MLX's bits, not only its values.** Copy MLX's exact formulation, including its sigmoid inside SwiGLU and its accumulation order. The tiled kernels reproduce MLX's route bit for bit, and a served chunk kept its logits to the bit.
- **Prove bits on hard inputs.** Use pseudo-random values that include values far from zero, at the real model's shapes, and cover:
  - an uneven tail (37 rows);
  - every row on one expert;
  - large spans (8,192 tokens);
  - inputs the kernel must refuse.
- **Watch kernel selection by shape.** MLX's `GatherQMM` picks a tiled or a per-row kernel by row count and expert count, and the two round differently. Paged and fully held expert tables once gave different replies for that reason, and it looked like an `astro_expert_cache` bug for days. When two paths disagree, compare greedy replies first, then suspect a shape-dependent kernel choice.
- **Probe at load and refuse cleanly.** A capability probe runs the kernel once against MLX's route when the model loads. It keeps the kernel only where the bits match and falls back otherwise. Every refusal is counted with its reason; a silent fallback cannot be measured.
- **Guard test fixtures against overflow.** A 32-bit index product in a fixture once produced negative expert ids and all-zero outputs. Use 64-bit arithmetic for index math in fixtures.

## Hardware generations

- **M5 tensor units (NAX).** MLX enables its NAX kernels from the real device generation. `MLX_METAL_GPU_ARCH` only changes which metallib is loaded; the kernels can only be switched off by building MLX without them. An M5 cannot stand in for an M1-M4.
- **Older Macs take MLX's route.** On a Mac without NAX the probe refuses the tiled kernels, and MLX's gather route runs, with its sorted copy and roughly twice the transient memory a token.
- **Tuning is per generation.** A tile shape, grid layout or chunk size tuned on M5 is a verdict about M5. Mark per-generation choices in code, and measure on real older hardware before changing what they run.

## Mechanics in this repository

- Kernels live in `.metal` files under a `metal/` folder beside their launcher, named `astro_<scope>_[<owner>_]<operation>`, each opening with its `// @` signature block. Shared helpers go in `.h` files, and shared bodies in `<name>_body.metal`.
- Add a `@compile-with` line for every template combination you launch. `cargo xtask check-metal-kernels` compiles each one offline exactly as MLX wraps it.
- Declare only through `declare_metal_kernel` with the build-script constant. Helpers several crates share go in `crates/engine/model-primitives/src/metal_common/`.
- Metal reserves type names, so a variable named `half` does not compile.
- Every kernel ships with:
  - its exactness contract (for example `cargo test-sorted-expert-tiled-matmul`);
  - a capability probe where serving depends on it;
  - a speed verdict kept as a `.cargo/config.toml` alias (for example `cargo measure-sorted-expert-tiled-layer-verdict`), so the answer can be checked again in seconds.
- GPU tests under `crates/` and `apps/` take `#[file_serial]`.

## Checklist before proposing a kernel or fusion

1. Split the time budget. Name the operation's share and its distance from both ceilings.
2. Look for an existing fused kernel, or a faster implementation elsewhere (MLX, oMLX, mlx-lm), and read it.
3. Ask whether the idea removes memory traffic (intermediates, reordering copies) or hand-offs. Those are what moved the needle. Fewer primitives alone did not.
4. Prove the bits against MLX's route first, then race in one process with alternating order at real shapes and two or more sizes.
5. Confirm once through the served engine, with a second model whose balance differs, and keep the race as a measurement alias.
