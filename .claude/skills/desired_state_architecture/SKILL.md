---
name: desired_state_architecture
description: The desired state of Astronomical's performance architecture - where a measurement is taken (physical limit, kernel, MLX evaluation, native C++ step, model stage, engine reply loop, policy replay) and what to build at each level. Use before measuring, instrumenting, or speeding up prefill or decode, before deciding whether work belongs in C++ inside MLX or in Rust, and before running any app-level or REST-level speed test. Read it before adding a timer, a counter, a `cargo measure-*` alias or temporary diagnostic code.
---

# Desired state architecture: measuring and speeding up inference

Follow `AGENTS.md` and `docs/glossary.md`; this skill adds how performance work is located and measured. It records lessons from issues #2277, #2329, #2330, #2338 and #2533. Each lesson was paid for with a wrong guess, so the rules are written as what to do first.

## The one rule

**A speed number taken at the REST or app level says that something is slow, never what.** A reply token is the sum of parts that live at different levels. Find the level that owns the time before changing anything, and measure at that level. An end-to-end run is the last step, to confirm a gain already explained lower down.

## The token budget

Every speed question starts by splitting one token (or one prompt chunk, or one draft round) into these parts. Write the split down, in milliseconds, before proposing a change.

| Part | What it is | Its limit |
|---|---|---|
| Kernel compute | GPU arithmetic: attention, projections, the gathered expert multiply | Memory bandwidth. An added expert row is bound by it. |
| Storage reads | Experts read from the SSD into the `astro_expert_cache` | The SSD's own rate, reached by parallel direct reads |
| Hand-offs | Each time the GPU waits for the host or the host for the GPU | Count them: evaluations, fences, readbacks a token |
| Host bookkeeping | Rust or C++ work between kernels: slot maps, demand counts, planning | CPU time, usually single-threaded |
| Policy | What is held in memory, so what must be read at all | Misses a token, against Belady's bound |

A change can only help the part it touches. If the part is at its physical limit, no code makes it faster; move to another part or lower the amount of work (fewer bytes, fewer rows, fewer stops).

## The measurement ladder

Climb from the bottom. Each rung has its own instrument; do not answer a lower rung's question with a higher rung's tool.

| Rung | Question it answers | Instrument (desired state) | Examples that exist |
|---|---|---|---|
| 0. Physical limit | How fast can this hardware do it at all? | A probe below MLX, or the known figure | SSD read rate; memory bandwidth per added expert row |
| 1. Kernel alone | What does this arithmetic cost at the model's real shapes? | A `direct_mlx` contract on synthetic tensors, no model on disk | `measure-expert-rows-per-read`, `measure-batched-kernel-row-exactness`, `measure-tree-attention-context-cost` |
| 2. MLX evaluation | How many waits does a token make and what does one cost? | A `direct_mlx` contract that builds the token's graph shape and prices each way of ordering it | `measure-graph-host-step` |
| 3. Native step | What does our own C++ cost on its own? | The native crate's own contract | `expert-record-reader`, `graph-host-step` |
| 4. Model stage | Where inside one forward or one round does the time go? | A standing stage tally in the model, read by a contract | `measure-dflash2-round-cost` |
| 5. Engine reply loop | What does the served path really do? | The engine loaded through `load_paging_experiment_engine`, always-on counters, attribution on request | `measure-in-graph-route-lookup`, `measure-mtp-reply-drafting`, `measure-expert-misses-speed` |
| 6. Policy replay | Would a different keeping rule or division of memory miss less? | Recorded routes replayed with no GPU | `measure-expert-misses` then `replay-expert-misses` |
| 7. App and REST | Does the user see the gain? | One confirming run on Development | The app itself |

Rules for the ladder:

- **Start at the lowest rung that can answer the question.** A kernel question never needs a model on disk. A policy question never needs a GPU.
- **A rung's number must be explained by the rungs below it.** If the engine measures 100 microseconds where the kernel measured 9, the difference is a finding, not noise: it is host bookkeeping, and it has to be named before any rule is built on the smaller number.
- **Rung 5 is the lowest rung that is true for served speed.** A model loaded directly has no decode tables, because the engine sizes them at its decode hand-off; a forward on a bare model measures a model nobody serves.
- **Rung 7 is never a diagnostic.** Use it once, after the gain is explained.

## Instrumentation, by level

The desired state is that every level carries standing instruments, so nobody adds temporary timers to answer a question twice.

### Kernels (Metal and MLX primitives)
- Price a kernel on synthetic tensors at the real model's shapes, with a floor (GPU only) beside it, so the kernel's share is visible.
- Report cost per unit of work: per row, per expert, per layer, per thousand context tokens. A single total hides the scaling law.
- Check exactness against the one-row step in the same contract. A faster kernel that changes tokens is a different model.
- State the physical limit the kernel is being compared with (bytes read at the machine's memory bandwidth).

### MLX evaluations and hand-offs
- Count hand-offs a token as a first-class figure: blocking evaluations, asynchronous submissions, cross-stream fences, host readbacks. Each is a wait.
- Price a hand-off both ways MLX can order it: Metal shared events and fast synch. The same graph costs about twenty times more a hand-off under shared events.
- Keep a token to one evaluation where possible. A second evaluation in a round is a second wait; join its roots to the first.
- Remember MLX runs every CPU-stream task in order on one thread. Work queued there waits behind whatever was queued first.

### Native C++ (crates beside MLX)
- Native code that must run inside MLX's graph or against its C++ types lives in its own small crate compiled against the provisioned, patched MLX sources, with a safe Rust owner and its own tests. `mlx-bridge` is not changed for it.
- A hook that must live inside MLX itself is a patch under `third-party/patches`, as small as possible, registered in `third-party/README.md`.
- Each native crate has a contract that prices it alone, and exposes counts (reads, bytes, nanoseconds) the engine can report without attribution.
- Every unsafe crossing carries a written soundness argument, and the failure path gets the same care as the success path: the compiler checks neither.

### The model and the engine
- Stage tallies belong in the code, not in temporary `eprintln!` lines: a draft round reports its draft, verify, pick and rewind times; a lookup reports its bookkeeping and its read separately.
- Always-on counters (misses, hits, bytes read, read wait, refusals, rounds, kept drafts) cost nothing to keep and are the first thing to read. `PerformanceAttribution` operations and counters are for the shares of time, on request, and slow the run: never read speed from an attributed run.
- A new serving way reports how often it was attempted, served and refused, by reason. A way that silently falls back cannot be measured.
- Anything that serves from a table records demand before and reads after, exactly as the path it replaces does, or evictions go wrong and the measurement compares two different policies.

### Policy (what is held in memory)
- Record routes once, replay many times. Routing depends on the model and the text, not on memory, so one recording answers every keeping rule in seconds.
- Compare against Belady's bound and against the lab's model of today's engine. The lab's model must come within ten percent of the engine's own count, or the lab does not stand for the engine.
- A constant found on one model's geometry (experts a layer, experts a token) is a ratio, not a number. Re-derive it for a new geometry before trusting it.

## How a measurement is run

- One process a case, so no case starts from tables another warmed.
- A warmup of about 500 tokens in and 50 out, then about 10,000 in and 500 out, at temperature 0.7 for code and 1.0 for general work. Greedy is a diagnostic only.
- `astro_ssd_prompt_cache` off, skill declaration off, optimized build, attribution off.
- Say whether the `macos_file_cache` was warm or evicted; measure both when reads are in question.
- A minute or two a case. If a case needs longer, the question belongs on a lower rung.
- Before and after in the same sitting, alternated when the difference is small. A difference inside the spread between two runs of the same code is not a result.
- Always run a guard: a second model whose balance of compute and reads differs (one bound by storage, one held mostly in memory). A rule that helps one can hurt the other.
- Prove replies are unchanged with a greedy reply compared token for token, including the fallback path forced on purpose.

## Where the work belongs

| The time is in | Write | Why |
|---|---|---|
| A hand-off between GPU and host | C++ inside MLX's graph (a primitive on the CPU stream) | Only code inside the evaluation can remove the wait |
| A kernel not at its physical limit | A Metal kernel, checked against the one-row step | MLX's general kernel is leaving bandwidth unused |
| Host bookkeeping on the hot path, once measured | The native step that already runs there | It removes a crossing and the Rust-side maps |
| Storage reads at the SSD's rate | Nothing lower: read fewer bytes | The limit is physical |
| Misses | Rust policy, proven in the replay lab first | It is a decision, not arithmetic |
| A guess about the future (read-ahead, prediction) | Nothing, until recall and precision are measured | A wrong guess costs bandwidth and evicts useful experts |

## Before proposing a speed-up

1. Read the memory notes and the tracking issue for the area. A lever that was built and lost is recorded there with its numbers.
2. Split the token budget and name the part the change touches.
3. State the physical limit of that part and how far the code is from it.
4. Name the rung that will measure it and the guard model.
5. Say what would make the idea wrong, and measure that first.

## Known gaps between today and the desired state

- A draft round and an in-graph lookup have no standing stage tally; their stages were timed with temporary code.
- The lookup's host bookkeeping (about a tenth of a millisecond a layer) is not separated from its read in any counter.
- Hand-offs a token are not a reported figure.
- Prediction recall and precision have no standing contract; the last measurement used temporary code.
- One model-level speed alias still measures a model without decode tables and understates reply speed; it should move onto the engine's reply loop.
