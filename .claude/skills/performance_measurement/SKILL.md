---
name: performance_measurement
description: How to measure and explain speed in Astronomical. Use before measuring, comparing, profiling or speeding up anything that touches MLX or the GPU (prompt reading, replies, drafting, kernels, streaming from the SSD), before adding a timer, counter or measurement alias, and before running any speed test through the app or the REST API. It says to find the cause with the MLX performance profiler first and to keep end-to-end runs for the very last step.
---

# Performance measurement: find out why, not only what

Follow `AGENTS.md` and `docs/glossary.md`; this skill adds how speed is measured and explained. It was written after the same mistake was made several times: a number from the app or the REST API looked wrong, more end-to-end runs were made to chase it, and hours later nobody knew more than "slower" or "not slower".

## The rule

**An end-to-end number says that something happened. It never says why.**

A reply is the sum of many parts at different levels: kernels, hand-offs between the host and the GPU, storage reads, host bookkeeping, and what is kept in memory. A timing taken at the REST API or in the app adds all of them together with the machine's temperature, the order of the runs and the randomness of sampling. No amount of repeating it separates them.

So every question about speed is answered with the **MLX performance profiler** (<https://github.com/aosama/mlx-performance-profiler>), at the level where the time is spent. It is in the repository as the crate `crates/tooling/mlx-performance-profiler`.

## Do not measure end to end to find something out

This is the part that is easy to agree with and then ignore under pressure, so it is stated as plainly as it can be.

- **Do not** run the app, the REST API, the release reference set or a whole reply journey to learn whether a change helped, where time goes, or which of two things is faster.
- **Do not** rerun an end-to-end number that looks wrong. **Do not** alternate two builds of the app to "settle" it. Both feel like progress and produce a second number with the same blind spot.
- **Do not** read a change's effect from reply speed in a single reply. One sampled reply varies by a quarter or more on a drafted model, on the same build, in the same minute.
- **Do not** compare a number with one taken in another sitting. The machine warms, and whichever case runs second reads slower.

An end-to-end run is allowed **once, at the very end**, when the profiler has already shown what changed and why, and every level below agrees. Its only job is to confirm that a person will see what the profiler predicted. If it disagrees, that is not a reason to run it again: it means a level is missing from the explanation, and the work goes back down to find it.

## What to do instead

1. **State the question as a comparison or a breakdown.** "Is A faster than B at this?" or "Where does the time of this step go?" Anything vaguer cannot be measured.
2. **Name the smallest thing that can answer it**: a kernel on synthetic tensors at the model's real shapes, one forward pass, one stage of a step, or one whole step of an engine. Smaller is better: it runs in seconds and needs no model on disk.
3. **Hand it to the profiler.**
   - One graph: `measure_job` gives the time of every node and kernel.
   - Two graphs: `compare_jobs` runs them over interleaved rounds with the order flipped, and says faster, slower or within spread.
   - Code that makes several evaluations of its own (a loop, a pipeline, an engine's step): `compare_procedures`, with the same rounds and verdict, plus how long the GPU was busy and how long it waited on the host.
4. **Believe the verdict, including "within spread".** A difference the profiler cannot tell from noise is not a difference. Do not go looking for a tool that will say otherwise.
5. **Explain the result from the level below it** before acting on it. A step that is slower must be slower in a kernel, in a wait, in a read, or in host code, and the profiler's pages say which.

## When the profiler cannot answer

Then there are only two possibilities, and neither is "measure it end to end".

- **Our code has a layer that cannot be observed.** A stage that cannot be run alone, a step that cannot be given the same starting state twice, work hidden inside a larger call: these are defects in our code's structure. Add the seam (a way to run that stage alone) or a standing tally the engine always keeps. The measurement then becomes possible, and stays possible for the next person.
- **The profiler lacks a capability.** Open an issue on its repository, say what is needed and why, and recommend a solution. There is standing permission to do this. Keep local paths, personal details and machine details out of the issue.

Falling back to a wall clock around the whole request is how the time gets wasted. Do not do it.

## The levels, from the bottom

Start at the lowest level that can answer the question. A level's result must be explained by the levels under it; where it is not, the unexplained part is the finding.

| Level | The question it answers |
|---|---|
| Physical limit | How fast can this hardware do it at all (memory bandwidth, the SSD's read rate)? |
| Kernel | What does this arithmetic cost at the model's real shapes, and how far is it from the limit? |
| Evaluation | How many times do the host and the GPU hand work to each other, and what does each hand-off cost? |
| Native step | What does our own C++ or Rust between kernels cost? |
| Model stage | Where inside one forward pass or one draft round does the time go? |
| Engine step | What does the served path do for one prompt or one reply token? |
| Keeping policy | Would keeping different things in memory mean fewer reads? (Replayed from recorded routes; needs no GPU.) |
| App and REST | Does a person see the gain? **Confirmation only, last, once.** |

A kernel question never needs a model on disk. A policy question never needs a GPU. Only the engine-step level is true for served speed, because an engine sizes and warms things that a model loaded directly does not have.

## How a measurement is run

- **Compare within one run.** Two things are compared as two jobs or two procedures of one comparison, never as two runs made at different times.
- **Use the model's real shapes and real data sizes.** Use the Shakespeare and source-code fixtures, not random text, when a model's input matters.
- **Keep it short.** Seconds for a kernel, a minute or two for an engine step. If a measurement needs longer, the question belongs on a lower level.
- **Leave out what would blur it.** Turn off the `astro_ssd_prompt_cache` and the skill declaration, use an optimized build, and never read speed from a run with attribution on.
- **Check exactness with speed.** A faster kernel that changes a row's bits is a different model. Prove equality bit for bit on pseudo-random inputs that include values far from zero.
- **Guard with a second model** whose balance differs: one held in memory and one read from the SSD. A rule that helps one can hurt the other.
- **Say what was not measured.** A verdict from one model, one prompt length or one memory limit is a verdict about that case.

## Building for measurement

- **Every level keeps standing instruments.** Counts that cost nothing (misses, reads, bytes, rounds, hand-offs) are always on and are the first thing to read. Nobody should add a temporary timer to answer a question a second time.
- **A new way of serving reports itself**: how often it was tried, how often it served, and how often it gave way to the old path, by reason. Something that falls back silently cannot be measured.
- **A stage must be runnable alone**, from a known starting state, so it can be handed to the profiler. When writing a hot path, write the seam with it.
- **Measurement code is kept.** A comparison that answered a question becomes a named measurement alias in `.cargo/config.toml`, with a comment saying what it answers and how long it takes, so the answer can be checked again in seconds.

## Where the fix belongs

| The time is in | Work on | Why |
|---|---|---|
| A wait between the GPU and the host | Code inside the MLX evaluation | Only code inside the evaluation can remove the wait |
| A kernel that is short of the physical limit | The kernel | The general kernel is leaving bandwidth unused |
| Host code on the hot path | The native step that already runs there | It removes a crossing |
| Reads at the SSD's own rate | Reading fewer bytes | The limit is physical |
| Reads that should not have happened | The keeping policy, proven by replay first | It is a decision, not arithmetic |
| A guess about the future (read-ahead) | Nothing, until its hit rate is measured | A wrong guess costs bandwidth and evicts what was useful |

## Before proposing a speed-up

1. Read the memory notes and the tracking issue for the area: a change that was tried and lost is recorded there with its numbers.
2. Name the part of the time the change touches, and how far that part is from its physical limit.
3. Name the profiler comparison that will show it, and the guard model.
4. Say what would make the idea wrong, and measure that first.
