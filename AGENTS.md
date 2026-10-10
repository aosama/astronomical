# Instructions for the Astronomical Project

- Read the north star vision at repo-root/docs/north-star-product-vision.md; everything else must be congruent with it.

- You keep repo-root/docs/performance-optimizations-lessons.md updated with lessons learnt about performance relevant to LLMs, VLMs and MLX APIs.

- MLX/GPU tests must never run in parallel. Those tests must use the Rust `serial_test` `#[serial]` annotation; direct Cargo invocations must also select one test thread. The same applies to throughput-measurement tests, performance tests, and any test that loads a real model in the GPU.

- All and any tests must have a built-in timeout with a maximum of 120 seconds. Exceptions can be made for performance-endurance tests and/or tests that reproduce out-of-memory (OOM) issues.
- SSD-paging journeys must have a built-in timeout of 60 seconds. If a journey exceeds that budget, use lower-level instrumentation and tests to identify the slow call path rather than increasing the journey timeout or repeating long model runs.

- Astronomical is expected to adapt to any laptop, any RAM size, any GPU wired memory limit. Do not hardwire or optimize the codebase just for this laptop that you are developing in.

- Keep full workspace verification and formatting verification only before committing and pushing, when the user asks you to commit. This codebase is slow to format-check, and full workspace test runs take a long time.

- Run cargo fmt or similar commands only before committing; that is, when the user asks you to commit, run cargo fmt.

## Defaults for this Codebase

- Any test case that utilizes an end-to-end model (a real model) should not be configured to allocate, acquire, or reserve more than 36 GB of RAM. Beyond that, macOS goes into resource contention and the test is meaningless. Exceptions can be made for explicit tests after user agreement; memorialize the agreement in code comments with the reasoning for the exception.
- The prompt prefill chunk size in tests should be set and fixed to 2048 tokens; this is the mechanism to guarantee the consistency and reproducibility of test results.

## Never Hardwire Local Developer Environment Into the Codebase

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, acceptance artifacts, documentation, GitHub content, or agent instructions.

## Never Kill the Running Stable Instance

- While developing this codebase, never kill, stop, terminate, or force-quit the running Stable instance: the `astronomicald` daemon on `127.0.0.1:6732`, the `~/Applications/Astronomical.app`, or any process bound to the Stable state directory `~/.astronomical`. Stable is the user's live daily driver and its process lifecycle is owned by macOS LaunchAgents, not by this repository.

## There are No Downstream Consumers or Dependencies

- There are no downstream consumers or other dependent applications — hence no need for deprecation or compatibility shims or any other techniques.
- There is no requirement for backward compatibility for the REST API surface. There are no downstream consumers of these surfaces, so REST API backward compatibility is not a constraint.

## Code File Length and Memory Measurement Units

- Code files should remain around the 500-line marker, not longer.
- Any end-user-facing file-size or memory value must use decimal SI gigabytes: 1 GB = 1,000,000,000 bytes. Do not show binary GiB values under a GB label.

## Prohibited Terminology: "qualification"

- The term "qualification" and all of its variants ("qualify", "qualified", "qualifier", etc.) are prohibited in this repository and in your discourse. Never use these terms in code, comments, tests, scripts, documentation, commit messages, GitHub content, or your replies to the user.

## Requirements for Performance Profiling and Attribution

- This codebase needs to be performance optimized; to achieve that, all our code, regardless of which part, needs to have performance logging and attribution that can be switched on and off through a config parameter. The performance logging needs to capture start time and end time of each operation. This will allow us to ATTRIBUTE performance issues to specific code parts. Without attribution we will be guessing why we are observing a slowdown, which is a bad state to be in; hence it is imperative that when you are editing code, refactoring, or creating new code, it follows a unified performance logging and attribution pattern.

- The highest priority for performance logging and attribution is the critical path involved in Model Loading from disk, Prompt Processing, Tokenization, Disk Cache, Expert Paging, and finally token generation. In short, any and all operations involved in actually serving a model request to the end user.

- If you need to run performance attribution to performance profile the code with attribution logging, you can do that through the e2e test cases versus doing it through a fully running development instance. If tests do not exist, you can then create a new test after a careful and comprehensive review of the existing journeys and/or e2e test cases.

## Test Fixtures and Their Reuse for Performance and Correctness Tests

- The fixture of Romeo and Juliet MUST be used as the source test input for LLMs; there should not be random text or tokens used for testing.

- Assert model normalization and execution with structural validity checks derived from config (layer count matches, hidden size matches, shard count is positive, total bytes equals sum of shard sizes, affine profiles contain valid bits and group sizes, end tokens are present). Do not assert golden-master constants like exact byte counts, exact shard counts, or exact affine profile sets that couple tests to one specific quantization artifact — those change with every packaging variant and should not block swapping the reference model.

## Instructions for Performance Throughput Tests Under apps/inference-worker/tests/performance_throughput

- A performance measurement should send at least 10,000 tokens of input and acquire 500 output tokens; plus or minus 10 percent is acceptable.
- A warmup run should be performed before the actual test; the desired warmup is 500 tokens of input and 50 tokens of output.
- SSD Cache must be disabled during performance throughput tests and measurements.
- Those performance throughput tests must run through a Rust optimized build.

## Memory Management Codebase

- The package under <repo-root>/crates/model-serving/src/memory must be where all memory management code is located. Including but not limited to policies, decisions, streaming and any other memory related calculations.
