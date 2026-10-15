# Instructions for the Astronomical Project

- Keep full workspace verification and formatting verification only before committing and pushing, when the user asks you to commit. This codebase is slow to format-check, and full workspace test runs take a long time.

- Run cargo fmt or similar commands only before committing; that is, when the user asks you to commit, run cargo fmt.

## Never Hardwire Local Developer Environment Into the Codebase

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, acceptance artifacts, documentation, GitHub content, or agent instructions.

## Never Kill the Running Stable Instance

- While developing this codebase, never kill, stop, terminate, or force-quit the running Stable instance: the `astronomicald` daemon on `127.0.0.1:6732`, the `~/Applications/Astronomical.app`, or any process bound to the Stable state directory `~/.astronomical`.

## There are No RESTAPI/IPC/Unix Socket Downstream Consumers

- There are no downstream consumers or other dependent applications — hence no need for deprecation or compatibility shims or any other techniques.
- There is no requirement for backward compatibility for the REST API surface. There are no downstream consumers of these surfaces, so REST API backward compatibility is not a constraint.

## Code File Length and Memory Measurement Units

- Any end-user-facing file-size or memory value must use decimal SI gigabytes: 1 GB = 1,000,000,000 bytes. Do not show binary GiB values under a GB label.

## Prohibited Terminology: "qualification" and "hermetic"

- Never use "qualification" or any variant ("qualify", "qualified", etc.) in code, comments, tests, scripts, documentation, commit messages, GitHub content, or replies.
- "hermetic" is equally banned and being decommissioned (issue #1208). Remove existing occurrences and replace with the most specific term below.
- Test classification vocabulary: **offline** (a property, not a scope) = sealed and deterministic — no network beyond localhost, no real model artifacts, no machine state, so it passes identically on any laptop; **unit** = in-process logic with fakes; **integration** = crossing a component seam through stubs or fixture binaries (unix-socket IPC, spawned fixture workers, real filesystem); **end-to-end (E2E)** = driving the real `astronomicald` binary over its real HTTP surface; **acceptance** = env-gated e2e tests loading real model artifacts. A test spanning several tiers is named by the highest seam it crosses.

## Testing Instructions For Astronomical

- MLX/GPU tests must never run in parallel. Those tests must use the Rust `serial_test` `#[serial]` annotation; direct Cargo invocations must also select one test thread. The same applies to throughput-measurement tests, performance tests, and any test that loads a real model in the GPU.

- All and any tests must have a built-in timeout with a maximum of 60 seconds. Exceptions can be made for performance-endurance tests and/or tests that reproduce out-of-memory (OOM) issues.

- SSD-paging e2e tests must have a built-in timeout of 60 seconds. If e2e test exceeds that budget, use lower-level instrumentation and tests to identify the slow call path rather than increasing the e2e test timeout or repeating long model runs.

- The fixture of Romeo and Juliet MUST be used as the source test input for LLMs; there should not be random text or tokens used for testing.

- Any test case that utilizes an end-to-end model (a real model) should not be configured to allocate, acquire, or reserve more than 36 GB of RAM. Beyond that, macOS goes into resource contention and the test is meaningless. Exceptions can be made for explicit tests after user agreement; memorialize the agreement in code comments with the reasoning for the exception.

- The prompt prefill chunk size in tests should be set and fixed to 2048 tokens; this is the mechanism to guarantee the consistency and reproducibility of test results.

## Performance Related

### Performance Lessons

- Keep <repo-root>/docs/performance-optimizations-lessons.md updated with lessons learnt about performance relevant to LLMs, VLMs and MLX APIs. 

- <repo-root>/docs/performance-optimizations-lessons.md holds only durable engineering lessons that enhance performance for LLMs, VLMs, and MLX APIs (never raw metrics): matmuls, MLX kernel differences, MLX-core caching, C++ gotchas, overly precise data types, and MoE expert sorting.

### Instructions for Performance Throughput Tests

- A performance measurement should send at least 10,000 tokens of input and acquire 500 output tokens; plus or minus 10 percent is acceptable.
- A warmup run should be performed before the actual test; the desired warmup is 500 tokens of input and 50 tokens of output.
- SSD Cache must be disabled during performance throughput tests and measurements.
- Those performance throughput tests must run through a Rust optimized build.

### Instructions for Performance Profiling and Attribution

- All code must follow the unified performance-logging pattern: per-operation start/end timestamps, switchable on/off through a config parameter. Without attribution, diagnosing a slowdown is guesswork.
- Highest priority: the request-serving critical path — Model Loading from disk, Prompt Processing, Tokenization, Disk Cache, Expert Paging, and token generation.
- Profile with attribution through e2e test cases, not a running development instance. If no suitable test exists, create one after a careful review of the existing e2e cases.

## Memory Management Ownership In This Codebase

- The package under <repo-root>/crates/model-serving/src/memory must be where all memory management code is located. Including but not limited to policies, decisions, streaming and any other memory related calculations.

## Documentation under <repo-root>/docs Writing Conventions

- Markdown in <repo-root>/docs/ must be terse, define every acronym verbosely on first use, and have all assumptions eliminated before any file there is edited or written. Do not create new files in docs/ without the user's explicit agreement; docs/ files are exempt from the repository's maximum file-length limit.

## Release Scripts Folder

- All **Release releated scripts** must live are under "<repo-root>/scripts/release" , they will help you through the release process.
