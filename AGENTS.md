# Instructions for the Astronomical Project

- Keep full workspace verification and formatting verification only before committing and pushing, when the user asks you to commit.

- Run cargo fmt or similar commands only before committing; that is, when the user asks you to commit, run cargo fmt.

## Never Hardwire Local Developer Environment Into the Codebase

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, acceptance artifacts, documentation, GitHub content, or agent instructions.

## There are No RESTAPI/IPC/Unix Socket Downstream Consumers

- There are no downstream consumers or other dependent applications — hence no need for deprecation or compatibility shims or any other techniques.
- There is no requirement for backward compatibility for the REST API surface. There are no downstream consumers of these surfaces, so REST API backward compatibility is not a constraint.

## Memory Measurement Units

- Any end-user-facing file-size or memory value must use the gigabyte convention Apple uses to market the machine: 1 GB = 1,073,741,824 bytes (2³⁰). A machine with 51,539,607,552 bytes of RAM is presented as 48 GB, exactly as the purchase sticker reads; its model ceiling is (48 − 8) = 40 GB. Never present decimal SI gigabytes (10⁹ bytes) to end users. Internal storage and wire formats remain raw bytes; only display rendering and whole-GB user inputs convert, always with 2³⁰.

## Prohibited Terminology: "qualification" and "hermetic"

- Never use "qualification" or any variant ("qualify", "qualified", etc.) in code, comments, tests, scripts, documentation, commit messages, GitHub content, or replies.
- "hermetic" is equally banned and being decommissioned (issue #1208). Remove existing occurrences and replace with the most specific term below.
- Test classification vocabulary: **offline** (a property, not a scope) = sealed and deterministic — no network beyond localhost, no real model artifacts, no machine state, so it passes identically on any laptop; **unit** = in-process logic with fakes; **integration** = crossing a component seam through stubs or fixture binaries (unix-socket IPC, spawned fixture workers, real filesystem); **end-to-end (E2E)** = driving the real `astronomicald` binary over its real HTTP surface; **acceptance** = env-gated e2e tests loading real model artifacts. A test spanning several tiers is named by the highest seam it crosses.

## Testing Instructions For Astronomical

- Running unit tests or offline tests in parallel is desired.

- MLX/GPU tests must never run in parallel. Those tests must use the Rust `serial_test` `#[serial]` annotation; The same applies to throughput-measurement tests, performance tests, and any test that loads a real model in the GPU.

- Tests are plain, Book-style `#[test]` / `#[tokio::test]` functions; hang protection lives at the runner level (job-level timeouts in CI, or a runner with per-test timeout support), not inside each test. A test that must bound a specific wait (polling, IPC) wraps that wait in an explicit in-test timeout. Target per-test budget: 60 seconds; performance-endurance tests and out-of-memory (OOM) reproduction tests are exempt. The legacy `astronomical-test-harness` wrapper macros remain only until their retirement lands (tracked as a dedicated issue); new unit tests must not adopt them.

- SSD-paging e2e journeys get a 120-second runner budget. If an e2e journey exceeds that budget, use lower-level instrumentation and tests to identify the slow call path rather than increasing the budget or repeating long model runs.

- The fixture of Romeo and Juliet MUST be used as the source test input for LLMs; there should not be random text or tokens used for testing.

- The prompt prefill chunk size in tests should be set and fixed to 2048 tokens; this is the mechanism to guarantee the consistency and reproducibility of test results.

## Performance Related

### Instructions for Performance Throughput Tests

- A performance measurement should send at least 10,000 tokens of input and acquire 500 output tokens; plus or minus 10 percent is acceptable.
- A warmup run should be performed before the actual test; the desired warmup is 500 tokens of input and 50 tokens of output.
- SSD Cache must be disabled during performance throughput tests and measurements.
- Those performance throughput tests must run through a Rust optimized build.
- MOST IMPORTANTLY performance tests must start at the lower most part of the callstack and more tests upwards through the call stack.

## Memory Management Ownership In This Codebase

- The package under <repo-root>/crates/model-serving/src/memory must be where all memory management code is located. Including but not limited to policies, decisions, streaming and any other memory related calculations.

## Release Scripts Folder

- All **Release releated scripts** must live are under "<repo-root>/scripts/release" , they will help you through the release process.
