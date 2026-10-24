# Instructions for the Astronomical Project

- Try to have code files not exceed around 600 lines of code give or take.

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

## Testing Instructions For Astronomical

- Running unit tests or offline tests in parallel is desired.

- MLX/GPU tests must never run in parallel. Those tests must use the Rust `serial_test` `#[serial]` annotation; The same applies to throughput-measurement tests, performance tests, and any test that loads a real model in the GPU.

- SSD-paging e2e journeys get a 120-second runner budget. If an e2e journey exceeds that budget, use lower-level instrumentation and tests to identify the slow call path rather than increasing the budget or repeating long model runs.

- The fixture of Romeo and Juliet MUST be used as the source test input for LLMs; there should not be random text or tokens used for testing.

- The prompt prefill chunk size in tests should be set and fixed to 2048 tokens; this is the mechanism to guarantee the consistency and reproducibility of test results.

## Performance Related

### Instructions for Performance Throughput Tests

- A performance measurement should send at least 10,000 tokens of input and acquire 500 output tokens; plus or minus 10 percent is acceptable.
- A warmup run should be performed before the actual test; the desired warmup is 500 tokens of input and 50 tokens of output.
- SSD Cache must be disabled during performance throughput tests and measurements.
- Those performance throughput tests must run through a Rust optimized build.
- MOST IMPORTANTLY performance tests must start at the lowermost part of the call stack and move tests upwards through the call stack.

## Release Scripts Folder

- All **Release-related scripts** must live under "<repo-root>/scripts/release", they will help you through the release process.

## Repository Tooling Lives In xtask

- `<repo-root>/xtask` exists so that repository tooling is typed, unit-tested Rust invoked as `cargo xtask <command>` instead of untested shell, with each subcommand keeping the output lines and exit codes of the shell tool it replaced so hooks and CI steps survive unchanged. Whenever you add or change such tooling, put it in xtask as a new or extended subcommand with its tests and document the invocation in the `xtask` alias comment of `.cargo/config.toml`;
- This repo prefers <repo-root>/xtask convention versus shell scripts, shell scripts are highly discouraged and the repo is moving off of shell scripts and reducing their usage.

## Verification Suites Live Under verification

- `<repo-root>/verification/` is for repository-wide checks that no single package owns; it exists because Cargo reserves integration-test `tests/` directories for packages, the root manifest is a virtual workspace with no package, and the name aligns with `cargo verify` and the macOS offline verification lane. Put real-model acceptance journeys and graphics processing unit contracts in `verification/acceptance`, whole-product daemon journeys in `verification/e2e`, and shell contracts required before any Rust build in `verification/tooling_contracts`; keep package-owned unit tests and integration-test binaries in `crates/*/` or `apps/*/`, and put repository tooling expressible as typed Rust in `xtask`.
