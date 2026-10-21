# Instructions for the Astronomical Project

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
- MOST IMPORTANTLY performance tests must start at the lower most part of the callstack and more tests upwards through the call stack.

## Memory Management Ownership In This Codebase

- The package under <repo-root>/crates/model-serving/src/memory must be where all memory management code is located. Including but not limited to policies, decisions, streaming and any other memory related calculations.

## Release Scripts Folder

- All **Release releated scripts** must live are under "<repo-root>/scripts/release" , they will help you through the release process.

## Repository Tooling Lives In xtask

- `<repo-root>/xtask` exists so that repository tooling (CI reporting such as the tier pyramid, verification glue such as the source-size and citation checks, third-party notices, the native build fingerprint and dependency bootstrap, change-scope classification, and cache pruning and saving) is typed, unit-tested Rust invoked as `cargo xtask <command>` instead of untested shell, with each subcommand keeping the output lines and exit codes of the shell tool it replaced so hooks and CI steps survive unchanged. Whenever you add or change such tooling, put it in xtask as a new or extended subcommand with its tests and document the invocation in the `xtask` alias comment of `.cargo/config.toml`; reserve shell for the release scripts under `scripts/release`, the thin CI wrappers that call xtask, and the tooling contracts that must run before any Rust build exists.

## Verification Suites Live Under verification

- `<repo-root>/verification` is the home of repository-wide verification that no single package owns: `verification/acceptance` (the astronomical-acceptance package of real-model, environment-gated acceptance journeys, GPU contracts, and the source-controlled throughput-history ledger), `verification/e2e` (the astronomical-e2e package of whole-product daemon journeys that spawn the real astronomicald binary over its HTTP/IPC surface), and `verification/tooling_contracts` (the shell contracts about repository tooling that must run before any Rust build exists). It exists because Cargo reserves a `tests/` directory beside a `Cargo.toml` for that package's own automatically discovered integration tests, and the root manifest is a virtual workspace with no package of its own, so a top-level `tests/` name carried none of that meaning; "verification" covers all three contents and matches vocabulary the project already uses (`cargo verify` and the "macOS offline verification" lane). Add to it only verification that spans the repository — new acceptance journeys or GPU contracts into `verification/acceptance`, new whole-product daemon journeys into `verification/e2e`, and new pre-Rust shell contracts into `verification/tooling_contracts` — while a package's own unit tests beside its source and its `tests/` integration-test binaries stay under that package in `crates/*/` or `apps/*/`, and repository tooling expressible as typed Rust goes in `xtask` instead.

## Development App Sessions Must Leave The User's CLI Working

- The CLI resolves its channel (Stable vs Development) from its own executable path (`apps/astronomical-cli/src/instance.rs`) and, per the strict channel isolation decision, never falls back to the other channel. A `astronomical` command whose resolved binary lives outside either app bundle is therefore bound to the Development channel and fails with "Astronomical isn't running" whenever the Development instance is not running — even while the Stable app runs normally.
- Any session that builds, launches, or quits "Astronomical Development.app" must finish by verifying the user's default `astronomical` command still reaches the instance the user actually uses: resolve `which astronomical`, follow every symlink to the real binary, and confirm the channel that path implies matches that instance.
- Never leave a PATH entry named `astronomical` pointing at a repository build. A repository-built CLI for testing the Development channel belongs under a distinct command name (for example `astronomical-dev`), so the channel the name implies matches the channel the binary is bound to.
