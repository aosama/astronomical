# Instructions for the Astronomical Project

## Never Hardwire Local Developer Environment Into the Codebase

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, acceptance artifacts, documentation, GitHub content, or agent instructions.

## There are No RESTAPI/IPC/Unix Socket Downstream Consumers

- There are no downstream consumers or other dependent applications — hence no need for deprecation or compatibility shims or any other techniques.
- There is no requirement for backward compatibility for the REST API surface. There are no downstream consumers of these surfaces, so REST API backward compatibility is not a constraint.

## Memory Measurement Units

- Any end-user-facing file-size or memory value must use the gigabyte convention Apple uses to market the machine: 1 GB = 1,073,741,824 bytes (2³⁰). A machine with 51,539,607,552 bytes of RAM is presented as 48 GB, exactly as the purchase sticker reads; its model ceiling is (48 − 8) = 40 GB.

## Prohibited Terminology: "qualification" and "hermetic"

- Never use "qualification" or any variant ("qualify", "qualified", etc.) in code, comments, tests, scripts, documentation, commit messages, GitHub content, or replies.
- "hermetic" is equally banned and being decommissioned (issue #1208). Remove existing occurrences and replace with the most specific term below.

## Testing Instructions For Astronomical

- Tests that use real full sized models and loads them in memory for further use by the GPU must use the Rust `serial_test` `#[serial]` annotation; The same applies to throughput-measurement tests and performance tests.

- SSD-paging e2e journeys get a 120-second runner budget. If an e2e journey exceeds that budget, use lower-level tests to identify the slow call path rather than increasing the budget or repeating long model runs.

- The fixture of Shakespear is highly recommended to be used as the input for LLMs; there should not be random text or tokens used for testing. Exceptions can be made where Shakespear is not suitable for the test -- for example a coding fixture.

## Performance Instructions

- A performance measurement should send at least 10,000 tokens of input and acquire 500 output tokens; plus or minus 10 percent is acceptable. Exceptions can be made if the user specifies it explicitly.
- A warmup run should be performed before the actual test; the desired warmup is 500 tokens of input and 50 tokens of output.
- SSD Cache must be disabled during performance throughput tests and measurements.
- Those performance throughput tests must run through a Rust optimized build.
- MOST IMPORTANTLY performance tests must start at the lowermost part of the call stack and move tests upwards through the call stack.

## Release Scripts Folder

- All **Release-related scripts** must live under "<repo-root>/scripts/release", they will help you through the release process.

## Repository Tooling Lives In xtask

- `<repo-root>/xtask` exists so that repository tooling is typed, unit-tested Rust invoked as `cargo xtask <command>` instead of untested shell, with each subcommand keeping the output lines and exit codes of the shell tool it replaced so hooks and CI steps survive unchanged. Whenever you add or change such tooling, put it in xtask as a new or extended subcommand with its tests and document the invocation in the `xtask` alias comment of `.cargo/config.toml`;
- This repo prefers <repo-root>/xtask convention versus shell scripts, shell scripts are highly discouraged and the repo is moving off of shell scripts and reducing their usage.
