# Instructions for the Astronomical Project

- This is our constitution at repo-root/docs/north-star-product-vision.md everything is derived from there.

- Call free functions through their owning module: import the module, never the bare function (`use crate::support;` then `support::run_journey_with_timeout(...)`), so every call site names its owner; all new and refactored Rust code must follow this.

- You keep repo-root/docs/performance-optimizations-lessons.md updated with lessons learnt about performance relevant to LLMs, VLMs and MLX APIs.

- All commands and tests must emit a live progress indicator instead of leaving the user with silent output.

- MLX/GPU acceptance journeys must never run in parallel. Each journey loads model weights into wired GPU memory, so concurrent journeys multiply that demand past the machine's physical limit and can hard-panic the whole system (watchdog starvation → forced power-off). This is enforced structurally: `scripts/run-bounded-cargo-test.sh` rejects any ignored-test invocation that asks for more than one test thread and injects `--test-threads=1` otherwise, so callers cannot parallelize real-model journeys. Test threads = 1 is not applicable for hermetic tests, those can be parallelized safely since they use CPU only.

- All and any tests must have a built in timeout with a maximum of 120 seconds. Exceptions can be made for tests that deal with performance endurance tests and/or reproducing OOM issues.

- Astronomical is expected to adapt to any laptop, any RAM size, any GPU wired memory limit. Do not hardwire or optimize the codebase just for this laptop that you are developing in.

- Keep full workspace verification and formatting verification only before committing and pushing when the user asks you to commit. This codebase is slow to format check and do a full workspace/test runs.

- Before every requested commit run scripts/verify-before-commit.sh; never substitute cargo test --workspace --all-targets because it runs broad integration binaries serially. Exceptions are for changes that you are confident does not relate to code or functionality, for example documentation or a version bump for our artifacts or a static site change or github CI build related and so on.

- Run cargo fmt or similar commands only before committing, i.e. when the user asks you to commit then you do the cargo fmt

## Local Environment Boundaries

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, acceptance artifacts, documentation, GitHub content, or agent instructions.

- Resolve user-controlled locations through configuration, environment variables, platform-standard application directories, command-line arguments, or explicit user file selection.

- Tests must use temporary directories, repository fixtures, or clearly fictional placeholder paths that cannot identify a developer workstation.

## Never Kill the Running Stable Instance

- While developing this codebase, never kill, stop, terminate, or force-quit the running Stable instance: the `astronomicald` daemon on `127.0.0.1:6732`, the `~/Applications/Astronomical.app`, or any process bound to the Stable state directory `~/.astronomical`. Stable is the user's live daily driver and its process lifecycle is owned by macOS LaunchAgents, not by this repository.
- All development, debugging, and validation run against the Development instance (`~/.astronomical-dev`, `127.0.0.1:6733`, `Astronomical Development.app`). If you need a clean slate, restart or rebuild the Development instance -- never the Stable one.
- This means no `pkill`/`killall`/`kill` against `astronomicald`, no `launchctl stop` on the Stable agent, and no killing the Stable app from the Dock or Activity Monitor. If a Stable process is wedged, report it and let the user handle it; do not terminate it yourself.

## There are No Downstream Consumers or Dependencies

- There are no downstream consumers or other dependant applications -- hence no need for deprication or compatibilty shims or any other techniques.

## Code File Length and Memory Measurement Units

- Code files should remain around the 500 lines marker not longer.
- Any end-user-facing file-size or memory value must use decimal SI gigabytes: 1 GB = 1,000,000,000 bytes. Do not show binary GiB values under a GB label.

## There is No Backward Compatibility Requirements for the REST API surface

- There is no requirement for Backward compatibility for the REST API surface. There are no downstream consumers of these surfaces so RESTAPI backward compatibility is not a constraint.

## Prohibited Terminology: "qualification"

- The term "qualification" and all of its variants ("qualify", "qualified", "qualifier", etc.) are prohibited in this repository and in your discourse. Never use these terms in code, comments, tests, scripts, documentation, commit messages, GitHub content, or your replies to the user.
- Describe the activity with the repository's own vocabulary instead: run the acceptance journeys and check the acceptance criteria that proves a model or feature is fit for its purpose.

## Requirements for Performance Profiling and Attribution

- This codebase needs to be performance optimized, to achieve that, all our code, regardless which part, needs to have performance logging and attribution that can be switched on and off through config parameter. The performance logging needs to capture start time and end time of each operation. this will allow us to ATTRIBUTE performance issues to specific code parts. Without attribution we will be guessing why we are observing a slow down, which is a bad state to be in, hence it is imperative that when you are editing code or refactoring or creating new code that they follow a unified performance logging and attribution pattern.

- The highest priority for performance logging and attribution is the critical path involved in Model Loading from disk, Prompt Processing, Tokenization, Disk Cache, Expert Paging and finally token generation. In short any and all operations invovled in actually serving a model request to the end user.

- If you need to run performance attribution to performance profile the code with attribution logging you can do that through the e2e test cases vs. doing it throuhg a fully running development instance. If tests do not exist you can then create a new test after careful and comprehensive review of the existing journeys and/or e2e test cases.

## Test Fixtures and Their Reuse for Performance And Correctness Tests

- The fixture of Romeo and Juliet MUST be used and the source test input for LLMs, there should not be radnom text or tokens used for testing.

- Assert model normalization and execution with structural validity checks derived from config (layer count matches, hidden size matches, shard count is positive, total bytes equals sum of shard sizes, affine profiles contain valid bits and group sizes, end tokens are present). Do not assert golden-master constants like exact byte counts, exact shard counts, or exact affine profile sets that couple tests to one specific quantization artifact — those change with every packaging variant and should not block swapping the reference model.

## Instructions for Performance Throughput Tests Under apps/inference-worker/tests/performance_throughput

- A performance measurement should at least send 10000 tokens input and acquire 1000 ouput tokens .. plus or minus 10% is acceptable.
- A warmup run should be performed before the actual test, desired is 1000 tokens input and 100 tokens output as warmup.
- SSD Cache must be disabled.
- Those performance throughput tests must run through an optimized build.

## Principles to Follow While Testing SSD Model Streaming

- It is a good practice while testing SSD model streaming to allocate RAM that is 50% of the model size on disk. This would be more realistic towards what RAM end users are likely to have available.
- Tests that use a real model duing SSD streaming should produce measurements for throughput covering (a) tokens per second during prefill/prompt processing (b) tokens per second during token generation.

## Memory Management Codebase

- The package under <repo-root>/crates/model-serving/src/memory must be where all memory management code is located. Including but not limited to policies, decisions, streaming and any other memory related calculations.

## Github Issues are Not Always Correct

- Do not take github issues as canonical or up-to-date. The intent might be good but dont treat their contents as the truth or as must implement in full. Discuss with the user first and use your judgement after careful discovery and introspection.
