---
description: "Rust conventions for Java-readable code: explicit owners, no production unwraps, clear errors, aggregated tests, and disciplined parallel Cargo verification"
applyTo: "**/*.rs"
---

# Rust Coding Instructions in Java-ish Style (source: rust.instructions.md)

## Intent (source: rust.instructions.md)

Write Rust that a Java-background maintainer can navigate confidently: explicit modules, named owners, predictable APIs, typed errors, and tests that are easy to run. Prefer readability and testability over compact idioms when they conflict.

## Priority (source: rust.instructions.md)

1. Correctness: no production unwrap/expect, no swallowed errors, no recoverable panic!.
2. Java-readable design: one clear owner per file, explicit service/state objects, simple control flow.
3. Testability: public seams are acceptable for crate-root tests; avoid privacy gymnastics.
4. Consistency: names, docs, tests, and errors match behavior.

## Source Layout And API (source: rust.instructions.md)

- Organize by domain responsibility, not Rust novelty.
- Each .rs file should have one primary public owner: struct, enum, trait, DTO, error enum, or pure formula module. Supporting types are fine when tightly related.
- Put stateful or business logic behind owner impl methods. Public free functions are fine for thin entrypoints, pure formulas, and stateless helpers.
- Use explicit, typed public APIs. Parse and validate boundary data early, then pass typed values inward.
- Broader visibility is acceptable when it makes tests and navigation clearer. Do not create complex Rust-only workarounds just to keep internals private.

## Error Handling And Runtime Behavior (source: rust.instructions.md)

- Use domain-relevant error enums or typed errors at module boundaries.
- Preserve lower-level causes when translating errors; do not discard context.
- Do not use assert! for runtime validation in production paths.
- Do not ignore fallible results unless the reason is explicit and documented.
- Network, process, and async work must have explicit timeout or cancellation behavior.
- Keep async orchestration inside clear owner types; avoid ad hoc task spawning and shared mutable state.

## Naming, Comments, And Docs (source: rust.instructions.md)

- Use names that explain role and domain meaning; avoid vague names like data, result, item, or tmp.
- Prefer explicit if/match and early returns over dense clever expressions.
- Add rustdoc for public types and public methods when contracts are not obvious.
- Comments should explain intent, constraints, and edge cases, not restate syntax.
- Keep comments, docs, errors, and public names congruent with behavior.

## Tests (source: rust.instructions.md)

- Test functions should start with should_ and describe behavior.
- Tests live under the crate-root tests/ directory, not in #[cfg(test)] mod tests blocks inside src/.
- Use a small number of auto-discovered integration-test binaries organized by execution boundary so hermetic, direct-MLX, and model-artifact qualification changes compile only their own test subtree; shared helpers live in tests/common/.
- If broader public visibility is needed for these crate-root tests, prefer the readable public seam over inline tests or privacy workarounds.

## Cargo Verification (source: rust.instructions.md)

- **Do not run multiple Cargo commands in parallel**. Cargo commands share package-cache and target-directory locks; concurrent runs can serialize, block each other, and produce meaningless timing evidence.
- Do not intentionally start multiple independent Rust compiler or Cargo verification processes in the same repository. Use one Cargo invocation with explicit job parallelism instead.
- Before starting a Cargo command after an interruption, inspect and stop stale Cargo, Rust compiler, or Rust lint-driver processes in that repository. Do not wait blindly on a file lock or start another Cargo command on top.
- Use an explicit parallel-job environment for non-trivial local builds. Set CARGO_BUILD_JOBS or pass -j <logical_cpu_count> using the machine's logical CPU count (sysctl -n hw.logicalcpu on macOS, nproc on Linux) unless the repo documents a lower memory-safe value.
- Do not set RUSTFLAGS globally to chase CPU utilization. RUSTFLAGS changes the artifact graph, can force rebuilds, and can alter generated code; use it only for a specific documented need.
- Keep repeated verification on one artifact graph. Do not casually switch between profiles, feature sets, target directories, or RUSTFLAGS; switching between debug, test, ci-test, release, different features, or different CARGO_TARGET_DIR values forces new compilation.
- Prefer one stable target directory per repo and profile. Use a disposable CARGO_TARGET_DIR=/tmp/... only for an intentional cold-build probe, and label it as such.
- Avoid many --test targets and many [[test]] entries; each produces another compiled and linked binary.
- Add --timings to slow Rust build/check/lint commands and inspect the generated report before changing profiles, features, dependencies, or test layout for speed.
- If a build appears to use only one CPU core for more than about 30 seconds, gather evidence before continuing: sample live rustc process count, summed rustc CPU, Cargo locks, command profile, target directory, and feature set. Treat low CPU plus long wall time as a build-orchestration problem until proven otherwise.
- Broad cargo test --tests or workspace tests are final confirmation tools, not the default feedback loop.
- If you see a warning from rust compiler about dead code, fix it right away.
