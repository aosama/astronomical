# Contributing to Astronomical

## Before Starting

- Open or join an issue before substantial feature work.
- Issue bodies are free-form. A clear goal, the evidence behind it, the intended scope, and the conditions that prove completion are welcome, and they are never enforced by a form.
- Keep changes focused on local Apple Silicon model serving.
- Prefer direct code and fewer layers over new frameworks, facades, managers, or services.
- Do not reduce model precision or hide changed model behavior behind an optimization.

## Privacy

Never submit credentials, authorization headers, prompt contents, personal paths, local model inventories, proprietary model files, private repository references, or machine-specific logs.

Use temporary directories and fictional placeholder paths in tests. Model-artifact tests must discover user-configured artifacts rather than hardwire a workstation location.

## Tests

Add the smallest meaningful behavior test for functional changes. Every test process must have a maximum 120-second timeout and visible progress.

Run focused tests while developing. Before proposing a pull request, run:

    cargo fmt --all -- --check
    cargo test-integration-and-rest
    scripts/ci/test-macos-menu-contracts.sh

Run direct MLX or model-artifact acceptance only when the change crosses those boundaries. State exactly which checks ran and which hardware-dependent checks could not run.

Routine integration, Representational State Transfer (REST), and commit verification tests retain one stable Cargo graph for warm iteration. Commit verification preserves caller-selected Cargo and compiler-cache configuration. Release and acceptance Cargo commands can use marker-owned disposable targets that are removed automatically. Use the `full-debug` profile only when complete packed symbols are required.

Cargo aliases in `.cargo/config.toml` are string-valued on purpose. Cargo merges the config file of every ancestor directory and joins array values across them, while a string from the nearest file replaces the outer value; an array alias defined in both this file and an ancestor `.cargo/config.toml` would therefore concatenate and double.

Historical Rust generations without an Astronomical ownership marker remain Cargo-owned. Remove them only through a deliberate whole-target `cargo clean` after other Cargo processes finish; never delete hashed generations by age. Suspected retired native CMake output is measurable with `cargo xtask scan-retired-native-output`.

## Pull Requests

- Link one or more existing Astronomical issues under `## Linked issue`; grouped forms such as `Fixes #N and #M` or `Closes #N, #M` are accepted. Use `Fixes #N`, `Closes #N`, or `Resolves #N` when merging should close the issue, `Refs #N` when it should remain open, and a context phrase such as `part of #N` (or a bare mention of an issue labeled `umbrella` or `epic`) for umbrella and tracking relationships that are real but not closing. Every issue number mentioned must carry one of these relationships; the compliance check reports the exact line of any bare mention with a suggested replacement, and warns when a closing keyword targets an already-closed issue. Pull requests are not valid substitutes for issues.
- Explain the user-visible or engineering problem.
- Describe the smallest implemented solution.
- Include verification evidence.
- Identify performance, memory, precision, licensing, or provenance effects.
- Keep unrelated cleanup out of the change.

Contributions are licensed under the repository Apache License 2.0 terms.
