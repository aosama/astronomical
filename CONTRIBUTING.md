# Contributing to Astronomical

## Before Starting

- Open or join an issue before substantial feature work.
- Keep changes focused on local Apple Silicon model serving.
- Prefer direct code and fewer layers over new frameworks, facades, managers, or services.
- Do not reduce model precision or hide changed model behavior behind an optimization.

## Privacy

Never submit credentials, authorization headers, prompt contents, personal paths, local model inventories, proprietary model files, private repository references, or machine-specific logs.

Use temporary directories and fictional placeholder paths in tests. Model-artifact tests must discover user-configured artifacts rather than hardwire a workstation location.

## Tests

Add the smallest meaningful behavior test for functional changes. Every test process must have a maximum 120-second timeout and visible progress.

Run focused tests while developing. Before proposing a pull request, run:

    scripts/verify-before-commit.sh
    scripts/test-macos-menu-contracts.sh

Run direct MLX or model-artifact qualification only when the change crosses those boundaries. State exactly which checks ran and which hardware-dependent checks could not run.

Routine hermetic, Representational State Transfer (REST), and commit verification tests retain one stable Cargo graph for warm iteration. Commit verification preserves caller-selected Cargo and compiler-cache configuration. Release and qualification commands create marker-owned disposable targets and remove them automatically; list named qualification commands with `scripts/run-disposable-cargo-journey.sh --list`. Use the `full-debug` profile only when complete packed symbols are required.

Preview retired native CMake output with `scripts/clean-retired-cargo-native-output.sh --dry-run`. Its explicit `--apply` mode holds every affected Cargo profile lock and preserves generated bindings and diagnostics.

Historical Rust generations without an Astronomical ownership marker remain Cargo-owned. Remove them only through a deliberate whole-target `cargo clean` after other Cargo processes finish; never delete hashed generations by age.

## Pull Requests

- Link exactly one existing Astronomical issue under `## Linked issue`. Use `Fixes #N`, `Closes #N`, or `Resolves #N` when merging should close the issue, and `Refs #N` when it should remain open. Pull requests are not valid substitutes for issues.
- Explain the user-visible or engineering problem.
- Describe the smallest implemented solution.
- Include verification evidence.
- Identify performance, memory, precision, licensing, or provenance effects.
- Keep unrelated cleanup out of the change.

Contributions are licensed under the repository Apache License 2.0 terms.
