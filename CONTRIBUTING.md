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

## Pull Requests

- Explain the user-visible or engineering problem.
- Describe the smallest implemented solution.
- Include verification evidence.
- Identify performance, memory, precision, licensing, or provenance effects.
- Keep unrelated cleanup out of the change.

Contributions are licensed under the repository Apache License 2.0 terms.
