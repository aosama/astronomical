# CI / CD Scripts

This folder contains scripts invoked by GitHub Actions workflows or actions. Keep these separate from the developer-run scripts at the root of `scripts/`; that root remains the home for local development and acceptance commands.

## Scripts

| Script | Role |
|---|---|
| `prewarm-native-build.sh` | Build the native runtime before Cargo compilation |
| `ci-step-timing.sh` | Record named workflow-step timing segments |
| `publish-ci-timing-summary.sh` | Write the workflow timing table to the run summary |
| `report-build-cache-restoration.sh` | Report cache-operation outcomes in hosted action logs |
| `test-macos-menu-contracts.sh` | Run menu contract checks in the macOS workflow |

The tier-pyramid, source-size, dependency-notices, native-fingerprint,
change-classification, cache-pruning, sccache-save-gate,
verification-tool-installer, and native-dependency-bootstrap tooling moved
into the `xtask` workspace crate:
`cargo xtask tier-pyramid [--update-baseline]`, `cargo xtask source-size
[--staged] [root]`, `cargo xtask notices [--check]`, `cargo xtask
native-fingerprint [--source-only] --profile core|core+memory-contract
[root]`, `cargo xtask classify-scope`, `cargo xtask prune-caches
[--dry-run] [--repo owner/name]`, `cargo xtask save-sccache
<primary-cache-key>`, `cargo xtask install-verification-tools [--prefix
DIR]`, and `cargo xtask bootstrap-native-dependencies [--cache-dir
ABSOLUTE_PATH] [--verify]` (see `.cargo/config.toml`). The native identity
policy itself lives in
the `astronomical-native-identity` crate (`crates/native-identity`),
consumed in-process by the Cargo build scripts.

The workflow structural contract is the typed Rust test
`cargo test -p xtask --test workflow_structural_contract`; the restoration-reporter
classification contract stays in `scripts/ci/tests/test-ci-native-cache-coordination.sh`,
alongside the other CI contract tests; the remaining script tests stay in `tests/tooling_contracts/`.
