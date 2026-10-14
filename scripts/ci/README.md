# CI / CD Scripts

This folder contains scripts invoked by GitHub Actions workflows or actions. Keep these separate from the developer-run scripts at the root of `scripts/`; that root remains the home for local development and acceptance commands.

## Scripts

| Script | Role |
|---|---|
| `bootstrap-native-dependencies.sh` | Provision pinned MLX and MLX-C archives before CMake |
| `prewarm-native-build.sh` | Build the native runtime before Cargo compilation |
| `save-sccache-cache.sh` | Decide whether a trimmed sccache directory may be uploaded |
| `prune-ci-caches.sh` | Prune surplus GitHub Actions caches |
| `ci-step-timing.sh` | Record named workflow-step timing segments |
| `publish-ci-timing-summary.sh` | Write the workflow timing table to the run summary |
| `report-build-cache-restoration.sh` | Report cache-operation outcomes in hosted action logs |
| `install-verification-tools.sh` | Install pinned cargo-about and sccache binaries |
| `test-macos-menu-contracts.sh` | Run menu contract checks in the macOS workflow |

The tier-pyramid, source-size, dependency-notices, native-fingerprint, and
change-classification tooling moved into the `xtask` workspace crate:
`cargo xtask tier-pyramid [--update-baseline]`, `cargo xtask source-size
[--staged] [root]`, `cargo xtask notices [--check]`, `cargo xtask
native-fingerprint [--source-only] --profile core|core+memory-contract
[root]`, and `cargo xtask classify-scope` (see `.cargo/config.toml`). The
native identity policy itself lives in the `astronomical-native-identity`
crate (`crates/native-identity`), consumed in-process by the Cargo build
scripts.

The CI cache/workflow contract test lives in `scripts/ci/tests/test-ci-native-cache-coordination.sh`, alongside the other CI contract tests; the remaining script tests stay in `tests/tooling_contracts/`.
