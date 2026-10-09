# CI / CD Scripts

This folder contains scripts invoked by GitHub Actions workflows or actions. Keep these separate from the developer-run scripts at the root of `scripts/`; that root remains the home for local development and acceptance commands.

## Scripts

| Script | Role |
|---|---|
| `bootstrap-native-dependencies.sh` | Provision pinned MLX and MLX-C archives before CMake |
| `prewarm-native-build.sh` | Build the native runtime before Cargo compilation |
| `native-build-cache-fingerprint.sh` | Emit the compatibility identity for the native build store |
| `save-sccache-cache.sh` | Decide whether a trimmed sccache directory may be uploaded |
| `prune-ci-caches.sh` | Prune surplus GitHub Actions caches |
| `ci-step-timing.sh` | Record named workflow-step timing segments |
| `publish-ci-timing-summary.sh` | Write the workflow timing table to the run summary |
| `report-build-cache-restoration.sh` | Report cache-operation outcomes in hosted action logs |
| `classify-ci-change-scope.sh` | Determine the event-specific CI verification scope |
| `install-verification-tools.sh` | Install pinned cargo-about and sccache binaries |
| `generate-rust-dependency-notices.sh` | Generate or check the Rust dependency notices |
| `test-macos-menu-contracts.sh` | Run menu contract checks in the macOS workflow |

The CI cache/workflow contract test remains in `tests/scripts/test-ci-native-cache-coordination.sh`, alongside the other script tests.
