# CI Scripts

Repository tooling is typed Rust in the `xtask` crate (`cargo xtask <command>`, catalogued in `.cargo/config.toml`). The few scripts left here stay POSIX shell on purpose. They run where no Rust toolchain exists, so porting them to xtask would force a Rust install and an xtask compile into jobs that otherwise never build Rust, adding minutes to the CI critical path.

## Scripts that stay shell, and why

| Script | Role | Why it stays shell |
|---|---|---|
| `ci-step-timing.sh` | Records named step-timing segments | It runs as the first and last command of almost every step in every job, including the Swift and Node jobs and steps that run before any toolchain is installed. |
| `publish-ci-timing-summary.sh` | Writes the timing table to the run summary | It is each job's final `if: always()` step, so it must work even when the job failed before a toolchain was set up. |
| `report-build-cache-restoration.sh` | Reports cache-restore outcomes in action logs | It runs inside the `astronomical-cache` composite action, before any toolchain exists. |
| `tests/test-ci-native-cache-coordination.sh` | Contract for the cache-restoration reporter | It tests the shell reporter above directly, so it lives and dies with it. |
| `timeout` | POSIX `timeout` for runners without GNU coreutils | The remaining shell contracts in `verification/tooling_contracts/` bound their subjects with it. |

## Tooling that moved to xtask

Former scripts in this folder and at the root of `scripts/` are now xtask commands:
- `cargo xtask prewarm-native-build`
- `cargo xtask check-test-channel-isolation`, which now also scans `verification/*/tests` and runs on the real repository in CI
- `cargo xtask menu-contracts`, for local use; CI runs the same `swift test` inline because its Swift job has no Rust
- `cargo xtask mlx-c-coverage-inventory`

The tier pyramid is local: `cargo xtask tier-pyramid` reports the repository's per-tier test distribution against `tier-pyramid-baseline.json`. Its behaviour on a fixture tree is an xtask unit test.

The workflow structural contracts are typed Rust tests (`cargo test -p xtask --test workflow_structural_contract` and its siblings). Every xtask unit test runs on the macOS build job (`cargo test -p xtask --bin xtask`).
