#!/usr/bin/env sh

# Refreshes crates/mlx-c-fork/COVERAGE_INVENTORY.md from live data: the
# provisioned pinned MLX-C headers and the merged bridge inventory of the
# published mlx-c-rust crate plus the astronomical-mlx-c-fork delta. The
# artifact is the human-reviewable record behind the coverage contract.
# Typed Rust tooling owns input validation, progress, and bounded execution.

set -eu

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
CDPATH='' cd -- "$repository_root"
exec cargo xtask mlx-c-coverage-inventory "$@"
