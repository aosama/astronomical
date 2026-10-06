#!/usr/bin/env sh

# Direct-MLX lane: one bounded cargo test invocation for the two feature-gated
# contract binaries, so unrelated hermetic and model-artifact acceptance
# modules never enter this direct-MLX graph. The same single Cargo invocation
# also compiles and runs the hermetic MLX-C coverage contract binary, so
# bridge drift against the pinned headers fails this lane instead of shipping
# silently; that binary is CPU-only and finishes in seconds, keeping the lane
# inside its budget.
#
# Command run (inside a disposable Cargo target owned by the lane):
#   timeout --foreground -k 5s 120s cargo --verbose test \
#     -p astronomical-model-serving -p astronomical-runtime-integration \
#     --features astronomical-model-serving/direct-mlx,astronomical-runtime-integration/mlx \
#     --test direct_mlx_tests --test mlx_c_coverage_contract_tests -- --test-threads=1
#
# One test thread keeps mutable environment and allocator-policy tests
# deterministic inside each direct-MLX test process.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if command -v timeout >/dev/null 2>&1; then
    timeout_executable="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_executable="$(command -v gtimeout)"
else
    print_error "GNU timeout is required; install Homebrew coreutils"
    exit 1
fi

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"

started_at_seconds="$(date +%s)"
printf '%s\n' "[direct-mlx-tests] status=start timeout_seconds=120 warm_test_eta_seconds=15"

# The disposable-target runner gives this lane an owned CARGO_TARGET_DIR and
# removes it when the cargo invocation exits; it also reuses an existing lane
# target when invoked from inside one instead of nesting.
if "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
        --lane direct-mlx -- \
        "$timeout_executable" --foreground -k 5s 120s \
        cargo --verbose test \
            --package astronomical-model-serving \
            --package astronomical-runtime-integration \
            --features astronomical-model-serving/direct-mlx,astronomical-runtime-integration/mlx \
            --test direct_mlx_tests \
            --test mlx_c_coverage_contract_tests \
            -- \
            --test-threads=1
then
    finished_at_seconds="$(date +%s)"
    elapsed_seconds=$((finished_at_seconds - started_at_seconds))
    printf '%s\n' "[direct-mlx-tests] status=success elapsed_seconds=${elapsed_seconds}"
    exit 0
else
    test_status=$?
fi

if [ "$test_status" -eq 124 ] || [ "$test_status" -eq 137 ]; then
    print_error "direct MLX tests exceeded the 120-second safety timeout"
else
    print_error "direct MLX tests failed with status ${test_status}"
fi
exit "$test_status"
