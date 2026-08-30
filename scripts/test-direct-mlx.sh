#!/usr/bin/env sh

# Runs only the direct-MLX contract binaries so unrelated hermetic and
# model-artifact acceptance modules never enter this direct-MLX graph.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if [ "$#" -ne 0 ]; then
    print_error "scripts/test-direct-mlx.sh does not accept arguments"
    exit 2
fi

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
if [ "${ASTRONOMICAL_CARGO_TARGET_LIFECYCLE:-}" != "disposable" ]; then
    exec "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
        --lane direct-mlx -- "${repository_root}/scripts/test-direct-mlx.sh"
fi
CDPATH='' cd -- "$repository_root"

if command -v timeout >/dev/null 2>&1; then
    timeout_executable="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_executable="$(command -v gtimeout)"
else
    print_error "GNU timeout is required; install Homebrew coreutils"
    exit 1
fi

started_at_seconds="$(date +%s)"
printf '%s\n' "[direct-mlx-tests] status=start timeout_seconds=120 warm_test_eta_seconds=15"

# One test thread keeps mutable environment and allocator-policy tests
# deterministic inside each direct-MLX test process.
if "$timeout_executable" --foreground -k 5s 120s \
    cargo --verbose test \
        --package astronomical-model-serving \
        --package astronomical-runtime-integration \
        --features astronomical-model-serving/direct-mlx,astronomical-runtime-integration/mlx \
        --test direct_mlx_tests \
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
