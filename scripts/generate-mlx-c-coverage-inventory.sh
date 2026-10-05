#!/usr/bin/env sh

# Refreshes crates/mlx-c-rust/COVERAGE_INVENTORY.md from live data: the
# provisioned pinned MLX-C headers and the bridge inventory the mlx-c-rust
# build script just compiled. The artifact is the human-reviewable record
# behind the coverage contract; the contract test fails on drift and names
# this script as the refresh command.
#
# The plain shared-target invocation is deliberate: mlx-feature check
# artifacts on the shared target are precedented by
# scripts/compile-mlx-memory-contract-lane.sh, and a disposable target would
# pay a cold compile on every refresh. The 600-second compile-class budget
# exists because a cold mlx-feature compile can exceed the 120-second
# test-class timeout.

set -eu

readonly SCRIPT_PREFIX="[mlx-c-coverage-inventory]"
readonly COMPILE_CLASS_TIMEOUT_SECONDS=600

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if [ "$#" -ne 0 ]; then
    print_error "scripts/generate-mlx-c-coverage-inventory.sh does not accept arguments"
    exit 2
fi

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
CDPATH='' cd -- "$repository_root"

if command -v timeout >/dev/null 2>&1; then
    timeout_executable="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_executable="$(command -v gtimeout)"
else
    print_error "GNU timeout is required; install Homebrew coreutils"
    exit 1
fi

inventory_path="${repository_root}/crates/mlx-c-rust/COVERAGE_INVENTORY.md"

started_at_seconds="$(date +%s)"
printf '%s\n' "${SCRIPT_PREFIX} status=start timeout_seconds=${COMPILE_CLASS_TIMEOUT_SECONDS}"

# The write-mode branch of the inventory test renders the artifact from the
# same live data the assertions use, so the committed file can never diverge
# from what the contract checks.
if MLX_C_COVERAGE_INVENTORY_PATH="${inventory_path}" \
    "$timeout_executable" --foreground -k 5s "${COMPILE_CLASS_TIMEOUT_SECONDS}s" \
    cargo test \
        --package astronomical-runtime-integration \
        --features mlx \
        --test mlx_c_coverage_contract_tests \
        should_keep_the_committed_coverage_inventory_current
then
    printf '%s\n' \
        "${SCRIPT_PREFIX} status=success artifact=${inventory_path} elapsed_seconds=$(( $(date +%s) - started_at_seconds ))"
    exit 0
fi
inventory_status=$?

if [ "$inventory_status" -eq 124 ] || [ "$inventory_status" -eq 137 ]; then
    print_error "coverage inventory generation exceeded the ${COMPILE_CLASS_TIMEOUT_SECONDS}-second compile-class timeout"
else
    print_error "coverage inventory generation failed with status ${inventory_status}"
fi
exit "$inventory_status"
