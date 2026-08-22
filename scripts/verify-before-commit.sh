#!/usr/bin/env sh

# Runs the complete local commit gate on Cargo's stable routine artifact graph.
# Release and model qualification remain isolated behind disposable journeys.

set -eu

readonly COMPILE_TIMEOUT_SECONDS=600
readonly TEST_TIMEOUT_SECONDS=120
readonly TOTAL_STEP_COUNT=16

COMPLETED_STEP_COUNT=0

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/verify-before-commit.sh"
    printf '%s\n' ""
    printf '%s\n' "Runs formatting, repository contracts, application journeys, and the required Rust test boundaries."
}

require_command() {
    required_command_name="$1"
    command -v "$required_command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${required_command_name}"
        exit 2
    }
}

resolve_timeout_executable() {
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 2
    fi
}

run_step() {
    step_name="$1"
    step_timeout_seconds="$2"
    shift 2

    step_started_at_seconds="$(date +%s)"
    printf '[commit-verification] step=%s status=start completed=%s/%s timeout_seconds=%s started_at=%s\n' \
        "$step_name" "$COMPLETED_STEP_COUNT" "$TOTAL_STEP_COUNT" \
        "$step_timeout_seconds" "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    if "$timeout_executable" --foreground -k 5s "${step_timeout_seconds}s" "$@"; then
        step_exit_code=0
    else
        step_exit_code=$?
        printf '[commit-verification] step=%s status=failed exit_code=%s elapsed_seconds=%s ended_at=%s\n' \
            "$step_name" "$step_exit_code" "$(( $(date +%s) - step_started_at_seconds ))" \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" >&2
        return "$step_exit_code"
    fi

    COMPLETED_STEP_COUNT=$((COMPLETED_STEP_COUNT + 1))
    printf '[commit-verification] step=%s status=success completed=%s/%s elapsed_seconds=%s ended_at=%s\n' \
        "$step_name" "$COMPLETED_STEP_COUNT" "$TOTAL_STEP_COUNT" \
        "$(( $(date +%s) - step_started_at_seconds ))" "$(date '+%Y-%m-%dT%H:%M:%S%z')"
}

main() {
    if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
        print_usage
        return
    fi
    if [ "$#" -ne 0 ]; then
        print_error "verify-before-commit.sh does not accept arguments"
        print_usage >&2
        exit 2
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    CDPATH='' cd -- "$repository_root"
    require_command cargo
    require_command date
    require_command node
    require_command sysctl
    resolve_timeout_executable

    logical_cpu_count="$(sysctl -n hw.logicalcpu)"
    case "$logical_cpu_count" in
        ''|*[!0-9]*|0)
            print_error "sysctl did not return a positive logical CPU count"
            exit 2
            ;;
    esac
    export CARGO_BUILD_JOBS="$logical_cpu_count"

    verification_started_at_seconds="$(date +%s)"
    printf '[commit-verification] status=start steps=%s cargo_target=%s rustc_wrapper=%s build_jobs=%s started_at=%s\n' \
        "$TOTAL_STEP_COUNT" "${CARGO_TARGET_DIR:-target}" "${RUSTC_WRAPPER:-none}" \
        "$CARGO_BUILD_JOBS" "$(date '+%Y-%m-%dT%H:%M:%S%z')"

    run_step rust-dependency-notices "$TEST_TIMEOUT_SECONDS" scripts/generate-rust-dependency-notices.sh --check
    run_step commit-release-isolation "$TEST_TIMEOUT_SECONDS" scripts/test-commit-release-isolation.sh
    run_step ci-native-cache-contract "$TEST_TIMEOUT_SECONDS" scripts/test-ci-native-cache-coordination.sh
    run_step cargo-artifact-lifecycle-contract "$TEST_TIMEOUT_SECONDS" scripts/test-cargo-artifact-lifecycle-contract.sh
    run_step cargo-artifact-cleanup-signal-contract "$TEST_TIMEOUT_SECONDS" scripts/test-cargo-artifact-cleanup-signal-contract.sh
    run_step legacy-native-output-cleanup-contract "$TEST_TIMEOUT_SECONDS" scripts/test-retired-cargo-native-output-cleanup.sh
    run_step commit-verification-contract "$TEST_TIMEOUT_SECONDS" scripts/test-verify-before-commit-contract.sh
    run_step format "$TEST_TIMEOUT_SECONDS" cargo fmt --all -- --check
    run_step test-channel-isolation-checker "$TEST_TIMEOUT_SECONDS" scripts/test-channel-isolation-checker-contract.sh
    run_step test-macos-app-validation-contract "$TEST_TIMEOUT_SECONDS" scripts/test-validate-macos-app-contract.sh
    run_step test-channel-isolation "$TEST_TIMEOUT_SECONDS" scripts/check-test-channel-isolation.sh
    run_step test-macos-menu-contracts "$TEST_TIMEOUT_SECONDS" scripts/test-macos-menu-contracts.sh
    run_step test-pull-request-policy-contracts "$TEST_TIMEOUT_SECONDS" node \
        --test --test-reporter=spec .github/scripts/pull-request-issue-compliance.test.js
    run_step test-observatory-contracts "$TEST_TIMEOUT_SECONDS" node --test --test-reporter=spec \
        apps/supervisor/console/console.test.js \
        apps/supervisor/console/library.test.js
    run_step compile-rust "$COMPILE_TIMEOUT_SECONDS" cargo verify-commit-rust \
        --timings --no-run --jobs "$logical_cpu_count"
    run_step run-rust "$TEST_TIMEOUT_SECONDS" cargo verify-commit-rust \
        --jobs "$logical_cpu_count" -- --quiet --test-threads "$logical_cpu_count"

    [ "$COMPLETED_STEP_COUNT" -eq "$TOTAL_STEP_COUNT" ] || {
        print_error "verification plan completed ${COMPLETED_STEP_COUNT} of ${TOTAL_STEP_COUNT} steps"
        exit 1
    }
    printf '[commit-verification] status=success steps=%s elapsed_seconds=%s ended_at=%s\n' \
        "$COMPLETED_STEP_COUNT" "$(( $(date +%s) - verification_started_at_seconds ))" \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')"
}

main "$@"
