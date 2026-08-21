#!/usr/bin/env sh

# Separates compilation from execution so cold qualification builds can finish
# without weakening the repository-wide 120-second test-process boundary.

set -eu

readonly COMPILE_TIMEOUT_SECONDS=600
readonly TEST_TIMEOUT_SECONDS=120

print_error() {
    printf '%s\n' "Error: $1" >&2
}

main() {
    if [ "$#" -lt 2 ] || [ "$1" != "cargo" ] || [ "$2" != "test" ]; then
        print_error "expected a cargo test command"
        exit 2
    fi
    shift 2

    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 2
    fi

    compile_started_at_seconds="$(date +%s)"
    printf '%s\n' "[bounded-cargo-test] phase=compile status=start timeout_seconds=${COMPILE_TIMEOUT_SECONDS} started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    "${timeout_executable}" --foreground -k 5s "${COMPILE_TIMEOUT_SECONDS}s" \
        cargo test --no-run "$@"
    printf '%s\n' "[bounded-cargo-test] phase=compile status=success elapsed_seconds=$(( $(date +%s) - compile_started_at_seconds ))"

    test_started_at_seconds="$(date +%s)"
    printf '%s\n' "[bounded-cargo-test] phase=test status=start timeout_seconds=${TEST_TIMEOUT_SECONDS} started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    if "${timeout_executable}" --foreground -k 5s "${TEST_TIMEOUT_SECONDS}s" \
        cargo test "$@"; then
        printf '%s\n' "[bounded-cargo-test] phase=test status=success elapsed_seconds=$(( $(date +%s) - test_started_at_seconds ))"
        return
    else
        test_exit_status="$?"
    fi

    if [ "$test_exit_status" -eq 124 ] || [ "$test_exit_status" -eq 137 ]; then
        print_error "cargo test exceeded the ${TEST_TIMEOUT_SECONDS}-second safety timeout"
    fi
    exit "$test_exit_status"
}

main "$@"
