#!/usr/bin/env sh

# Separates compilation from execution so cold acceptance builds can finish
# without weakening the repository-wide 120-second test-process boundary.

set -eu

readonly COMPILE_TIMEOUT_SECONDS=600
readonly DEFAULT_TEST_TIMEOUT_SECONDS=120
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-$DEFAULT_TEST_TIMEOUT_SECONDS}"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

# Scans the caller's arguments for libtest threading options. Real-model ignored
# journeys each load model weights into wired GPU memory, so parallel journeys
# multiply that demand past the machine's physical limit and hard-panic the whole
# system. Reuses the global variables ignored_tests_requested and caller_test_threads.
scan_test_threading_arguments() {
    ignored_tests_requested=0
    caller_test_threads=""
    threads_value_is_pending=0
    for argument in "$@"; do
        if [ "$threads_value_is_pending" -eq 1 ]; then
            caller_test_threads="$argument"
            threads_value_is_pending=0
            continue
        fi
        case "$argument" in
            --ignored)
                ignored_tests_requested=1
                ;;
            --test-threads=*)
                caller_test_threads="${argument#--test-threads=}"
                ;;
            --test-threads)
                threads_value_is_pending=1
                ;;
        esac
    done
}

# Rejects parallel execution for ignored tests before any compilation starts.
enforce_serial_ignored_tests() {
    if [ "$ignored_tests_requested" -ne 1 ]; then
        return
    fi
    if [ -n "$caller_test_threads" ] && [ "$caller_test_threads" != "1" ]; then
        print_error "refusing to run ignored tests with --test-threads ${caller_test_threads}: each ignored test may load real model weights into wired GPU memory and parallel journeys exceed the machine's physical memory limit"
        exit 2
    fi
}

main() {
    if [ "$#" -lt 2 ] || [ "$1" != "cargo" ] || [ "$2" != "test" ]; then
        print_error "expected a cargo test command"
        exit 2
    fi
    shift 2

    scan_test_threading_arguments "$@"
    enforce_serial_ignored_tests

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

    # Serial enforcement must mutate main's own positional parameters here: a
    # set -- inside a function would only affect that function's scope. A caller
    # cannot parallelize real-model ignored tests by omitting or reshaping args.
    if [ "$ignored_tests_requested" -eq 1 ] && [ -z "$caller_test_threads" ]; then
        printf '%s\n' "[bounded-cargo-test] phase=serial-enforcement status=enforced reason=ignored-tests-may-load-real-model-weights"
        seen_argument_separator=0
        for argument in "$@"; do
            if [ "$argument" = "--" ]; then
                seen_argument_separator=1
            fi
        done
        if [ "$seen_argument_separator" -eq 1 ]; then
            set -- "$@" "--test-threads=1"
        else
            set -- "$@" "--" "--test-threads=1"
        fi
    fi

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
