#!/usr/bin/env sh

# Proves the cache-restoration reporter classifies every restore outcome
# consistently and rejects inconsistent states. The workflow structural
# contract that used to live here is the typed Rust test
# `cargo test -p xtask --test workflow_structural_contract`; the
# native-fingerprint and change-scope journeys are unit tests beside their
# sources in `crates/native-identity` and `xtask` (issue #1250).

set -eu

SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe CI cache-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

assert_cache_classification() {
    expected_classification="$1"
    cache_step_outcome="$2"
    cache_hit="$3"
    matched_key="$4"
    report_output="$(
        CACHE_OWNER='native-build' \
        CACHE_OPERATION='restore' \
        CACHE_STEP_OUTCOME="$cache_step_outcome" \
        CACHE_HIT="$cache_hit" \
        CACHE_MATCHED_KEY="$matched_key" \
        CACHE_PRIMARY_KEY='astronomical-v2-native-build-current' \
        CACHE_STARTED_AT_EPOCH_SECONDS='100' \
        CACHE_FINISHED_AT_EPOCH_SECONDS='112' \
        GITHUB_STEP_SUMMARY="${SANDBOX_DIRECTORY}/step-summary.md" \
        "${repository_root}/scripts/ci/report-build-cache-restoration.sh"
    )"
    case "$report_output" in
        *"owner=native-build operation=restore classification=${expected_classification} elapsed_seconds=12"*) ;;
        *)
            print_error "cache state was not classified as ${expected_classification}: ${report_output}"
            exit 1
            ;;
    esac
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "test-ci-native-cache-coordination.sh does not accept arguments"
        exit 2
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-ci-native-cache.XXXXXX")"

    printf '%s\n' '[ci-native-cache-test] case=cache-classification status=start'
    assert_cache_classification primary success true 'astronomical-v2-native-build-current'
    assert_cache_classification fallback success false 'astronomical-v2-native-build-previous'
    assert_cache_classification miss success '' ''
    assert_cache_classification error failure '' ''
    if CACHE_OWNER='native-build' \
        CACHE_OPERATION='restore' \
        CACHE_STEP_OUTCOME='success' \
        CACHE_HIT='true' \
        CACHE_MATCHED_KEY='different-key' \
        CACHE_PRIMARY_KEY='astronomical-v2-native-build-current' \
        CACHE_STARTED_AT_EPOCH_SECONDS='100' \
        CACHE_FINISHED_AT_EPOCH_SECONDS='112' \
        "${repository_root}/scripts/ci/report-build-cache-restoration.sh" >/dev/null 2>&1
    then
        print_error "an inconsistent primary cache state was accepted"
        exit 1
    fi
    printf '%s\n' '[ci-native-cache-test] case=cache-classification status=success'
}

main "$@"
