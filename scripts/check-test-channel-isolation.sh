#!/usr/bin/env sh

# Prevents serving and acceptance tests from reaching active Stable state.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "check-test-channel-isolation.sh does not accept arguments"
        exit 2
    fi
    command -v rg >/dev/null 2>&1 || {
        print_error "ripgrep is required for test channel isolation checks"
        exit 2
    }

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    check_started_at="$(date +%s)"
    printf '%s step=test-channel-isolation status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    forbidden_pattern='load_from_home_directory\(|AstronomicalRuntimeInstance::Stable|~/\.astronomical("|/)|"\.astronomical("|/)'
    # Scan every package test tree, not a fixed trio, so a contract drift in
    # any crate's tests fails this check instead of being silently skipped.
    # The directory list lives in the positional parameters because the script
    # runs under POSIX sh, which has no arrays; each glob match is checked with
    # [ -d ] so a glob with no match drops out instead of reaching rg as text.
    set --
    for search_glob in \
        "${repository_root}/apps/*/tests" \
        "${repository_root}/crates/*/tests" \
        "${repository_root}/tests/*/tests" \
        "${repository_root}/tests/fixtures/*/src"; do
        for candidate in $search_glob; do
            if [ -d "$candidate" ]; then
                set -- "$@" "$candidate"
            fi
        done
    done
    if [ "$#" -gt 0 ] && forbidden_matches="$(
        rg --line-number --glob '*.rs' "$forbidden_pattern" "$@" 2>&1
    )"
    then
        printf '%s\n' "$forbidden_matches" >&2
        print_error "tests must use Development configuration or isolated Development-shaped state"
        exit 1
    else
        ripgrep_exit_code=$?
    fi
    [ "$ripgrep_exit_code" -eq 1 ] || {
        print_error "test channel isolation scan failed: ${forbidden_matches}"
        exit "$ripgrep_exit_code"
    }
    printf '%s step=test-channel-isolation status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(( $(date +%s) - check_started_at ))"
}

main "$@"
