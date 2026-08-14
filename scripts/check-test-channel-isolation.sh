#!/usr/bin/env sh

# Prevents serving and qualification tests from reaching active Stable state.

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
    forbidden_pattern='load_from_(default_location|home_directory)\(|AstronomicalRuntimeInstance::Stable|~/\.astronomical("|/)|"\.astronomical("|/)'
    if forbidden_matches="$(
        rg --line-number --glob '*.rs' "$forbidden_pattern" \
            "${repository_root}/apps/inference-worker/tests" \
            "${repository_root}/apps/supervisor/tests" \
            "${repository_root}/crates/model-serving/tests" \
            "${repository_root}/experimental" 2>&1
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
