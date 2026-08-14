#!/usr/bin/env sh

# Proves the source guard accepts Development and rejects alternate Stable access paths.

set -eu

readonly CHECK_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe guard-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

run_checker_expect_failure() {
    expected_diagnostic="$1"
    output_file="$2"
    if timeout "$CHECK_TIMEOUT_SECONDS" "$sandbox_checker" > "$output_file" 2>&1; then
        print_error "channel-isolation checker unexpectedly accepted Stable access"
        exit 1
    else
        checker_exit_code=$?
    fi
    [ "$checker_exit_code" -eq 1 ] || {
        print_error "checker returned ${checker_exit_code}, expected policy failure 1"
        exit 1
    }
    grep -F "$expected_diagnostic" "$output_file" >/dev/null || {
        print_error "checker did not report the expected Stable access: ${expected_diagnostic}"
        exit 1
    }
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "test-channel-isolation-checker-contract.sh does not accept arguments"
        exit 2
    fi
    for required_command in timeout mktemp grep; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-channel-guard.XXXXXX")"
    mkdir -p \
        "${SANDBOX_DIRECTORY}/scripts" \
        "${SANDBOX_DIRECTORY}/apps/inference-worker/tests" \
        "${SANDBOX_DIRECTORY}/apps/supervisor/tests" \
        "${SANDBOX_DIRECTORY}/crates/model-serving/tests" \
        "${SANDBOX_DIRECTORY}/experimental"
    cp "${repository_root}/scripts/check-test-channel-isolation.sh" \
        "${SANDBOX_DIRECTORY}/scripts/check-test-channel-isolation.sh"
    sandbox_checker="${SANDBOX_DIRECTORY}/scripts/check-test-channel-isolation.sh"
    safe_fixture="${SANDBOX_DIRECTORY}/apps/inference-worker/tests/channel_fixture.rs"

    printf '%s\n' '[channel-isolation-guard-test] case=development-is-accepted status=start'
    printf '%s\n' 'AstronomicalConfig::load_from_development_location();' > "$safe_fixture"
    timeout "$CHECK_TIMEOUT_SECONDS" "$sandbox_checker" >/dev/null
    printf '%s\n' '[channel-isolation-guard-test] case=development-is-accepted status=success'

    printf '%s\n' '[channel-isolation-guard-test] case=stable-loader-is-rejected status=start'
    printf '%s\n' 'AstronomicalConfig::load_from_default_location();' > "$safe_fixture"
    run_checker_expect_failure \
        'AstronomicalConfig::load_from_default_location' \
        "${SANDBOX_DIRECTORY}/stable-loader.log"
    printf '%s\n' '[channel-isolation-guard-test] case=stable-loader-is-rejected status=success'

    printf '%s\n' '[channel-isolation-guard-test] case=typed-stable-instance-is-rejected status=start'
    printf '%s\n' 'let channel = AstronomicalRuntimeInstance::Stable;' > "$safe_fixture"
    run_checker_expect_failure \
        'AstronomicalRuntimeInstance::Stable' \
        "${SANDBOX_DIRECTORY}/typed-stable.log"
    printf '%s\n' '[channel-isolation-guard-test] case=typed-stable-instance-is-rejected status=success'

    printf '%s\n' '[channel-isolation-guard-test] case=stable-state-label-is-rejected status=start'
    printf '%s\n' 'let state = "~/.astronomical";' > "$safe_fixture"
    run_checker_expect_failure '"~/.astronomical"' "${SANDBOX_DIRECTORY}/stable-state.log"
    printf '%s\n' '[channel-isolation-guard-test] case=stable-state-label-is-rejected status=success'
}

main "$@"
