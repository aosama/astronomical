#!/usr/bin/env sh

# Verifies the Stable workflow orchestration without building or launching model binaries.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

assert_call_log() {
    expected_call_log="$1"
    actual_call_log="$2"
    if ! diff -u "$expected_call_log" "$actual_call_log"; then
        print_error "delegated Stable workflow calls did not match"
        exit 1
    fi
}

main() {
    for required_command in timeout mktemp diff; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-stable-workflow.XXXXXX")"
    sandbox_scripts_directory="${SANDBOX_DIRECTORY}/scripts"
    mkdir -p "$sandbox_scripts_directory"
    cp "${repository_root}/scripts/make-install-astronomical-stable-app.sh" \
        "${sandbox_scripts_directory}/make-install-astronomical-stable-app.sh"

    cat > "${sandbox_scripts_directory}/make-astronomical-app.sh" <<'BUILDER'
#!/usr/bin/env sh
printf 'build %s\n' "$*" >> "${CALL_LOG:?CALL_LOG is required}"
exit "${FAKE_BUILD_EXIT_CODE:-0}"
BUILDER
    cat > "${sandbox_scripts_directory}/install-astronomical-stable-app.sh" <<'INSTALLER'
#!/usr/bin/env sh
printf 'install %s\n' "$*" >> "${CALL_LOG:?CALL_LOG is required}"
INSTALLER
    chmod +x "${sandbox_scripts_directory}"/*.sh

    call_log="${SANDBOX_DIRECTORY}/calls.log"
    expected_call_log="${SANDBOX_DIRECTORY}/expected-calls.log"

    printf '%s\n' '[stable-workflow-test] case=build-failure-blocks-install status=start'
    : > "$call_log"
    if CALL_LOG="$call_log" FAKE_BUILD_EXIT_CODE=17 timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${sandbox_scripts_directory}/make-install-astronomical-stable-app.sh" \
        > "${SANDBOX_DIRECTORY}/failure-output.log" 2>&1; then
        print_error "workflow unexpectedly succeeded after the Stable build failed"
        exit 1
    else
        workflow_exit_code=$?
    fi
    [ "$workflow_exit_code" -eq 17 ] || {
        print_error "workflow returned ${workflow_exit_code}, expected the build failure code 17"
        exit 1
    }
    printf '%s\n' 'build --channel stable' > "$expected_call_log"
    assert_call_log "$expected_call_log" "$call_log"
    printf '%s\n' '[stable-workflow-test] case=build-failure-blocks-install status=success'

    printf '%s\n' '[stable-workflow-test] case=successful-build-installs status=start'
    : > "$call_log"
    CALL_LOG="$call_log" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${sandbox_scripts_directory}/make-install-astronomical-stable-app.sh" \
        > "${SANDBOX_DIRECTORY}/success-output.log" 2>&1
    printf '%s\n' 'build --channel stable' 'install ' > "$expected_call_log"
    assert_call_log "$expected_call_log" "$call_log"
    printf '%s\n' '[stable-workflow-test] case=successful-build-installs status=success'

    printf '%s\n' '[stable-workflow-test] case=dry-run-previews-install status=start'
    : > "$call_log"
    CALL_LOG="$call_log" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${sandbox_scripts_directory}/make-install-astronomical-stable-app.sh" --dry-run \
        > "${SANDBOX_DIRECTORY}/dry-run-output.log" 2>&1
    printf '%s\n' 'build --channel stable' 'install --dry-run' > "$expected_call_log"
    assert_call_log "$expected_call_log" "$call_log"
    printf '%s\n' '[stable-workflow-test] case=dry-run-previews-install status=success'
}

main "$@"
