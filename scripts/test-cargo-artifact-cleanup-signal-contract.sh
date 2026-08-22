#!/usr/bin/env sh

# Proves that interruption during storage measurement retains one owned target
# instead of recursively starting a second cleanup over active filesystem work.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""
LIFECYCLE_PROCESS_ID=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${LIFECYCLE_PROCESS_ID:-}" ]; then
        kill "$LIFECYCLE_PROCESS_ID" 2>/dev/null || true
        wait "$LIFECYCLE_PROCESS_ID" 2>/dev/null || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        rm -rf "$SANDBOX_DIRECTORY"
    fi
}
trap cleanup 0

main() {
    command -v mktemp >/dev/null 2>&1 || {
        print_error "required command is unavailable: mktemp"
        exit 2
    }
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 2
    fi

    if [ "${ASTRONOMICAL_SIGNAL_CONTRACT_BOUNDED:-}" != "true" ]; then
        export ASTRONOMICAL_SIGNAL_CONTRACT_BOUNDED=true
        exec "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" "$0"
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    subject="${repository_root}/scripts/run-in-disposable-cargo-target.sh"
    # Commit verification has its own outer target, but this contract must
    # create and interrupt a separate owner to exercise cleanup re-entry.
    unset ASTRONOMICAL_CARGO_TARGET_LIFECYCLE
    unset ASTRONOMICAL_CARGO_TARGET_LANE
    unset CARGO_TARGET_DIR
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-cleanup-signal.XXXXXX")"
    lane_root="${SANDBOX_DIRECTORY}/cargo-lanes"
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    target_record="${SANDBOX_DIRECTORY}/target-record"
    cleanup_ready="${SANDBOX_DIRECTORY}/cleanup-ready"
    lifecycle_log="${SANDBOX_DIRECTORY}/lifecycle.log"
    mkdir -p "$lane_root" "$fake_command_directory"

    cat > "${fake_command_directory}/du" <<'DU'
#!/usr/bin/env sh
set -eu
printf '4\t%s\n' "$2"
printf '%s\n' ready > "${ASTRONOMICAL_TEST_CLEANUP_READY:?}"
exec sleep 30
DU
    cat > "${SANDBOX_DIRECTORY}/capture-target.sh" <<'FIXTURE'
#!/usr/bin/env sh
set -eu
printf '%s\n' "${CARGO_TARGET_DIR:?}" > "${ASTRONOMICAL_TEST_TARGET_RECORD:?}"
FIXTURE
    chmod +x "${fake_command_directory}/du" "${SANDBOX_DIRECTORY}/capture-target.sh"

    printf '%s\n' '[cargo-artifact-cleanup-signal-test] status=start'
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        ASTRONOMICAL_TEST_CLEANUP_READY="$cleanup_ready" \
        ASTRONOMICAL_TEST_TARGET_RECORD="$target_record" \
        PATH="${fake_command_directory}:${PATH}" \
        "$subject" --lane cleanup-signal -- \
        "${SANDBOX_DIRECTORY}/capture-target.sh" > "$lifecycle_log" 2>&1 &
    LIFECYCLE_PROCESS_ID="$!"

    readiness_attempts=0
    while [ ! -f "$cleanup_ready" ] && [ "$readiness_attempts" -lt 50 ]; do
        sleep 0.1
        readiness_attempts=$((readiness_attempts + 1))
    done
    [ -f "$cleanup_ready" ] || {
        print_error "cleanup measurement fixture did not become ready"
        exit 1
    }

    kill -TERM "$LIFECYCLE_PROCESS_ID"
    lifecycle_exit_status=0
    wait "$LIFECYCLE_PROCESS_ID" || lifecycle_exit_status=$?
    LIFECYCLE_PROCESS_ID=""
    [ "$lifecycle_exit_status" -eq 143 ] || {
        print_error "interrupted cleanup returned unexpected status ${lifecycle_exit_status}"
        exit 1
    }

    original_target="$(cat "$target_record")"
    set -- "${original_target}.removing."*
    retained_target="$1"
    [ -d "$retained_target" ] && [ -f "${retained_target}/.astronomical-disposable-cargo-target" ] || {
        print_error "interrupted measurement did not retain its marker-owned target"
        exit 1
    }
    grep -F 'status=cleanup-skipped-retained' "$lifecycle_log" >/dev/null || {
        print_error "interrupted measurement did not report retained cleanup ownership"
        exit 1
    }
    rm -rf "$retained_target"
    [ -z "$(ls -A "$lane_root")" ] || {
        print_error "signal contract retained an unreviewed lane artifact"
        exit 1
    }
    printf '%s\n' '[cargo-artifact-cleanup-signal-test] status=success'
}

main "$@"
