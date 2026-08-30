#!/usr/bin/env sh

# Runs one foreground build journey in an owned Cargo target and removes that
# target afterward. High-churn acceptance and release graphs remain useful
# through sccache without accumulating linked binaries and symbols in the repo.

set -eu

readonly LIFECYCLE_MARKER_NAME=".astronomical-disposable-cargo-target"
readonly LIFECYCLE_MARKER_VERSION="astronomical-disposable-cargo-target-v1"
readonly CHILD_TERMINATION_GRACE_SECONDS=3
readonly KILL_CONFIRMATION_SECONDS=1
readonly PROGRESS_INTERVAL_SECONDS=5

DISPOSABLE_TARGET_DIRECTORY=""
DISPOSABLE_TARGET_MARKER=""
IS_CLEANUP_PENDING="false"
IS_CLEANUP_ACTIVE="false"
CLEANUP_WORKER_PROCESS_ID=""
CLEANUP_PROGRESS_PROCESS_ID=""
CLEANUP_OPERATION=""
MEASURED_ALLOCATED_BYTES=""
CHILD_PROCESS_ID=""
LANE_NAME=""
SELECTED_RUSTC_WRAPPER=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/run-in-disposable-cargo-target.sh --lane NAME -- COMMAND [ARGUMENT ...]"
    printf '%s\n' ""
    printf '%s\n' "Runs a foreground command from the repository root with a unique"
    printf '%s\n' "CARGO_TARGET_DIR, then removes only that marker-owned directory."
}

require_command() {
    required_command_name="$1"
    command -v "$required_command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${required_command_name}"
        exit 1
    }
}

parse_arguments() {
    if [ "$#" -lt 4 ] || [ "$1" != "--lane" ]; then
        print_usage >&2
        exit 2
    fi

    LANE_NAME="$2"
    shift 2
    case "$LANE_NAME" in
        ''|*[!a-z0-9-]*|-*|*-)
            print_error "lane name must contain lowercase letters, digits, and internal hyphens only"
            exit 2
            ;;
    esac

    if [ "$1" != "--" ]; then
        print_usage >&2
        exit 2
    fi
    shift
    [ "$#" -gt 0 ] || {
        print_error "a foreground command is required"
        exit 2
    }

}

validate_owned_target() {
    target_directory="$1"
    expected_lane_name="$2"
    marker_path="${target_directory}/${LIFECYCLE_MARKER_NAME}"

    [ -d "$target_directory" ] && [ ! -L "$target_directory" ] || {
        print_error "refusing to use a missing, non-directory, or symbolic-link Cargo target: ${target_directory}"
        return 1
    }
    [ -f "$marker_path" ] && [ ! -L "$marker_path" ] || {
        print_error "refusing to use Cargo target without an owned lifecycle marker: ${target_directory}"
        return 1
    }
    marker_identity="$(cat "$marker_path")"
    [ "$marker_identity" = "${LIFECYCLE_MARKER_VERSION}:${expected_lane_name}" ] || {
        print_error "refusing Cargo target with an unexpected lifecycle marker: ${target_directory}"
        return 1
    }
}

allocated_bytes() {
    measured_directory="$1"
    measurement_output="${measured_directory}/.astronomical-storage-measurement.$$"
    du -sk "$measured_directory" > "$measurement_output" &
    CLEANUP_WORKER_PROCESS_ID="$!"
    CLEANUP_OPERATION="measuring"
    report_operation_progress "$CLEANUP_WORKER_PROCESS_ID" "$CLEANUP_OPERATION" &
    CLEANUP_PROGRESS_PROCESS_ID="$!"
    measurement_exit_status=0
    wait "$CLEANUP_WORKER_PROCESS_ID" || measurement_exit_status=$?
    kill "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
    wait "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
    CLEANUP_WORKER_PROCESS_ID=""
    CLEANUP_PROGRESS_PROCESS_ID=""
    CLEANUP_OPERATION=""
    if [ "$measurement_exit_status" -ne 0 ]; then
        rm -f -- "$measurement_output"
        print_error "could not measure allocated storage for ${measured_directory}"
        return 1
    fi
    allocated_kibibyte_line="$(cat "$measurement_output")"
    rm -f -- "$measurement_output"
    allocated_kibibytes="${allocated_kibibyte_line%%[[:space:]]*}"
    case "$allocated_kibibytes" in
        ''|*[!0-9]*)
            print_error "could not measure allocated storage for ${measured_directory}"
            return 1
            ;;
    esac
    MEASURED_ALLOCATED_BYTES="$((allocated_kibibytes * 1024))"
}

report_operation_progress() {
    observed_process_id="$1"
    operation_status="$2"
    operation_elapsed_seconds=0
    progress_sleep_process_id=""
    trap 'if [ -n "$progress_sleep_process_id" ]; then kill "$progress_sleep_process_id" 2>/dev/null || true; fi; exit 0' HUP INT TERM
    while :; do
        sleep "$PROGRESS_INTERVAL_SECONDS" &
        progress_sleep_process_id="$!"
        wait "$progress_sleep_process_id" 2>/dev/null || return
        progress_sleep_process_id=""
        kill -0 "$observed_process_id" 2>/dev/null || return
        operation_elapsed_seconds=$((operation_elapsed_seconds + PROGRESS_INTERVAL_SECONDS))
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=${operation_status}-progress elapsed_seconds=${operation_elapsed_seconds}" >&2
    done
}

cleanup_owned_target() {
    [ "$IS_CLEANUP_PENDING" = "true" ] || return 0
    if [ "$IS_CLEANUP_ACTIVE" = "true" ]; then
        print_error "refusing to re-enter active Cargo target cleanup"
        return 1
    fi
    IS_CLEANUP_ACTIVE="true"
    cleanup_started_at_seconds="$(date +%s)"

    if [ ! -e "$DISPOSABLE_TARGET_DIRECTORY" ] && [ ! -L "$DISPOSABLE_TARGET_DIRECTORY" ]; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=already-removed elapsed_seconds=0"
        return 0
    fi

    if ! validate_owned_target "$DISPOSABLE_TARGET_DIRECTORY" "$LANE_NAME"; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-refused target_directory=${DISPOSABLE_TARGET_DIRECTORY}" >&2
        return 1
    fi

    removal_target_directory="${DISPOSABLE_TARGET_DIRECTORY}.removing.$$"
    if [ -e "$removal_target_directory" ] || [ -L "$removal_target_directory" ]; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        print_error "refusing occupied Cargo cleanup handoff path: ${removal_target_directory}"
        return 1
    fi
    if ! mv -- "$DISPOSABLE_TARGET_DIRECTORY" "$removal_target_directory"; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        print_error "failed to hand off owned Cargo target for cleanup: ${DISPOSABLE_TARGET_DIRECTORY}"
        return 1
    fi
    DISPOSABLE_TARGET_DIRECTORY="$removal_target_directory"
    DISPOSABLE_TARGET_MARKER="${DISPOSABLE_TARGET_DIRECTORY}/${LIFECYCLE_MARKER_NAME}"
    if ! validate_owned_target "$DISPOSABLE_TARGET_DIRECTORY" "$LANE_NAME"; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-refused-after-handoff target_directory=${DISPOSABLE_TARGET_DIRECTORY}" >&2
        return 1
    fi

    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-start target_directory=${DISPOSABLE_TARGET_DIRECTORY}"
    if ! allocated_bytes "$DISPOSABLE_TARGET_DIRECTORY"; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        return 1
    fi
    target_allocated_bytes="$MEASURED_ALLOCATED_BYTES"
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=removing allocated_bytes=${target_allocated_bytes}"
    rm -rf -- "$DISPOSABLE_TARGET_DIRECTORY" &
    CLEANUP_WORKER_PROCESS_ID="$!"
    CLEANUP_OPERATION="removing"
    report_operation_progress "$CLEANUP_WORKER_PROCESS_ID" "$CLEANUP_OPERATION" &
    CLEANUP_PROGRESS_PROCESS_ID="$!"
    removal_exit_status=0
    wait "$CLEANUP_WORKER_PROCESS_ID" || removal_exit_status=$?
    kill "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
    wait "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
    CLEANUP_WORKER_PROCESS_ID=""
    CLEANUP_PROGRESS_PROCESS_ID=""
    CLEANUP_OPERATION=""
    if [ "$removal_exit_status" -ne 0 ]; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        print_error "failed to remove owned Cargo target: ${DISPOSABLE_TARGET_DIRECTORY}"
        return 1
    fi
    if [ -e "$DISPOSABLE_TARGET_DIRECTORY" ] || [ -L "$DISPOSABLE_TARGET_DIRECTORY" ]; then
        IS_CLEANUP_PENDING="false"
        IS_CLEANUP_ACTIVE="false"
        print_error "owned Cargo target remains after removal: ${DISPOSABLE_TARGET_DIRECTORY}"
        return 1
    fi
    IS_CLEANUP_PENDING="false"
    IS_CLEANUP_ACTIVE="false"
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=removed allocated_bytes=${target_allocated_bytes} elapsed_seconds=$(( $(date +%s) - cleanup_started_at_seconds ))"
}

# Signal traps reach this state owner through handle_signal.
# shellcheck disable=SC2329
finish_interrupted_cleanup() {
    signal_name="$1"
    signal_exit_status="$2"
    trap - EXIT HUP INT TERM
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-interrupted signal=${signal_name} operation=${CLEANUP_OPERATION:-handoff}" >&2

    if [ "$CLEANUP_OPERATION" = "removing" ] && [ -n "$CLEANUP_WORKER_PROCESS_ID" ]; then
        removal_exit_status=0
        wait "$CLEANUP_WORKER_PROCESS_ID" 2>/dev/null || removal_exit_status=$?
        if [ -n "$CLEANUP_PROGRESS_PROCESS_ID" ]; then
            kill "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
            wait "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
        fi
        if [ "$removal_exit_status" -eq 0 ] \
            && [ ! -e "$DISPOSABLE_TARGET_DIRECTORY" ] \
            && [ ! -L "$DISPOSABLE_TARGET_DIRECTORY" ]; then
            printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=removed-during-interruption"
        else
            printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-incomplete-retained target_directory=${DISPOSABLE_TARGET_DIRECTORY}" >&2
        fi
    else
        if [ -n "$CLEANUP_WORKER_PROCESS_ID" ]; then
            kill "$CLEANUP_WORKER_PROCESS_ID" 2>/dev/null || true
            wait "$CLEANUP_WORKER_PROCESS_ID" 2>/dev/null || true
        fi
        if [ -n "$CLEANUP_PROGRESS_PROCESS_ID" ]; then
            kill "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
            wait "$CLEANUP_PROGRESS_PROCESS_ID" 2>/dev/null || true
        fi
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-skipped-retained target_directory=${DISPOSABLE_TARGET_DIRECTORY}" >&2
    fi
    IS_CLEANUP_PENDING="false"
    IS_CLEANUP_ACTIVE="false"
    exit "$signal_exit_status"
}

stop_child_process_group() {
    stop_reason="$1"
    [ -n "$CHILD_PROCESS_ID" ] || return 0
    if kill -0 -- "-${CHILD_PROCESS_ID}" 2>/dev/null; then
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=stopping-process-group reason=${stop_reason} grace_seconds=${CHILD_TERMINATION_GRACE_SECONDS}" >&2
        kill -TERM -- "-${CHILD_PROCESS_ID}" 2>/dev/null || true
        termination_seconds_remaining="$CHILD_TERMINATION_GRACE_SECONDS"
        while kill -0 -- "-${CHILD_PROCESS_ID}" 2>/dev/null \
            && [ "$termination_seconds_remaining" -gt 0 ]; do
            sleep 1
            termination_seconds_remaining=$((termination_seconds_remaining - 1))
        done
        if kill -0 -- "-${CHILD_PROCESS_ID}" 2>/dev/null; then
            printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=kill-escalation grace_seconds=${CHILD_TERMINATION_GRACE_SECONDS}" >&2
            kill -KILL -- "-${CHILD_PROCESS_ID}" 2>/dev/null || true
            kill_wait_seconds_remaining="$KILL_CONFIRMATION_SECONDS"
            while kill -0 -- "-${CHILD_PROCESS_ID}" 2>/dev/null \
                && [ "$kill_wait_seconds_remaining" -gt 0 ]; do
                sleep 1
                kill_wait_seconds_remaining=$((kill_wait_seconds_remaining - 1))
            done
        fi
    fi
    wait "$CHILD_PROCESS_ID" 2>/dev/null || true
    if kill -0 -- "-${CHILD_PROCESS_ID}" 2>/dev/null; then
        print_error "child process group remained active after termination: ${CHILD_PROCESS_ID}"
        return 1
    fi
}

# Signal traps dispatch this owner indirectly by name.
# shellcheck disable=SC2329
handle_signal() {
    signal_name="$1"
    signal_exit_status="$2"
    if [ "$IS_CLEANUP_ACTIVE" = "true" ]; then
        finish_interrupted_cleanup "$signal_name" "$signal_exit_status"
    fi
    trap - EXIT HUP INT TERM
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=interrupted signal=${signal_name}" >&2
    if ! stop_child_process_group "signal-${signal_name}"; then
        IS_CLEANUP_PENDING="false"
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-skipped-active-process-group target_directory=${DISPOSABLE_TARGET_DIRECTORY}" >&2
        exit "$signal_exit_status"
    fi
    cleanup_owned_target || true
    exit "$signal_exit_status"
}

run_nested_command() {
    repository_root="$1"
    shift
    existing_target_directory="${CARGO_TARGET_DIR:-}"
    existing_lane_name="${ASTRONOMICAL_CARGO_TARGET_LANE:-}"
    [ -n "$existing_target_directory" ] && [ -n "$existing_lane_name" ] || {
        print_error "nested disposable lifecycle is missing its target ownership"
        exit 1
    }
    validate_owned_target "$existing_target_directory" "$existing_lane_name"
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=reused owner_lane=${existing_lane_name} target_directory=${existing_target_directory}"
    CDPATH='' cd -- "$repository_root"
    exec "$@"
}

run_disposable_command() {
    repository_root="$1"
    lane_root="$2"
    shift 2

    lane_root="${lane_root%/}"
    case "$lane_root" in
        ''|/|.|..|[!/]*)
            print_error "Cargo lane root must be an absolute non-root directory: ${lane_root}"
            exit 1
            ;;
    esac
    [ -d "$lane_root" ] && [ ! -L "$lane_root" ] && [ -w "$lane_root" ] || {
        print_error "Cargo lane root must be an existing writable directory: ${lane_root}"
        exit 1
    }
    SELECTED_RUSTC_WRAPPER="${RUSTC_WRAPPER:-}"
    if [ -n "$SELECTED_RUSTC_WRAPPER" ]; then
        require_command "$SELECTED_RUSTC_WRAPPER"
    elif command -v sccache >/dev/null 2>&1; then
        SELECTED_RUSTC_WRAPPER="sccache"
    fi

    DISPOSABLE_TARGET_DIRECTORY="$(mktemp -d "${lane_root}/astronomical-cargo-${LANE_NAME}.XXXXXX")"
    DISPOSABLE_TARGET_MARKER="${DISPOSABLE_TARGET_DIRECTORY}/${LIFECYCLE_MARKER_NAME}"
    printf '%s\n' "${LIFECYCLE_MARKER_VERSION}:${LANE_NAME}" > "$DISPOSABLE_TARGET_MARKER"
    IS_CLEANUP_PENDING="true"

    export CARGO_TARGET_DIR="$DISPOSABLE_TARGET_DIRECTORY"
    export ASTRONOMICAL_CARGO_TARGET_LIFECYCLE="disposable"
    export ASTRONOMICAL_CARGO_TARGET_LANE="$LANE_NAME"
    if [ -n "$SELECTED_RUSTC_WRAPPER" ]; then
        RUSTC_WRAPPER="$SELECTED_RUSTC_WRAPPER"
        export RUSTC_WRAPPER
    else
        unset RUSTC_WRAPPER
    fi

    trap 'cleanup_owned_target || true' EXIT
    trap 'handle_signal HUP 129' HUP
    trap 'handle_signal INT 130' INT
    trap 'handle_signal TERM 143' TERM

    journey_started_at_seconds="$(date +%s)"
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=start target_directory=${DISPOSABLE_TARGET_DIRECTORY} rustc_wrapper=${SELECTED_RUSTC_WRAPPER:-none} started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    # A dedicated process group lets interruption stop nested Cargo and native build children before their target disappears.
    set -m
    (
        CDPATH='' cd -- "$repository_root"
        exec "$@"
    ) &
    CHILD_PROCESS_ID="$!"
    set +m
    child_exit_status=0
    wait "$CHILD_PROCESS_ID" || child_exit_status=$?
    process_group_exit_status=0
    stop_child_process_group "command-finished" || process_group_exit_status=$?
    CHILD_PROCESS_ID=""
    printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=command-finished exit_status=${child_exit_status} elapsed_seconds=$(( $(date +%s) - journey_started_at_seconds ))"

    if [ "$process_group_exit_status" -ne 0 ]; then
        IS_CLEANUP_PENDING="false"
        trap - EXIT HUP INT TERM
        printf '%s\n' "[cargo-artifact-lifecycle] lane=${LANE_NAME} status=cleanup-skipped-active-process-group target_directory=${DISPOSABLE_TARGET_DIRECTORY}" >&2
        exit "$process_group_exit_status"
    fi

    cleanup_exit_status=0
    cleanup_owned_target || cleanup_exit_status=$?
    trap - EXIT HUP INT TERM
    if [ "$child_exit_status" -ne 0 ]; then
        exit "$child_exit_status"
    fi
    if [ "$cleanup_exit_status" -ne 0 ]; then
        exit "$cleanup_exit_status"
    fi
    exit 0
}

main() {
    parse_arguments "$@"

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    shift 3
    require_command cat
    require_command date
    require_command du
    require_command kill
    require_command mktemp
    require_command mv
    require_command rm
    require_command sleep

    if [ "${ASTRONOMICAL_CARGO_TARGET_LIFECYCLE:-}" = "disposable" ]; then
        run_nested_command "$repository_root" "$@"
    fi

    selected_lane_root="${ASTRONOMICAL_CARGO_LANE_ROOT:-${TMPDIR:-/tmp}}"
    [ -d "$selected_lane_root" ] || {
        print_error "Cargo lane root must be an existing directory: ${selected_lane_root}"
        exit 1
    }
    selected_lane_root="$(CDPATH='' cd -- "$selected_lane_root" && pwd -P)"
    run_disposable_command "$repository_root" "$selected_lane_root" "$@"
}

main "$@"
