#!/usr/bin/env sh

# Post-build validation for Astronomical.app
#
# Terminates any stale Astronomical menu and daemon processes, launches the
# freshly built daemon, validates the backend, then launches and verifies the
# user-visible menu process:
#   (a) the daemon is running and status is healthy
#   (b) models are listed and correct
#   (c) one of the models replies with text
#   (d) the menu bar app process is running after validation
#
# Exit codes:
#   0 — all validations passed
#   1 — validation failure
#   2 — usage error
#
# Designed to be called after scripts/make-astronomical-app.sh succeeds.

set -eu

# ── Constants ──────────────────────────────────────────────────────────

SUPERVISOR_BASE_URL=""
readonly DAEMON_STARTUP_TIMEOUT_SECONDS=120
readonly RUNNING_DAEMON_IDLE_TIMEOUT_SECONDS=120
readonly CHAT_COMPLETION_TIMEOUT_SECONDS=120
readonly CHAT_MAX_TOKENS=512
readonly GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS=10
readonly STUCK_DETECTION_SECONDS=30
readonly POLL_INTERVAL_SECONDS=3

# ── State (set during main, cleaned up by trap) ─────────────────────────

LAUNCHED_DAEMON_PID=""
VALIDATION_TEMP_DIR=""
VALIDATION_MODEL_ID=""

# ── Cleanup ────────────────────────────────────────────────────────────

cleanup() {
    # Only remove a worker when cleaning up the daemon this validation launched.
    # A successful validation deliberately clears this PID to leave the app usable.
    if [ -n "${LAUNCHED_DAEMON_PID:-}" ]; then
        kill -TERM "${LAUNCHED_DAEMON_PID}" 2>/dev/null || true
        worker_pids="$(pgrep -x -f "astronomical-inference-worker" 2>/dev/null || true)"
        for worker_pid in ${worker_pids:-}; do
            kill -TERM "${worker_pid}" 2>/dev/null || true
        done
    fi

    # Remove the temporary directory.
    if [ -n "${VALIDATION_TEMP_DIR:-}" ]; then
        case "${VALIDATION_TEMP_DIR}" in
            /|.|..)
                # Refuse to remove unsafe paths.
                ;;
            *)
                rm -rf "${VALIDATION_TEMP_DIR}" 2>/dev/null || true
                ;;
        esac
    fi
}
trap cleanup 0

# ── Helpers ────────────────────────────────────────────────────────────

step_started_at_seconds=0

start_step() {
    step_name="$1"
    step_started_at_seconds="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name"
}

finish_step() {
    step_name="$1"
    step_status="$2"
    step_finished_at_seconds="$(date +%s)"
    step_elapsed_seconds=$((step_finished_at_seconds - step_started_at_seconds))
    printf '%s step=%s status=%s elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name" "$step_status" "$step_elapsed_seconds"
}

print_usage() {
    printf '%s\n' "Usage: scripts/validate-astronomical-app.sh [--app-bundle PATH]"
    printf '%s\n' ""
    printf '%s\n' "Validates a freshly built Astronomical.app by launching its daemon and"
    printf '%s\n' "running health, model-listing, and chat-completion checks."
    printf '%s\n' ""
    printf '%s\n' "  --app-bundle PATH  Path to the Astronomical.app bundle."
    printf '%s\n' "                     Defaults to target/astronomical-macos-release/Astronomical.app"
    printf '%s\n' ""
    printf '%s\n' "Requires: curl, jq, pgrep, kill"
    printf '%s\n' ""
    printf '%s\n' "The daemon must be able to load a model from ~/.astronomical/config.json"
    printf '%s\n' "or from the model_directories configured there."
}

print_error() {
    printf '%s\n' "Error: $1" >&2
}

require_command() {
    required_command="$1"
    if ! command -v "$required_command" >/dev/null 2>&1; then
        print_error "required command is unavailable: $required_command"
        exit 2
    fi
}

# ── Terminate any running Astronomical daemon ──────────────────────────

terminate_running_menu() {
    start_step "terminate-running-menu"

    menu_pids="$(pgrep -x "astronomical-menu" 2>/dev/null || true)"
    if [ -z "${menu_pids:-}" ]; then
        finish_step "terminate-running-menu" "skipped"
        return 0
    fi

    for pid in ${menu_pids:-}; do
        printf '  terminating stale menu PID=%s\n' "$pid"
        kill -TERM "$pid" 2>/dev/null || true
    done

    waited_seconds=0
    while [ "$waited_seconds" -lt "$GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS" ]; do
        remaining_menu="$(pgrep -x "astronomical-menu" 2>/dev/null || true)"
        if [ -z "${remaining_menu:-}" ]; then
            finish_step "terminate-running-menu" "success"
            return 0
        fi
        sleep 1
        waited_seconds=$((waited_seconds + 1))
    done

    remaining_menu="$(pgrep -x "astronomical-menu" 2>/dev/null || true)"
    for pid in ${remaining_menu:-}; do
        printf '  force-killing stale menu PID=%s\n' "$pid"
        kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1

    final_menu="$(pgrep -x "astronomical-menu" 2>/dev/null || true)"
    if [ -n "${final_menu:-}" ]; then
        print_error "could not terminate stale Astronomical menu process(es): ${final_menu}"
        finish_step "terminate-running-menu" "failed"
        return 1
    fi

    finish_step "terminate-running-menu" "success"
    return 0
}

terminate_running_daemon() {
    start_step "terminate-running-daemon"

    # Use -x for exact command name match to avoid matching this script itself
    # or other processes whose arguments happen to contain these strings.
    daemon_pids="$(pgrep -x "astronomicald" 2>/dev/null || true)"
    worker_pids="$(pgrep -x "astronomical-inference-worker" 2>/dev/null || true)"

    if [ -z "${daemon_pids:-}" ] && [ -z "${worker_pids:-}" ]; then
        finish_step "terminate-running-daemon" "skipped"
        return 0
    fi

    # Send SIGTERM to all matching processes.
    for pid in ${daemon_pids:-} ${worker_pids:-}; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    # Wait up to GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS for graceful shutdown.
    waited_seconds=0
    while [ "$waited_seconds" -lt "$GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS" ]; do
        remaining_daemon="$(pgrep -x "astronomicald" 2>/dev/null || true)"
        remaining_worker="$(pgrep -x "astronomical-inference-worker" 2>/dev/null || true)"
        if [ -z "${remaining_daemon:-}" ] && [ -z "${remaining_worker:-}" ]; then
            finish_step "terminate-running-daemon" "success"
            return 0
        fi
        sleep 1
        waited_seconds=$((waited_seconds + 1))
    done

    # Force kill any stragglers.
    remaining_daemon="$(pgrep -x "astronomicald" 2>/dev/null || true)"
    remaining_worker="$(pgrep -x "astronomical-inference-worker" 2>/dev/null || true)"
    for pid in ${remaining_daemon:-} ${remaining_worker:-}; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1

    # Final check.
    final_daemon="$(pgrep -x "astronomicald" 2>/dev/null || true)"
    final_worker="$(pgrep -x "astronomical-inference-worker" 2>/dev/null || true)"
    if [ -n "${final_daemon:-}" ] || [ -n "${final_worker:-}" ]; then
        print_error "could not terminate running Astronomical processes"
        finish_step "terminate-running-daemon" "failed"
        return 1
    fi

    finish_step "terminate-running-daemon" "success"
    return 0
}

# ── Validation: health endpoint ─────────────────────────────────────────

validate_health() {
    start_step "validate-health"
    health_response="$(curl --silent --max-time 5 "${SUPERVISOR_BASE_URL}/health")"
    if [ "$health_response" = "ok" ]; then
        finish_step "validate-health" "success"
        return 0
    fi
    print_error "health endpoint returned: ${health_response:-<empty>}"
    finish_step "validate-health" "failed"
    return 1
}

# ── Validation: status endpoint reports ready ────────────────────────────

validate_status_ready() {
    start_step "validate-status-ready"
    status_response="$(curl --silent --max-time 5 "${SUPERVISOR_BASE_URL}/v1/status")"
    status_value="$(printf '%s' "$status_response" | jq -r '.status // empty')"
    activity_value="$(printf '%s' "$status_response" | jq -r '.activity // empty')"

    printf '  status=%s activity=%s\n' "$status_value" "$activity_value"

    if [ "$status_value" = "ready" ] && [ "$activity_value" = "idle" ]; then
        finish_step "validate-status-ready" "success"
        return 0
    fi
    print_error "expected status=ready activity=idle, got status=${status_value} activity=${activity_value}"
    finish_step "validate-status-ready" "failed"
    return 1
}

# ── Validation: models are listed and correct ──────────────────────────

validate_models_listed() {
    start_step "validate-models-listed"
    models_response="$(curl --silent --max-time 5 "${SUPERVISOR_BASE_URL}/v1/models")"
    model_count="$(printf '%s' "$models_response" | jq '.data | length')"
    model_ids="$(printf '%s' "$models_response" | jq -r '.data[].id' 2>/dev/null || true)"

    printf '  discovered %s models:\n' "$model_count"
    if [ -n "$model_ids" ]; then
        printf '%s\n' "$model_ids" | while read -r model_id; do
            printf '    - %s\n' "$model_id"
        done
    fi

    if [ "$model_count" -eq 0 ] 2>/dev/null; then
        print_error "no models discovered — check model_directories in ~/.astronomical/config.json"
        finish_step "validate-models-listed" "failed"
        return 1
    fi

    VALIDATION_MODEL_ID="$(printf '%s' "$models_response" | jq -r '.data | map(.id) | sort | first // empty')"
    if [ -z "$VALIDATION_MODEL_ID" ]; then
        print_error "no usable model identifier was returned"
        finish_step "validate-models-listed" "failed"
        return 1
    fi

    printf '  validation model: %s\n' "$VALIDATION_MODEL_ID"

    finish_step "validate-models-listed" "success"
    return 0
}

# ── Validation: menu bar process is launched ────────────────────────────

validate_menu_launched() {
    start_step "validate-menu-launched"

    waited_seconds=0
    while [ "$waited_seconds" -lt 10 ]; do
        menu_pids="$(pgrep -x "astronomical-menu" 2>/dev/null || true)"
        if [ -n "${menu_pids:-}" ]; then
            printf '  menu PID(s): %s\n' "$menu_pids"
            finish_step "validate-menu-launched" "success"
            return 0
        fi
        sleep 1
        waited_seconds=$((waited_seconds + 1))
    done

    print_error "menu bar app did not launch an astronomical-menu process"
    finish_step "validate-menu-launched" "failed"
    return 1
}

# ── Main ────────────────────────────────────────────────────────────────

main() {
    app_bundle_path=""

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --app-bundle)
                if [ "$#" -lt 2 ]; then
                    print_error "--app-bundle requires a path argument"
                    print_usage >&2
                    exit 2
                fi
                app_bundle_path="$2"
                shift 2
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "unrecognized argument: $1"
                print_usage >&2
                exit 2
                ;;
        esac
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"

    if [ -z "$app_bundle_path" ]; then
        app_bundle_path="${repository_root}/target/astronomical-macos-release/Astronomical.app"
    fi

    # ── Prerequisite checks ─────────────────────────────────────────────

    require_command curl
    require_command jq
    require_command pgrep
    require_command kill
    require_command open

    if [ ! -d "$app_bundle_path" ]; then
        print_error "app bundle not found: ${app_bundle_path}"
        print_error "run scripts/make-astronomical-app.sh first"
        exit 1
    fi

    menu_executable="${app_bundle_path}/Contents/MacOS/astronomical-menu"
    if [ ! -x "$menu_executable" ]; then
        print_error "menu executable not found or not executable: ${menu_executable}"
        exit 1
    fi

    daemon_executable="${app_bundle_path}/Contents/MacOS/astronomicald"
    if [ ! -x "$daemon_executable" ]; then
        print_error "daemon executable not found or not executable: ${daemon_executable}"
        exit 1
    fi

    worker_executable="${app_bundle_path}/Contents/MacOS/astronomical-inference-worker"
    if [ ! -x "$worker_executable" ]; then
        print_error "worker executable not found or not executable: ${worker_executable}"
        exit 1
    fi

    app_license_path="${app_bundle_path}/Contents/Resources/LICENSE"
    if [ ! -f "$app_license_path" ]; then
        print_error "app license not found: ${app_license_path}"
        exit 1
    fi
    third_party_notices_path="${app_bundle_path}/Contents/Resources/THIRD_PARTY_NOTICES"
    if [ ! -f "$third_party_notices_path" ]; then
        print_error "third-party notices not found: ${third_party_notices_path}"
        exit 1
    fi
    rust_dependency_notices_path="${app_bundle_path}/Contents/Resources/RUST_DEPENDENCY_NOTICES"
    if [ ! -s "$rust_dependency_notices_path" ]; then
        print_error "Rust dependency notices not found or empty: ${rust_dependency_notices_path}"
        exit 1
    fi

    config_path="${HOME}/.astronomical/config.json"
    if [ ! -f "$config_path" ]; then
        print_error "Astronomical config not found: ${config_path}"
        print_error "start the daemon once to create the template, then add model_directories"
        exit 1
    fi
    supervisor_bind_address="$(jq --raw-output '.supervisor.bind_address // "127.0.0.1:6732"' "$config_path")"
    if [ -z "$supervisor_bind_address" ] || [ "$supervisor_bind_address" = "null" ]; then
        print_error "supervisor.bind_address must be a non-empty string"
        exit 1
    fi
    SUPERVISOR_BASE_URL="http://${supervisor_bind_address}"

    # Create a temporary directory for request bodies.
    VALIDATION_TEMP_DIR="$(mktemp -d)" || {
        print_error "failed to create temporary directory"
        exit 1
    }

    printf '%s\n' ""
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' "  Astronomical Post-Build Validation"
    printf '%s\n' "  App bundle:   ${app_bundle_path}"
    printf '%s\n' "  Config:       ${config_path}"
    printf '%s\n' "  API endpoint: ${SUPERVISOR_BASE_URL}"
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' ""

    # Do not interrupt a request already being served by the installed app.
    if ! wait_for_running_daemon_idle_before_replacement; then
        exit 1
    fi

    # ── Terminate any stale Astronomical menu ────────────────────────────

    if ! terminate_running_menu; then
        print_error "failed to terminate stale menu"
        exit 1
    fi

    # ── Terminate any running Astronomical daemon ────────────────────────

    if ! terminate_running_daemon; then
        print_error "failed to terminate running daemon"
        exit 1
    fi

    # ── Launch the freshly built validation daemon ───────────────────────

    if ! launch_bundled_daemon; then
        exit 1
    fi

    # ── Step 4/9: Wait for the daemon to become ready ────────────────────

    if ! wait_for_daemon_ready; then
        print_error "daemon did not become ready within ${DAEMON_STARTUP_TIMEOUT_SECONDS}s"
        # Collect diagnostic info before exiting (trap will clean up).
        diag_status="$(curl --silent --max-time 5 "${SUPERVISOR_BASE_URL}/v1/status" 2>/dev/null || printf 'unreachable')"
        printf '  diagnostic status: %s\n' "$diag_status"
        exit 1
    fi

    # ── Step 5/9: Validate health endpoint ────────────────────────────────

    if ! validate_health; then
        exit 1
    fi

    # ── Step 6/9: Validate status is ready ────────────────────────────────

    if ! validate_status_ready; then
        exit 1
    fi

    # ── Step 7/9: Validate models are listed ──────────────────────────────

    if ! validate_models_listed; then
        exit 1
    fi

    # ── Step 8/9: Validate chat completion returns text ───────────────────

    if ! validate_chat_completion; then
        exit 1
    fi

    # Validation requests must not pollute the interactive session counters.
    if ! terminate_running_daemon; then
        print_error "failed to terminate validation daemon"
        exit 1
    fi
    LAUNCHED_DAEMON_PID=""

    if ! launch_bundled_daemon; then
        exit 1
    fi
    if ! wait_for_daemon_ready; then
        print_error "clean interactive daemon did not become ready"
        exit 1
    fi
    LAUNCHED_DAEMON_PID=""

    # ── Launch and validate the user-visible menu bar app ────────────────

    start_step "launch-menu"
    printf '  launching menu bar app...\n'
    if open "$app_bundle_path" 2>/dev/null; then
        finish_step "launch-menu" "success"
    else
        print_error "failed to launch menu bar app"
        finish_step "launch-menu" "failed"
        exit 1
    fi

    if ! validate_menu_launched; then
        exit 1
    fi

    # ── Summary ──────────────────────────────────────────────────────────

    printf '%s\n' ""
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' "  All validations passed"
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' ""

    return 0
}

validation_script_directory="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)"
# shellcheck source=scripts/validate-astronomical-app-runtime.sh
. "${validation_script_directory}/validate-astronomical-app-runtime.sh"

main "$@"
