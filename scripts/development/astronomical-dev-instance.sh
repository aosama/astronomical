#!/usr/bin/env sh

# Start, stop, restart, or inspect the isolated Development instance.
#
# Development only by construction: this script hardcodes the development
# bundle, port 6733, and the ~/.astronomical-dev state directory, and it
# accepts no overrides, so it can never touch the Stable instance, port
# 6732, or the ~/.astronomical state directory.
#
# Any ASTRONOMICAL_* environment variables exported by the caller flow
# through to the daemon and its worker child (for example the
# ASTRONOMICAL_MLX_WIRED_CLAIM=1 wired-residency experiment flag).
#
# Usage:
#   scripts/development/astronomical-dev-instance.sh status
#   scripts/development/astronomical-dev-instance.sh start
#   scripts/development/astronomical-dev-instance.sh stop
#   scripts/development/astronomical-dev-instance.sh restart

set -eu

REPOSITORY_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
APP_BUNDLE_PATH="${REPOSITORY_ROOT}/target/astronomical-macos-development.noindex/Astronomical Development.app"
DAEMON_EXECUTABLE_PATH="${APP_BUNDLE_PATH}/Contents/MacOS/astronomicald"
DAEMON_BASE_URL="http://127.0.0.1:6733"
STATE_DIRECTORY="${HOME}/.astronomical-dev"
LOG_DIRECTORY="${STATE_DIRECTORY}/logs"
HEALTH_WAIT_SECONDS=30
SHUTDOWN_WAIT_SECONDS=10

print_usage() {
    printf '%s\n' "Usage: scripts/development/astronomical-dev-instance.sh status|start|stop|restart"
}

fail() {
    printf '%s\n' "Error: $1" >&2
    exit 1
}

require_command() {
    command -v "$1" > /dev/null 2>&1 || fail "required command is unavailable: $1"
}

require_command curl
require_command jq
require_command git

[ -f "$DAEMON_EXECUTABLE_PATH" ] || fail "the Development app bundle is missing at:
  $APP_BUNDLE_PATH
build it first:  scripts/build-development-app.sh"

daemon_process_ids() {
    pgrep -f "astronomicald --instance development" 2>/dev/null || true
}

daemon_is_responding() {
    curl -s --max-time 2 "${DAEMON_BASE_URL}/v1/health" > /dev/null 2>&1
}

latest_log_file() {
    ls -t "${LOG_DIRECTORY}/$1".* 2>/dev/null | head -1 || true
}

wait_for_health() {
    waited_seconds=0
    while [ "$waited_seconds" -lt "$HEALTH_WAIT_SECONDS" ]; do
        if daemon_is_responding; then
            return 0
        fi
        sleep 2
        waited_seconds=$((waited_seconds + 2))
    done
    return 1
}

fetch_status_json() {
    curl -s --max-time 3 "${DAEMON_BASE_URL}/v1/status"
}

status_field() {
    printf '%s' "$status_json" | jq -r "$1 // \"-\""
}

apple_gigabytes() {
    printf '%s' "$1" | awk '{ printf "%.2f GB", $1 / 1073741824 }'
}

stop_instance() {
    daemon_process_ids="$(daemon_process_ids)"
    if [ -z "$daemon_process_ids" ]; then
        printf '%s\n' "Development instance is not running."
        return 0
    fi
    shutdown_status_code="$(curl -s --max-time 3 -X POST \
        "${DAEMON_BASE_URL}/v1/control/shutdown" -o /dev/null -w '%{http_code}' 2>/dev/null || echo 000)"
    waited_seconds=0
    while [ "$waited_seconds" -lt "$SHUTDOWN_WAIT_SECONDS" ]; do
        [ -z "$(daemon_process_ids)" ] && break
        sleep 1
        waited_seconds=$((waited_seconds + 1))
    done
    if [ -n "$(daemon_process_ids)" ]; then
        fail "the development daemon (pid $daemon_process_ids) did not exit within ${SHUTDOWN_WAIT_SECONDS}s of the graceful shutdown request (HTTP $shutdown_status_code); inspect it manually - this script never sends signals blindly"
    fi
    printf '%s\n' "Development instance stopped (graceful shutdown, HTTP $shutdown_status_code)."
}

start_instance() {
    if [ -n "$(daemon_process_ids)" ] || daemon_is_responding; then
        fail "the development instance is already running; use restart or stop first"
    fi
    # Exported ASTRONOMICAL_* variables intentionally flow to the daemon and
    # its worker child (for example ASTRONOMICAL_MLX_WIRED_CLAIM=1).
    nohup "$DAEMON_EXECUTABLE_PATH" --instance development \
        > /tmp/astronomical-dev-daemon-stdout.log 2>&1 &
    daemon_launch_pid=$!
    disown "$daemon_launch_pid" 2>/dev/null || true
    if wait_for_health; then
        printf '%s\n' "Development instance is up: ${DAEMON_BASE_URL} (daemon pid $daemon_launch_pid)"
        printf '%s\n' "Logs: ${LOG_DIRECTORY}"
    else
        fail "the development daemon did not become healthy within ${HEALTH_WAIT_SECONDS}s; check /tmp/astronomical-dev-daemon-stdout.log and ${LOG_DIRECTORY}"
    fi
}

print_status() {
    if ! daemon_is_responding; then
        printf '%s\n' "Development instance: not responding on ${DAEMON_BASE_URL}."
        daemon_process_ids="$(daemon_process_ids)"
        if [ -n "$daemon_process_ids" ]; then
            printf '%s\n' "A daemon process exists (pid $daemon_process_ids) but is not answering - it may still be starting or wedged."
        fi
        return 0
    fi
    status_json="$(fetch_status_json)"
    printf '%s\n' "Development instance: $(status_field '.status') ($(status_field '.activity'))"
    printf '%s\n' "  model:             $(status_field '.ready_model_id')  [$(status_field '.expert_memory_mode')]"
    printf '%s\n' "  machine ceiling:   $(apple_gigabytes "$(status_field '.machine_mlx_memory_ceiling_bytes')")"
    printf '%s\n' "  effective ceiling: $(apple_gigabytes "$(status_field '.mlx_memory_ceiling_bytes')") (configured $(status_field '.configured_maximum_mlx_memory_gb'), pending $(status_field '.pending_mlx_memory_ceiling_bytes'))"
    printf '%s\n' "  model floor:       $(apple_gigabytes "$(status_field '.minimum_mlx_memory_ceiling_bytes')") | active now: $(apple_gigabytes "$(status_field '.mlx_memory_snapshot.active_memory_bytes // 0')")"
    printf '%s\n' "  session:           $(status_field '.serving_session.completed_request_count') requests · $(status_field '.serving_session.average_prefill_tok_per_second' | awk '{ printf "%.0f", $1 }') tok/s prefill · $(status_field '.serving_session.average_generation_tok_per_second' | awk '{ printf "%.0f", $1 }') tok/s decode"
    daemon_commit="$(status_field '.application.commit')"
    if [ "$daemon_commit" != "-" ]; then
        head_commit="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        case "$head_commit" in
            "$daemon_commit"*)
                printf '%s\n' "  build: ${daemon_commit} (matches worktree HEAD; dirty flag $(status_field '.application.is_dirty'))"
                ;;
            *)
                printf '%s\n' "  WARNING: the running daemon was built from commit ${daemon_commit}," | cut -c1-120
                printf '%s\n' "  but this worktree is at ${head_commit}. Rebuild with scripts/build-development-app.sh"
                printf '%s\n' "  and restart to serve current code."
                ;;
        esac
    fi
    printf '%s\n' "  logs: $(latest_log_file daemon) + $(latest_log_file worker)"
}

case "${1:-}" in
    status) print_status ;;
    stop) stop_instance ;;
    start) start_instance ;;
    restart)
        stop_instance
        start_instance
        print_status
        ;;
    *) print_usage; exit 2 ;;
esac
