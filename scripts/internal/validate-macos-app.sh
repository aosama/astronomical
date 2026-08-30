#!/usr/bin/env sh

# Validates exactly one channel-specific app without stopping another instance.

set -eu

readonly WAIT_TIMEOUT_SECONDS=120
readonly CLEANUP_TIMEOUT_SECONDS=10
APP_BUNDLE_PATH=""
RUN_REAL_MODEL_JOURNEY="false"
LIVE_SERVING_PROBE_MODEL_IDENTIFIER=""
BUNDLE_ONLY="false"
LAUNCHED_DAEMON_PID=""
LAUNCHED_MENU_PID=""
VALIDATION_TEMP_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_daemon_diagnostics() {
    daemon_log_file="${VALIDATION_TEMP_DIRECTORY}/daemon.log"
    if [ -f "$daemon_log_file" ]; then
        perl -ne 'print if $. <= 200' "$daemon_log_file" >&2
    fi
}

print_usage() {
    printf '%s\n' "Usage: scripts/internal/validate-macos-app.sh [--app-bundle PATH] [--bundle-only] [--real-model MODEL_ID]"
    printf '%s\n' ""
    printf '%s\n' "Default validation is isolated and does not load a model."
    printf '%s\n' "--real-model additionally runs a bounded Romeo and Juliet chat journey with the exact advertised model."
}

cleanup() {
    if [ -n "${LAUNCHED_MENU_PID:-}" ]; then
        terminate_process_bounded "$LAUNCHED_MENU_PID" "validation-menu"
    fi
    if [ -n "${LAUNCHED_DAEMON_PID:-}" ]; then
        terminate_process_bounded "$LAUNCHED_DAEMON_PID" "validation-daemon"
    fi
    if [ -n "${VALIDATION_TEMP_DIRECTORY:-}" ] && [ -d "$VALIDATION_TEMP_DIRECTORY" ]; then
        case "$VALIDATION_TEMP_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe validation directory" ;;
            *) rm -rf "$VALIDATION_TEMP_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

wait_for_process_exit() {
    process_identifier="$1"
    timeout_seconds="$2"
    waited_seconds=0
    while kill -0 "$process_identifier" 2>/dev/null; do
        if [ "$waited_seconds" -ge "$timeout_seconds" ]; then return 1; fi
        sleep 1
        waited_seconds=$((waited_seconds + 1))
        if [ $((waited_seconds % 5)) -eq 0 ]; then
            printf '%s process=%s status=stopping elapsed_seconds=%s\n' \
                "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$process_identifier" "$waited_seconds"
        fi
    done
}

terminate_process_bounded() {
    process_identifier="$1"
    process_description="$2"
    child_process_identifiers="$(pgrep -P "$process_identifier" 2>/dev/null || true)"
    kill -TERM "$process_identifier" 2>/dev/null || true
    if ! wait_for_process_exit "$process_identifier" "$CLEANUP_TIMEOUT_SECONDS"; then
        printf '%s process=%s status=force-stop description=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$process_identifier" "$process_description"
        kill -KILL "$process_identifier" 2>/dev/null || true
    fi
    wait "$process_identifier" 2>/dev/null || true
    for child_process_identifier in $child_process_identifiers; do
        if kill -0 "$child_process_identifier" 2>/dev/null; then
            kill -TERM "$child_process_identifier" 2>/dev/null || true
            if ! wait_for_process_exit "$child_process_identifier" "$CLEANUP_TIMEOUT_SECONDS"; then
                kill -KILL "$child_process_identifier" 2>/dev/null || true
            fi
        fi
    done
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        print_error "required command is unavailable: $1"
        exit 2
    fi
}

start_step() {
    current_step_name="$1"
    current_step_started_at="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name"
}

finish_step() {
    printf '%s step=%s status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name" \
        "$(( $(date +%s) - current_step_started_at ))"
}

wait_for_url() {
    expected_url="$1"
    expected_description="$2"
    waited_seconds=0
    while [ "$waited_seconds" -lt "$WAIT_TIMEOUT_SECONDS" ]; do
        if curl --silent --fail --max-time 2 "$expected_url" >/dev/null 2>&1; then
            return 0
        fi
        if [ -n "${LAUNCHED_DAEMON_PID:-}" ] && ! kill -0 "$LAUNCHED_DAEMON_PID" 2>/dev/null; then
            print_error "daemon exited while waiting for ${expected_description}"
            print_daemon_diagnostics
            return 1
        fi
        sleep 1
        waited_seconds=$((waited_seconds + 1))
        printf '%s step=%s status=waiting elapsed_seconds=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name" "$waited_seconds"
    done
    print_error "timed out waiting for ${expected_description}"
    return 1
}

read_plist_value() {
    plutil -extract "$1" raw -o - "${APP_BUNDLE_PATH}/Contents/Info.plist"
}

validate_real_model() {
    start_step "real-model-romeo-and-juliet"
    printf '%s\n' "  Shared GPU and wired memory are not isolated; this explicit journey may affect Stable latency."
    if ! jq --exit-status --arg model "$LIVE_SERVING_PROBE_MODEL_IDENTIFIER" \
        'any(.data[]; .id == $model)' "$models_file" >/dev/null; then
        print_error "live serving probe model is not advertised: ${LIVE_SERVING_PROBE_MODEL_IDENTIFIER}"
        exit 1
    fi
    model_identifier="$LIVE_SERVING_PROBE_MODEL_IDENTIFIER"
    printf '%s model=%s status=selected\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$model_identifier"
    romeo_fixture="${repository_root}/apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
    request_file="${VALIDATION_TEMP_DIRECTORY}/chat-request.json"
    jq --null-input --arg model "$model_identifier" --rawfile prompt "$romeo_fixture" \
        '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:16,stream:false}' > "$request_file"
    if ! curl --silent --show-error --fail --max-time 120 \
        --header 'Content-Type: application/json' --data-binary "@${request_file}" \
        "${supervisor_base_url}/v1/chat/completions" > "${VALIDATION_TEMP_DIRECTORY}/chat-response.json"; then
        print_error "live serving probe request failed for model: ${model_identifier}"
        print_daemon_diagnostics
        exit 1
    fi
    if ! jq --exit-status \
        '.choices[0].message | [.content, .reasoning_content] | any(type == "string" and length > 0)' \
        "${VALIDATION_TEMP_DIRECTORY}/chat-response.json" >/dev/null; then
        print_error "live serving probe model returned no assistant text or reasoning output: ${model_identifier}"
        exit 1
    fi
    finish_step
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
    # The generated Development app remains directly runnable even though its
    # parent .noindex directory keeps it out of Spotlight search results.
    APP_BUNDLE_PATH="${repository_root}/target/astronomical-macos-development.noindex/Astronomical Development.app"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --app-bundle)
                [ "$#" -ge 2 ] || { print_error "--app-bundle requires a path"; exit 2; }
                APP_BUNDLE_PATH="$2"
                shift 2
                ;;
            --real-model)
                [ "$#" -ge 2 ] || { print_error "--real-model requires a model identifier"; exit 2; }
                case "$2" in
                    ''|-*) print_error "--real-model requires a model identifier"; exit 2 ;;
                esac
                RUN_REAL_MODEL_JOURNEY="true"
                LIVE_SERVING_PROBE_MODEL_IDENTIFIER="$2"
                shift 2
                ;;
            --bundle-only)
                BUNDLE_ONLY="true"
                shift
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

    for required_command in curl jq plutil codesign perl pgrep; do require_command "$required_command"; done
    if [ "$BUNDLE_ONLY" = "true" ] && [ "$RUN_REAL_MODEL_JOURNEY" = "true" ]; then
        print_error "--bundle-only and --real-model cannot be combined"
        exit 2
    fi
    [ -d "$APP_BUNDLE_PATH" ] || { print_error "app bundle not found: ${APP_BUNDLE_PATH}"; exit 1; }
    daemon_executable="${APP_BUNDLE_PATH}/Contents/MacOS/astronomicald"
    menu_executable="${APP_BUNDLE_PATH}/Contents/MacOS/astronomical-menu"
    worker_executable="${APP_BUNDLE_PATH}/Contents/MacOS/astronomical-inference-worker"
    sparkle_framework="${APP_BUNDLE_PATH}/Contents/Frameworks/Sparkle.framework"
    for bundled_executable in "$daemon_executable" "$menu_executable" "$worker_executable"; do
        [ -x "$bundled_executable" ] || { print_error "bundled executable is unavailable: ${bundled_executable}"; exit 1; }
    done
    [ -d "$sparkle_framework" ] || { print_error "bundled Sparkle framework is unavailable: ${sparkle_framework}"; exit 1; }
    for packaged_resource in LICENSE THIRD_PARTY_NOTICES RUST_DEPENDENCY_NOTICES SPARKLE_LICENSE Astronomical.icns; do
        [ -s "${APP_BUNDLE_PATH}/Contents/Resources/${packaged_resource}" ] || {
            print_error "required bundled resource is unavailable: ${packaged_resource}"
            exit 1
        }
    done
    bundled_metallib_path="${APP_BUNDLE_PATH}/Contents/Resources/share/mlx/mlx.metallib"
    if [ -L "$bundled_metallib_path" ] || [ ! -s "$bundled_metallib_path" ]; then
        print_error "required bundled MLX AOT metallib is unavailable"
        exit 1
    fi

    start_step "bundle-identity-and-signature"
    application_channel="$(read_plist_value AstronomicalChannel)"
    application_version="$(read_plist_value CFBundleShortVersionString)"
    application_build_number="$(read_plist_value CFBundleVersion)"
    application_commit="$(read_plist_value AstronomicalBuildCommit)"
    application_is_dirty="$(read_plist_value AstronomicalBuildDirty)"
    application_build_date="$(read_plist_value AstronomicalBuildDate)"
    bundle_identifier="$(read_plist_value CFBundleIdentifier)"
    bundle_icon_file="$(read_plist_value CFBundleIconFile)"
    supervisor_port="$(read_plist_value AstronomicalSupervisorPort)"
    state_directory_name="$(read_plist_value AstronomicalStateDirectoryName)"
    case "$application_channel" in
        stable)
            expected_supervisor_port="6732"
            expected_state_directory_name=".astronomical"
            expected_bundle_identifier="dev.astronomical.app"
            [ "$application_is_dirty" = "false" ] || { print_error "Stable bundle must not be dirty"; exit 1; }
            ;;
        development)
            expected_supervisor_port="6733"
            expected_state_directory_name=".astronomical-dev"
            expected_bundle_identifier="dev.astronomical.app.development"
            ;;
        *) print_error "invalid bundle channel"; exit 1 ;;
    esac
    case "$supervisor_port" in ''|*[!0-9]*) print_error "invalid bundle supervisor port"; exit 1 ;; esac
    case "$application_build_number" in ''|*[!0-9]*) print_error "invalid bundle build number"; exit 1 ;; esac
    case "$application_build_date" in ????????) ;; *) print_error "bundle build date must use YYYYMMDD"; exit 1 ;; esac
    case "$application_build_date" in *[!0-9]*) print_error "bundle build date must use YYYYMMDD"; exit 1 ;; esac
    [ -n "$application_version" ] || { print_error "bundle version is unavailable"; exit 1; }
    [ -n "$application_commit" ] || { print_error "bundle commit is unavailable"; exit 1; }
    [ "$bundle_icon_file" = "Astronomical.icns" ] || { print_error "bundle icon identity is invalid"; exit 1; }
    case "$application_is_dirty" in true|false) ;; *) print_error "invalid bundle dirty marker"; exit 1 ;; esac
    [ "$supervisor_port" = "$expected_supervisor_port" ] || { print_error "bundle channel and supervisor port disagree"; exit 1; }
    [ "$state_directory_name" = "$expected_state_directory_name" ] || { print_error "bundle channel and state directory disagree"; exit 1; }
    [ "$bundle_identifier" = "$expected_bundle_identifier" ] || { print_error "bundle channel and identifier disagree"; exit 1; }
    codesign --verify --deep --strict "$APP_BUNDLE_PATH"
    codesign --verify --deep --strict "$sparkle_framework"
    daemon_version_output="$("$daemon_executable" --version)"
    case "$daemon_version_output" in
        *"${application_version}"*"${application_commit}"*) ;;
        *) print_error "app and bundled daemon identities do not match"; exit 1 ;;
    esac
    supervisor_base_url="http://127.0.0.1:${supervisor_port}"
    finish_step

    if [ "$BUNDLE_ONLY" = "true" ]; then
        printf '%s\n' "Validated ${application_version} ${application_channel} (${application_commit}) bundle without launching it."
        return
    fi

    if curl --silent --fail --max-time 2 "${supervisor_base_url}/health" >/dev/null 2>&1; then
        print_error "${application_channel} endpoint is already occupied; validation will not stop an unowned instance"
        exit 1
    fi
    stable_was_available="false"
    if [ "$application_channel" = "development" ] \
        && curl --silent --fail --max-time 2 "http://127.0.0.1:6732/health" >/dev/null 2>&1; then
        stable_was_available="true"
    fi

    VALIDATION_TEMP_DIRECTORY="$(mktemp -d)"
    start_step "launch-${application_channel}-daemon"
    "$daemon_executable" --instance "$application_channel" \
        > "${VALIDATION_TEMP_DIRECTORY}/daemon.log" 2>&1 &
    LAUNCHED_DAEMON_PID="$!"
    wait_for_url "${supervisor_base_url}/health" "${application_channel} health"
    status_file="${VALIDATION_TEMP_DIRECTORY}/status.json"
    curl --silent --show-error --fail --max-time 10 "${supervisor_base_url}/v1/status" > "$status_file"
    jq --exit-status \
        --arg channel "$application_channel" --arg version "$application_version" --arg commit "$application_commit" \
        '.application.channel == $channel and .application.version == $version and .application.commit == $commit' \
        "$status_file" >/dev/null
    models_file="${VALIDATION_TEMP_DIRECTORY}/models.json"
    curl --silent --show-error --fail --max-time 10 "${supervisor_base_url}/v1/models" > "$models_file"
    jq --exit-status '.data | type == "array"' "$models_file" >/dev/null
    finish_step

    start_step "launch-channel-specific-menu"
    "$menu_executable" > "${VALIDATION_TEMP_DIRECTORY}/menu.log" 2>&1 &
    LAUNCHED_MENU_PID="$!"
    sleep 2
    kill -0 "$LAUNCHED_MENU_PID" 2>/dev/null || { print_error "menu process exited during validation"; exit 1; }
    kill -TERM "$LAUNCHED_MENU_PID"
    if ! wait_for_process_exit "$LAUNCHED_MENU_PID" "$CLEANUP_TIMEOUT_SECONDS"; then
        terminate_process_bounded "$LAUNCHED_MENU_PID" "validation-menu"
    else
        wait "$LAUNCHED_MENU_PID" 2>/dev/null || true
    fi
    LAUNCHED_MENU_PID=""
    finish_step

    if [ "$RUN_REAL_MODEL_JOURNEY" = "true" ]; then validate_real_model; fi

    start_step "shutdown-${application_channel}-daemon"
    curl --silent --show-error --fail --max-time 10 --request POST \
        "${supervisor_base_url}/v1/control/shutdown" >/dev/null
    if ! wait_for_process_exit "$LAUNCHED_DAEMON_PID" "$WAIT_TIMEOUT_SECONDS"; then
        terminate_process_bounded "$LAUNCHED_DAEMON_PID" "validation-daemon"
        print_error "daemon did not complete graceful shutdown"
        exit 1
    fi
    wait "$LAUNCHED_DAEMON_PID"
    LAUNCHED_DAEMON_PID=""
    finish_step

    if [ "$stable_was_available" = "true" ]; then
        start_step "confirm-stable-remained-available"
        curl --silent --show-error --fail --max-time 10 "http://127.0.0.1:6732/health" >/dev/null
        finish_step
    fi
    printf '%s\n' "Validated ${application_version} ${application_channel} (${application_commit}) without replacing another instance."
}

main "$@"
