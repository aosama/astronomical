#!/usr/bin/env sh

# Proves packaged validation targets the operator-selected advertised model.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=30
readonly QUALIFICATION_MODEL_IDENTIFIER="fixture-qualified-text-model"
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe validator-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

create_fixture_app() {
    fixture_app_bundle="$1"
    mkdir -p \
        "${fixture_app_bundle}/Contents/Frameworks/Sparkle.framework" \
        "${fixture_app_bundle}/Contents/MacOS" \
        "${fixture_app_bundle}/Contents/Resources"
    printf '%s\n' fixture > "${fixture_app_bundle}/Contents/Info.plist"
    for packaged_resource in LICENSE THIRD_PARTY_NOTICES RUST_DEPENDENCY_NOTICES SPARKLE_LICENSE Astronomical.icns; do
        printf '%s\n' fixture > "${fixture_app_bundle}/Contents/Resources/${packaged_resource}"
    done

    cat > "${fixture_app_bundle}/Contents/MacOS/astronomicald" <<'DAEMON'
#!/usr/bin/env sh
if [ "${1:-}" = "--version" ]; then
    printf '%s\n' 'astronomicald 0.0.0-test (abc1234)'
    exit 0
fi
printf '%s\n' "$$" > "$FAKE_DAEMON_PID_FILE"
trap 'exit 0' TERM INT
while :; do sleep 1; done
DAEMON
    cat > "${fixture_app_bundle}/Contents/MacOS/astronomical-menu" <<'MENU'
#!/usr/bin/env sh
trap 'exit 0' TERM INT
while :; do sleep 1; done
MENU
    cat > "${fixture_app_bundle}/Contents/MacOS/astronomical-inference-worker" <<'WORKER'
#!/usr/bin/env sh
exit 0
WORKER
    chmod +x "${fixture_app_bundle}/Contents/MacOS/"*
}

create_fake_commands() {
    fake_command_directory="$1"
    mkdir -p "$fake_command_directory"

    cat > "${fake_command_directory}/codesign" <<'CODESIGN'
#!/usr/bin/env sh
exit 0
CODESIGN
    cat > "${fake_command_directory}/pgrep" <<'PGREP'
#!/usr/bin/env sh
exit 1
PGREP
    cat > "${fake_command_directory}/plutil" <<'PLUTIL'
#!/usr/bin/env sh
if [ "${1:-}" != "-extract" ]; then exit 1; fi
case "${2:-}" in
    AstronomicalChannel) printf '%s\n' development ;;
    CFBundleShortVersionString) printf '%s\n' 0.0.0-test ;;
    CFBundleVersion) printf '%s\n' 1 ;;
    AstronomicalBuildCommit) printf '%s\n' abc1234 ;;
    AstronomicalBuildDirty) printf '%s\n' false ;;
    AstronomicalBuildDate) printf '%s\n' 20260819 ;;
    CFBundleIdentifier) printf '%s\n' dev.astronomical.app.development ;;
    CFBundleIconFile) printf '%s\n' Astronomical.icns ;;
    AstronomicalSupervisorPort) printf '%s\n' 6733 ;;
    AstronomicalStateDirectoryName) printf '%s\n' .astronomical-dev ;;
    *) exit 1 ;;
esac
PLUTIL
    cat > "${fake_command_directory}/curl" <<'CURL'
#!/usr/bin/env sh
request_data_argument=""
request_url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --data-binary)
            request_data_argument="$2"
            shift 2
            ;;
        --header|--max-time|--request)
            shift 2
            ;;
        --silent|--show-error|--fail)
            shift
            ;;
        *)
            request_url="$1"
            shift
            ;;
    esac
done
case "$request_url" in
    http://127.0.0.1:6732/health)
        exit 22
        ;;
    */health)
        if [ ! -s "$FAKE_DAEMON_PID_FILE" ] \
            || ! kill -0 "$(cat "$FAKE_DAEMON_PID_FILE")" 2>/dev/null; then
            exit 22
        fi
        printf '%s\n' '{"status":"ok"}'
        ;;
    */v1/status)
        printf '%s\n' '{"application":{"channel":"development","version":"0.0.0-test","commit":"abc1234"}}'
        ;;
    */v1/models)
        if [ "${FAKE_INCLUDE_QUALIFICATION_MODEL:-true}" = "true" ]; then
            jq --null-input --arg model "$FAKE_EXPECTED_MODEL_IDENTIFIER" \
                '{data:[{id:"larger-model-returned-first"},{id:$model}]}'
        else
            printf '%s\n' '{"data":[{"id":"larger-model-returned-first"}]}'
        fi
        ;;
    */v1/chat/completions)
        request_file_path="${request_data_argument#@}"
        jq --exit-status --arg expected_model "$FAKE_EXPECTED_MODEL_IDENTIFIER" \
            '.model == $expected_model' "$request_file_path" >/dev/null
        jq --raw-output '.model' "$request_file_path" > "$FAKE_REQUESTED_MODEL_FILE"
        if [ "${FAKE_RESPONSE_HAS_OUTPUT:-true}" = "true" ]; then
            printf '%s\n' '{"choices":[{"message":{"content":null,"reasoning_content":"A bounded reasoning response"}}]}'
        else
            printf '%s\n' '{"choices":[{"message":{"content":null,"reasoning_content":null}}]}'
        fi
        ;;
    */v1/control/shutdown)
        if [ -s "$FAKE_DAEMON_PID_FILE" ]; then
            kill -TERM "$(cat "$FAKE_DAEMON_PID_FILE")"
        fi
        ;;
    *)
        printf '%s\n' "unexpected fake curl URL: ${request_url}" >&2
        exit 1
        ;;
esac
CURL
    chmod +x "${fake_command_directory}/"*
}

run_validator() {
    validator_output_file="$1"
    shift
    if PATH="${fake_command_directory}:${PATH}" \
        FAKE_DAEMON_PID_FILE="$fake_daemon_pid_file" \
        FAKE_REQUESTED_MODEL_FILE="$fake_requested_model_file" \
        FAKE_EXPECTED_MODEL_IDENTIFIER="$QUALIFICATION_MODEL_IDENTIFIER" \
        FAKE_RESPONSE_HAS_OUTPUT="${FAKE_RESPONSE_HAS_OUTPUT:-true}" \
        timeout "$SUBJECT_TIMEOUT_SECONDS" "$validator_script" \
        --app-bundle "$fixture_app_bundle" "$@" > "$validator_output_file" 2>&1; then
        return 0
    else
        validator_exit_code=$?
    fi
    return "$validator_exit_code"
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "test-validate-macos-app-contract.sh does not accept arguments"
        exit 2
    fi
    for required_command in timeout mktemp jq grep; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    validator_script="${repository_root}/scripts/internal/validate-macos-app.sh"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-app-validator.XXXXXX")"
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    fixture_app_bundle="${SANDBOX_DIRECTORY}/Astronomical Development.app"
    fake_daemon_pid_file="${SANDBOX_DIRECTORY}/daemon.pid"
    fake_requested_model_file="${SANDBOX_DIRECTORY}/requested-model.txt"
    create_fixture_app "$fixture_app_bundle"
    create_fake_commands "$fake_command_directory"

    printf '%s\n' '[app-validator-test] case=explicit-advertised-model status=start'
    if ! run_validator "${SANDBOX_DIRECTORY}/success.log" \
        --real-model "$QUALIFICATION_MODEL_IDENTIFIER"; then
        perl -ne 'print if $. <= 200' "${SANDBOX_DIRECTORY}/success.log" >&2
        print_error "validator rejected the advertised qualification model"
        exit 1
    fi
    [ "$(cat "$fake_requested_model_file")" = "$QUALIFICATION_MODEL_IDENTIFIER" ] || {
        print_error "validator did not request the explicit qualification model"
        exit 1
    }
    printf '%s\n' '[app-validator-test] case=explicit-advertised-model status=success'

    printf '%s\n' '[app-validator-test] case=unadvertised-model status=start'
    rm -f "$fake_requested_model_file"
    if FAKE_INCLUDE_QUALIFICATION_MODEL=false run_validator \
        "${SANDBOX_DIRECTORY}/unadvertised.log" --real-model "$QUALIFICATION_MODEL_IDENTIFIER"; then
        print_error "validator accepted an unadvertised qualification model"
        exit 1
    fi
    grep -F "qualification model is not advertised: ${QUALIFICATION_MODEL_IDENTIFIER}" \
        "${SANDBOX_DIRECTORY}/unadvertised.log" >/dev/null || {
        print_error "validator did not report the unavailable qualification model"
        exit 1
    }
    [ ! -e "$fake_requested_model_file" ] || {
        print_error "validator started generation for an unadvertised model"
        exit 1
    }
    printf '%s\n' '[app-validator-test] case=unadvertised-model status=success'

    printf '%s\n' '[app-validator-test] case=empty-assistant-output status=start'
    if FAKE_RESPONSE_HAS_OUTPUT=false run_validator \
        "${SANDBOX_DIRECTORY}/empty-output.log" --real-model "$QUALIFICATION_MODEL_IDENTIFIER"; then
        print_error "validator accepted an empty assistant response"
        exit 1
    fi
    grep -F "qualification model returned no assistant text or reasoning output: ${QUALIFICATION_MODEL_IDENTIFIER}" \
        "${SANDBOX_DIRECTORY}/empty-output.log" >/dev/null || {
        print_error "validator did not report the empty assistant response"
        exit 1
    }
    printf '%s\n' '[app-validator-test] case=empty-assistant-output status=success'

    printf '%s\n' '[app-validator-test] case=missing-model-argument status=start'
    if PATH="${fake_command_directory}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "$validator_script" --real-model > "${SANDBOX_DIRECTORY}/missing-argument.log" 2>&1; then
        print_error "validator accepted --real-model without an identifier"
        exit 1
    fi
    grep -F -- '--real-model requires a model identifier' \
        "${SANDBOX_DIRECTORY}/missing-argument.log" >/dev/null || {
        print_error "validator did not explain the missing model identifier"
        exit 1
    }
    printf '%s\n' '[app-validator-test] case=missing-model-argument status=success'
}

main "$@"
