#!/usr/bin/env sh

# Validates Developer ID identity, Hardened Runtime, and secure timestamps before notarization.

set -eu

APP_BUNDLE=""
EXPECTED_TEAM_ID=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

signature_details() {
    codesign --display --verbose=4 "$1" 2>&1
}

validate_code_object() {
    code_object_path="$1"
    [ -e "$code_object_path" ] || { print_error "code object is unavailable: ${code_object_path}"; exit 1; }
    code_object_name="$(basename -- "$code_object_path")"
    validation_started_at="$(date +%s)"
    printf '%s operation=validate-code-signature object="%s" status=start\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$code_object_name"
    code_signature_details="$(signature_details "$code_object_path")"
    printf '%s\n' "$code_signature_details" | grep -F "Authority=Developer ID Application:" >/dev/null || {
        print_error "code object is not signed by Developer ID Application: ${code_object_path}"
        exit 1
    }
    printf '%s\n' "$code_signature_details" | grep -F "TeamIdentifier=${EXPECTED_TEAM_ID}" >/dev/null || {
        print_error "code object Team ID does not match release configuration: ${code_object_path}"
        exit 1
    }
    printf '%s\n' "$code_signature_details" | grep -E 'flags=.*\(runtime\)' >/dev/null || {
        print_error "code object does not enable Hardened Runtime: ${code_object_path}"
        exit 1
    }
    printf '%s\n' "$code_signature_details" | grep -F 'Timestamp=' >/dev/null || {
        print_error "code object does not have a secure timestamp: ${code_object_path}"
        exit 1
    }
    printf '%s operation=validate-code-signature object="%s" status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$code_object_name" \
        "$(( $(date +%s) - validation_started_at ))"
}

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --app-bundle) [ "$#" -ge 2 ] || { print_error "--app-bundle requires a path"; exit 2; }; APP_BUNDLE="$2"; shift 2 ;;
            --team-id) [ "$#" -ge 2 ] || { print_error "--team-id requires a value"; exit 2; }; EXPECTED_TEAM_ID="$2"; shift 2 ;;
            *) print_error "unrecognized argument: $1"; exit 2 ;;
        esac
    done
    [ -d "$APP_BUNDLE" ] || { print_error "--app-bundle must name an application bundle"; exit 2; }
    case "$EXPECTED_TEAM_ID" in
        ???????*) ;;
        *) print_error "--team-id is required"; exit 2 ;;
    esac
    for required_command in codesign find grep; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 1
        }
    done

    printf '%s operation=verify-app-seal status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    app_seal_started_at="$(date +%s)"
    codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"
    printf '%s operation=verify-app-seal status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(( $(date +%s) - app_seal_started_at ))"
    sparkle_version_directory="${APP_BUNDLE}/Contents/Frameworks/Sparkle.framework/Versions/B"
    for code_object_path in \
        "${sparkle_version_directory}/Autoupdate" \
        "${sparkle_version_directory}/Updater.app" \
        "${sparkle_version_directory}/XPCServices/Downloader.xpc" \
        "${sparkle_version_directory}/XPCServices/Installer.xpc" \
        "${APP_BUNDLE}/Contents/Frameworks/Sparkle.framework" \
        "${APP_BUNDLE}/Contents/MacOS/astronomical-menu" \
        "${APP_BUNDLE}/Contents/MacOS/astronomicald" \
        "${APP_BUNDLE}/Contents/MacOS/astronomical-inference-worker" \
        "$APP_BUNDLE"
    do
        validate_code_object "$code_object_path"
    done
    find "$APP_BUNDLE" \( \
        -type d \( -name '*.app' -o -name '*.xpc' -o -name '*.framework' \) \
        -o -type f -perm -111 \
    \) -print | while IFS= read -r discovered_code_object_path; do
        validate_code_object "$discovered_code_object_path"
    done
    printf '%s\n' "Validated Developer ID distribution signature for every bundled code object."
}

main "$@"
