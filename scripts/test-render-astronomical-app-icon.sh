#!/usr/bin/env sh

# Exercises the end-user icon journey from version/date inputs to a valid macOS icon family.

set -eu

readonly RENDER_TIMEOUT_SECONDS=120
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe icon test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

main() {
    for required_command in timeout mktemp swift iconutil; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    icon_renderer="${repository_root}/scripts/render-astronomical-app-icon.swift"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-app-icon.XXXXXX")"
    iconset_directory="${SANDBOX_DIRECTORY}/Astronomical.iconset"
    icns_path="${SANDBOX_DIRECTORY}/Astronomical.icns"

    printf '%s\n' '[app-icon-test] case=version-and-date-produce-complete-icon-family status=start'
    timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "$iconset_directory" \
        --version "0.2.0" \
        --build-date "20260814"
    for required_icon_name in \
        icon_16x16.png icon_16x16@2x.png \
        icon_32x32.png icon_32x32@2x.png \
        icon_128x128.png icon_128x128@2x.png \
        icon_256x256.png icon_256x256@2x.png \
        icon_512x512.png icon_512x512@2x.png
    do
        [ -s "${iconset_directory}/${required_icon_name}" ] || {
            print_error "rendered icon family is missing ${required_icon_name}"
            exit 1
        }
    done
    timeout "$RENDER_TIMEOUT_SECONDS" iconutil --convert icns \
        --output "$icns_path" "$iconset_directory"
    [ -s "$icns_path" ] || {
        print_error "iconutil did not produce Astronomical.icns"
        exit 1
    }
    printf '%s\n' '[app-icon-test] case=version-and-date-produce-complete-icon-family status=success'

    printf '%s\n' '[app-icon-test] case=malformed-build-date-is-rejected status=start'
    if timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "${SANDBOX_DIRECTORY}/invalid.iconset" \
        --version "0.2.0" \
        --build-date "2026-08-14" >/dev/null 2>&1
    then
        print_error "icon renderer unexpectedly accepted a non-YYYYMMDD build date"
        exit 1
    fi
    printf '%s\n' '[app-icon-test] case=malformed-build-date-is-rejected status=success'
}

main "$@"
