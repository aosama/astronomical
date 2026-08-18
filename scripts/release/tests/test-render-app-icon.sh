#!/usr/bin/env sh

# Proves Stable has a clean deterministic icon while Development remains visibly distinct.

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

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    icon_renderer="${repository_root}/scripts/internal/render-macos-app-icon.swift"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-app-icon.XXXXXX")"
    stable_iconset_a="${SANDBOX_DIRECTORY}/Stable-A.iconset"
    stable_iconset_b="${SANDBOX_DIRECTORY}/Stable-B.iconset"
    development_iconset_a="${SANDBOX_DIRECTORY}/Development-A.iconset"
    development_iconset_b="${SANDBOX_DIRECTORY}/Development-B.iconset"
    icns_path="${SANDBOX_DIRECTORY}/Astronomical.icns"

    printf '%s\n' '[app-icon-test] case=stable-icon-is-clean-and-deterministic status=start'
    timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "$stable_iconset_a" --channel stable
    timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "$stable_iconset_b" --channel stable
    timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "$development_iconset_a" --channel development
    timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "$development_iconset_b" --channel development
    for required_icon_name in \
        icon_16x16.png icon_16x16@2x.png \
        icon_32x32.png icon_32x32@2x.png \
        icon_128x128.png icon_128x128@2x.png \
        icon_256x256.png icon_256x256@2x.png \
        icon_512x512.png icon_512x512@2x.png
    do
        [ -s "${stable_iconset_a}/${required_icon_name}" ] || {
            print_error "rendered icon family is missing ${required_icon_name}"
            exit 1
        }
        cmp "${stable_iconset_a}/${required_icon_name}" \
            "${stable_iconset_b}/${required_icon_name}" >/dev/null || {
            print_error "Stable icon is not deterministic: ${required_icon_name}"
            exit 1
        }
        cmp "${development_iconset_a}/${required_icon_name}" \
            "${development_iconset_b}/${required_icon_name}" >/dev/null || {
            print_error "Development icon is not deterministic: ${required_icon_name}"
            exit 1
        }
        if cmp "${stable_iconset_a}/${required_icon_name}" \
            "${development_iconset_a}/${required_icon_name}" >/dev/null; then
            print_error "Development identity is absent from ${required_icon_name}"
            exit 1
        fi
    done
    timeout "$RENDER_TIMEOUT_SECONDS" iconutil --convert icns \
        --output "$icns_path" "$stable_iconset_a"
    [ -s "$icns_path" ] || {
        print_error "iconutil did not produce Astronomical.icns"
        exit 1
    }
    printf '%s\n' '[app-icon-test] case=stable-icon-is-clean-and-deterministic status=success'

    printf '%s\n' '[app-icon-test] case=invalid-channel-is-rejected status=start'
    if timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "${SANDBOX_DIRECTORY}/invalid.iconset" \
        --channel "nightly" >/dev/null 2>&1
    then
        print_error "icon renderer unexpectedly accepted an unsupported channel"
        exit 1
    fi
    printf '%s\n' '[app-icon-test] case=invalid-channel-is-rejected status=success'

    printf '%s\n' '[app-icon-test] case=duplicate-options-are-rejected status=start'
    if timeout "$RENDER_TIMEOUT_SECONDS" swift "$icon_renderer" \
        --output-directory "${SANDBOX_DIRECTORY}/duplicate.iconset" \
        --channel stable --channel development >/dev/null 2>&1
    then
        print_error "icon renderer unexpectedly accepted a duplicate channel"
        exit 1
    fi
    printf '%s\n' '[app-icon-test] case=duplicate-options-are-rejected status=success'
}

main "$@"
