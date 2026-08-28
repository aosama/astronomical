#!/usr/bin/env sh

# Mounts the final DMG and proves its installation journey and Apple trust chain.

set -eu

DMG_PATH=""
WORK_DIRECTORY=""
MOUNT_POINT=""
IS_MOUNTED="false"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

run_step() {
    step_name="$1"
    shift
    step_started_at="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name"
    "$@"
    printf '%s step=%s status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name" "$(( $(date +%s) - step_started_at ))"
}

cleanup() {
    if [ "$IS_MOUNTED" = "true" ] && [ -d "$MOUNT_POINT" ]; then
        hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
    fi
    if [ -n "${WORK_DIRECTORY:-}" ] && [ -d "$WORK_DIRECTORY" ]; then
        case "$WORK_DIRECTORY" in
            "${TMPDIR:-/tmp}"/astronomical-dmg-validation.*) rm -rf "$WORK_DIRECTORY" ;;
            *) print_error "refusing to remove unexpected validation directory: ${WORK_DIRECTORY}" ;;
        esac
    fi
}
trap cleanup 0

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --dmg) [ "$#" -ge 2 ] || { print_error "--dmg requires a path"; exit 2; }; DMG_PATH="$2"; shift 2 ;;
            *) print_error "unrecognized argument: $1"; exit 2 ;;
        esac
    done
    [ -s "$DMG_PATH" ] || { print_error "--dmg must name a nonempty disk image"; exit 2; }
    for required_command in codesign ditto hdiutil readlink spctl xcrun; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 1
        }
    done

    run_step "verify-dmg-checksum" hdiutil verify "$DMG_PATH"
    run_step "validate-stapled-ticket" xcrun stapler validate "$DMG_PATH"
    run_step "assess-notarized-dmg" spctl --assess --type open \
        --context context:primary-signature --verbose=4 "$DMG_PATH"

    WORK_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-dmg-validation.XXXXXX")"
    MOUNT_POINT="${WORK_DIRECTORY}/mounted-volume"
    mkdir -p "$MOUNT_POINT"
    run_step "mount-dmg" hdiutil attach -readonly -nobrowse -noautoopen \
        -mountpoint "$MOUNT_POINT" "$DMG_PATH"
    IS_MOUNTED="true"
    [ -d "${MOUNT_POINT}/Astronomical.app" ] || { print_error "DMG does not contain Astronomical.app"; exit 1; }
    bundled_metallib="${MOUNT_POINT}/Astronomical.app/Contents/Resources/share/mlx/mlx.metallib"
    if [ -L "$bundled_metallib" ] || [ ! -s "$bundled_metallib" ]; then
        print_error "DMG Astronomical.app is missing mlx.metallib"
        exit 1
    fi
    [ -L "${MOUNT_POINT}/Applications" ] || { print_error "DMG does not contain an Applications link"; exit 1; }
    [ "$(readlink "${MOUNT_POINT}/Applications")" = "/Applications" ] || {
        print_error "DMG Applications link does not target /Applications"
        exit 1
    }
    [ -s "${MOUNT_POINT}/.DS_Store" ] || { print_error "DMG does not contain Finder layout metadata"; exit 1; }
    [ -s "${MOUNT_POINT}/.background/background.png" ] || { print_error "DMG does not contain installation guidance"; exit 1; }
    run_step "verify-mounted-app-signature" codesign --verify --deep --strict --verbose=2 \
        "${MOUNT_POINT}/Astronomical.app"
    run_step "assess-mounted-app" spctl --assess --type execute --verbose=4 \
        "${MOUNT_POINT}/Astronomical.app"

    installed_app="${WORK_DIRECTORY}/Applications/Astronomical.app"
    mkdir -p "${WORK_DIRECTORY}/Applications"
    run_step "copy-app-to-applications" ditto "${MOUNT_POINT}/Astronomical.app" "$installed_app"
    run_step "verify-installed-app-signature" codesign --verify --deep --strict --verbose=2 "$installed_app"
    run_step "assess-installed-app" spctl --assess --type execute --verbose=4 "$installed_app"
    printf '%s\n' "Validated notarized drag-to-Applications DMG: ${DMG_PATH}"
}

main "$@"
