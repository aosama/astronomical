#!/usr/bin/env sh

# Exercises the complete drag-to-Applications image journey with a fictional app bundle.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=120
SANDBOX_DIRECTORY=""
MOUNT_POINT=""
IS_MOUNTED="false"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ "$IS_MOUNTED" = "true" ] && [ -d "$MOUNT_POINT" ]; then
        hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe DMG test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

main() {
    for required_command in hdiutil mktemp readlink timeout; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-dmg-test.XXXXXX")"
    fixture_app="${SANDBOX_DIRECTORY}/Astronomical.app"
    output_dmg="${SANDBOX_DIRECTORY}/Astronomical.dmg"
    MOUNT_POINT="${SANDBOX_DIRECTORY}/mounted"
    mkdir -p "${fixture_app}/Contents/MacOS" "$MOUNT_POINT"
    printf '%s\n' '<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundlePackageType</key><string>APPL</string></dict></plist>' \
        > "${fixture_app}/Contents/Info.plist"
    printf '%s\n' '#!/usr/bin/env sh' 'exit 0' > "${fixture_app}/Contents/MacOS/fixture"
    chmod +x "${fixture_app}/Contents/MacOS/fixture"

    printf '%s\n' '[dmg-test] case=drag-to-applications-journey status=start'
    timeout "$SUBJECT_TIMEOUT_SECONDS" "${repository_root}/scripts/release/create-dmg.sh" \
        --app-bundle "$fixture_app" --output "$output_dmg"
    hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$MOUNT_POINT" "$output_dmg" >/dev/null
    IS_MOUNTED="true"
    [ -d "${MOUNT_POINT}/Astronomical.app" ] || { print_error "DMG omitted Astronomical.app"; exit 1; }
    [ -L "${MOUNT_POINT}/Applications" ] || { print_error "DMG omitted Applications link"; exit 1; }
    [ "$(readlink "${MOUNT_POINT}/Applications")" = "/Applications" ] || {
        print_error "Applications link has the wrong target"
        exit 1
    }
    [ -s "${MOUNT_POINT}/.DS_Store" ] || { print_error "DMG omitted Finder layout metadata"; exit 1; }
    [ -s "${MOUNT_POINT}/.background/background.png" ] || {
        print_error "DMG omitted drag-to-Applications guidance"
        exit 1
    }
    [ -e "${MOUNT_POINT}/.metadata_never_index" ] || {
        print_error "DMG permits unnecessary Spotlight indexing"
        exit 1
    }
    [ -e "${MOUNT_POINT}/.fseventsd/no_log" ] || {
        print_error "DMG permits unnecessary file-system event logging"
        exit 1
    }
    hdiutil detach "$MOUNT_POINT" >/dev/null
    IS_MOUNTED="false"
    printf '%s\n' '[dmg-test] case=drag-to-applications-journey status=success'
}

main "$@"
