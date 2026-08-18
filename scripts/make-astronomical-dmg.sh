#!/usr/bin/env sh

# Builds a conventional macOS drag-to-Applications image with Finder-owned layout metadata.

set -eu

APP_BUNDLE=""
OUTPUT_DMG=""
WORK_DIRECTORY=""
MOUNT_POINT=""
IS_MOUNTED="false"
PARTIAL_OUTPUT_DMG=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ "$IS_MOUNTED" = "true" ] && [ -d "$MOUNT_POINT" ]; then
        hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
    fi
    if [ -n "${WORK_DIRECTORY:-}" ] && [ -d "$WORK_DIRECTORY" ]; then
        case "$WORK_DIRECTORY" in
            "${TMPDIR:-/tmp}"/astronomical-dmg.*) rm -rf "$WORK_DIRECTORY" ;;
            *) print_error "refusing to remove unexpected DMG work directory: ${WORK_DIRECTORY}" ;;
        esac
    fi
    if [ -n "${PARTIAL_OUTPUT_DMG:-}" ] && [ -e "$PARTIAL_OUTPUT_DMG" ]; then
        rm -f "$PARTIAL_OUTPUT_DMG"
    fi
}
trap cleanup 0

start_step() {
    current_step="$1"
    step_started_at="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step"
}

finish_step() {
    printf '%s step=%s status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step" "$(( $(date +%s) - step_started_at ))"
}

parse_arguments() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --app-bundle) [ "$#" -ge 2 ] || { print_error "--app-bundle requires a path"; exit 2; }; APP_BUNDLE="$2"; shift 2 ;;
            --output) [ "$#" -ge 2 ] || { print_error "--output requires a path"; exit 2; }; OUTPUT_DMG="$2"; shift 2 ;;
            *) print_error "unrecognized argument: $1"; exit 2 ;;
        esac
    done
    [ -d "$APP_BUNDLE" ] || { print_error "--app-bundle must name an application bundle"; exit 2; }
    [ -n "$OUTPUT_DMG" ] || { print_error "--output is required"; exit 2; }
    [ ! -e "$OUTPUT_DMG" ] || { print_error "output already exists: ${OUTPUT_DMG}"; exit 1; }
    [ ! -e "/Volumes/Astronomical" ] || {
        print_error "an Astronomical volume is already mounted; eject it before building a DMG"
        exit 1
    }
}

apply_finder_layout() {
    osascript - "$MOUNT_POINT" <<'APPLESCRIPT'
on run arguments
    set mountedVolumePath to item 1 of arguments
    set backgroundImage to POSIX file (mountedVolumePath & "/.background/background.png") as alias
    tell application "Finder"
        set mountedVolume to POSIX file mountedVolumePath as alias
        open mountedVolume
        set volumeWindow to container window of mountedVolume
        set current view of volumeWindow to icon view
        set toolbar visible of volumeWindow to false
        set statusbar visible of volumeWindow to false
        set bounds of volumeWindow to {180, 180, 820, 600}
        set iconOptions to icon view options of volumeWindow
        set arrangement of iconOptions to not arranged
        set icon size of iconOptions to 128
        set text size of iconOptions to 14
        set background picture of iconOptions to backgroundImage
        set position of item "Astronomical.app" of mountedVolume to {175, 205}
        set position of item "Applications" of mountedVolume to {455, 205}
        update mountedVolume without registering applications
        delay 1
        close volumeWindow
    end tell
end run
APPLESCRIPT
}

main() {
    parse_arguments "$@"
    for required_command in chflags ditto hdiutil mktemp mv osascript swift sync touch; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 1
        }
    done

    output_parent="$(dirname -- "$OUTPUT_DMG")"
    mkdir -p "$output_parent"
    output_name="$(basename -- "$OUTPUT_DMG")"
    PARTIAL_OUTPUT_DMG="${output_parent}/.${output_name}.partial.$$.dmg"
    WORK_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-dmg.XXXXXX")"
    volume_root="${WORK_DIRECTORY}/volume-root"
    writable_dmg="${WORK_DIRECTORY}/Astronomical-writable.dmg"
    MOUNT_POINT="${WORK_DIRECTORY}/mounted-volume"
    mkdir -p "$volume_root" "$MOUNT_POINT"

    start_step "stage-installation-volume"
    ditto "$APP_BUNDLE" "${volume_root}/Astronomical.app"
    ln -s /Applications "${volume_root}/Applications"
    mkdir -p "${volume_root}/.background"
    swift "$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)/render-astronomical-dmg-background.swift" \
        "${volume_root}/.background/background.png"
    touch "${volume_root}/.metadata_never_index"
    chflags hidden "${volume_root}/.background" "${volume_root}/.metadata_never_index"
    finish_step

    start_step "create-writable-dmg"
    hdiutil create -volname "Astronomical" -srcfolder "$volume_root" \
        -fs HFS+ -format UDRW "$writable_dmg"
    finish_step

    start_step "apply-finder-layout"
    hdiutil attach -readwrite -noverify -noautoopen -mountpoint "$MOUNT_POINT" "$writable_dmg" >/dev/null
    IS_MOUNTED="true"
    mkdir -p "${MOUNT_POINT}/.fseventsd"
    touch "${MOUNT_POINT}/.fseventsd/no_log"
    chflags hidden "${MOUNT_POINT}/.fseventsd"
    apply_finder_layout
    sync
    hdiutil detach "$MOUNT_POINT"
    IS_MOUNTED="false"
    finish_step

    start_step "compress-and-verify-dmg"
    hdiutil convert "$writable_dmg" -format UDZO -imagekey zlib-level=9 -o "$PARTIAL_OUTPUT_DMG"
    hdiutil verify "$PARTIAL_OUTPUT_DMG"
    [ -s "$PARTIAL_OUTPUT_DMG" ] || { print_error "compressed DMG was not created"; exit 1; }
    mv "$PARTIAL_OUTPUT_DMG" "$OUTPUT_DMG"
    PARTIAL_OUTPUT_DMG=""
    finish_step
}

main "$@"
