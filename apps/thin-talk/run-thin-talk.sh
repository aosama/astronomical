#!/usr/bin/env sh
#
# Builds and launches the Thin Talk macOS app for one channel.
#
# Thin Talk is a thin client: it talks to the already-running Astronomical
# supervisor over its public REST endpoints, so the bundle carries only the
# chat executable and its metadata. It bundles no daemon, worker, Sparkle
# framework, or MLX metallib.
#
# Usage: apps/thin-talk/run-thin-talk.sh [--channel development|stable]
#
# Exit codes:
#   0 — build and launch succeeded
#   1 — build or bundle failure

set -eu

APPLICATION_CHANNEL="development"

print_usage() {
    printf '%s\n' "Usage: apps/thin-talk/run-thin-talk.sh [--channel development|stable]"
    printf '%s\n' ""
    printf '%s\n' "Builds the Thin Talk executable, assembles a lean app bundle, and"
    printf '%s\n' "launches it. Development is the safe default."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --channel)
            [ "$#" -ge 2 ] || { printf '%s\n' "Error: --channel requires a value" >&2; exit 1; }
            APPLICATION_CHANNEL="$2"
            shift 2
            continue
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            printf '%s\n' "Error: unrecognized argument: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

case "$APPLICATION_CHANNEL" in
    development)
        bundle_name="Thin Talk Development"
        bundle_identifier="dev.astronomical.thin-talk.development"
        supervisor_port="6733"
        state_directory_name=".astronomical-dev"
        icon_channel="development"
        ;;
    stable)
        bundle_name="Thin Talk"
        bundle_identifier="dev.astronomical.thin-talk"
        supervisor_port="6732"
        state_directory_name=".astronomical"
        icon_channel="stable"
        ;;
    *)
        printf '%s\n' "Error: channel must be development or stable" >&2
        exit 1
        ;;
esac

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
package_path="${repository_root}/apps/thin-talk"
release_directory="${repository_root}/target/thin-talk-${APPLICATION_CHANNEL}.noindex"
app_bundle_path="${release_directory}/${bundle_name}.app"

printf '%s operation=build-thin-talk status=start channel=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$APPLICATION_CHANNEL"

rm -rf "$app_bundle_path"
swift build --configuration release --package-path "$package_path" --product ThinTalk

mkdir -p "${app_bundle_path}/Contents/MacOS"
mkdir -p "${app_bundle_path}/Contents/Resources"

{
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
    printf '%s\n' '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    printf '%s\n' '<plist version="1.0">'
    printf '%s\n' '<dict>'
    printf '  <key>CFBundleName</key><string>%s</string>\n' "$bundle_name"
    printf '  <key>CFBundleDisplayName</key><string>%s</string>\n' "$bundle_name"
    printf '%s\n' '  <key>CFBundleExecutable</key><string>ThinTalk</string>'
    printf '  <key>CFBundleIdentifier</key><string>%s</string>\n' "$bundle_identifier"
    printf '  <key>CFBundleVersion</key><string>1</string>\n'
    printf '  <key>CFBundleShortVersionString</key><string>0.1.0</string>\n'
    printf '  <key>LSMinimumSystemVersion</key><string>14.0</string>\n'
    printf '  <key>CFBundleIconFile</key><string>Astronomical.icns</string>\n'
    printf '  <key>AstronomicalChannel</key><string>%s</string>\n' "$APPLICATION_CHANNEL"
    printf '  <key>AstronomicalSupervisorPort</key><integer>%s</integer>\n' "$supervisor_port"
    printf '  <key>AstronomicalStateDirectoryName</key><string>%s</string>\n' "$state_directory_name"
    printf '%s\n' '  <key>CFBundlePackageType</key><string>APPL</string>'
    printf '%s\n' '</dict>'
    printf '%s\n' '</plist>'
} > "${app_bundle_path}/Contents/Info.plist"

cp "${package_path}/.build/release/ThinTalk" \
    "${app_bundle_path}/Contents/MacOS/ThinTalk"
chmod +x "${app_bundle_path}/Contents/MacOS/ThinTalk"

iconset_directory="${app_bundle_path}/Contents/Resources/Astronomical.iconset"
icon_resource="${app_bundle_path}/Contents/Resources/Astronomical.icns"
swift "${repository_root}/scripts/internal/render-macos-app-icon.swift" \
    --output-directory "$iconset_directory" \
    --channel "$icon_channel"
iconutil --convert icns --output "$icon_resource" "$iconset_directory"
rm -rf "$iconset_directory"
[ -s "$icon_resource" ] || {
    printf '%s\n' "Error: generated macOS icon is unavailable" >&2
    exit 1
}

plutil -lint "${app_bundle_path}/Contents/Info.plist"
codesign --force --sign - "${app_bundle_path}"

printf '%s operation=build-thin-talk status=success bundle=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$app_bundle_path"
printf '%s\n' ""
printf '%s\n' "Thin Talk is ready."
printf '%s\n' "  Launch: open \"$app_bundle_path\""
printf '%s\n' ""

open "$app_bundle_path"
