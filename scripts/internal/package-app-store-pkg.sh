#!/usr/bin/env sh

# Packages a validated App Store channel app bundle into the signed installer
# package that App Store Connect accepts.
#
# The bundle must already be assembled and signed by
# scripts/internal/build-macos-app.sh --channel app-store. The installer
# package is signed with the third-party Mac Developer Installer certificate;
# App Store Connect rejects unsigned or Developer ID-signed packages.

set -eu

APP_BUNDLE_PATH=""
INSTALLER_IDENTITY=""
OUTPUT_PKG_PATH=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/internal/package-app-store-pkg.sh --app-bundle PATH --installer-identity NAME [--output-pkg PATH]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --app-bundle)
            [ "$#" -ge 2 ] || { print_error "--app-bundle requires a value"; exit 2; }
            [ -z "$APP_BUNDLE_PATH" ] || { print_error "--app-bundle may be supplied only once"; exit 2; }
            APP_BUNDLE_PATH="$2"
            shift 2
            ;;
        --installer-identity)
            [ "$#" -ge 2 ] || { print_error "--installer-identity requires a value"; exit 2; }
            [ -z "$INSTALLER_IDENTITY" ] || { print_error "--installer-identity may be supplied only once"; exit 2; }
            INSTALLER_IDENTITY="$2"
            shift 2
            ;;
        --output-pkg)
            [ "$#" -ge 2 ] || { print_error "--output-pkg requires a value"; exit 2; }
            [ -z "$OUTPUT_PKG_PATH" ] || { print_error "--output-pkg may be supplied only once"; exit 2; }
            OUTPUT_PKG_PATH="$2"
            shift 2
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

[ -d "$APP_BUNDLE_PATH" ] || { print_error "app bundle not found: ${APP_BUNDLE_PATH}"; exit 2; }
[ -n "$INSTALLER_IDENTITY" ] || { print_error "an installer identity is required"; exit 2; }

info_plist="${APP_BUNDLE_PATH}/Contents/Info.plist"
[ -s "$info_plist" ] || { print_error "bundle Info.plist is unavailable: ${info_plist}"; exit 1; }
bundle_channel="$(plutil -extract AstronomicalChannel raw -o - "$info_plist" 2>/dev/null)" || {
    print_error "bundle does not declare an Astronomical channel"
    exit 1
}
[ "$bundle_channel" = "app-store" ] || {
    print_error "only App Store channel bundles can be packaged for App Store Connect; this bundle declares ${bundle_channel}"
    exit 1
}
application_version="$(plutil -extract CFBundleShortVersionString raw -o - "$info_plist")"

if ! security find-identity -v | grep -F -- "\"${INSTALLER_IDENTITY}\"" >/dev/null; then
    print_error "installer identity is not available as a valid code signing identity: ${INSTALLER_IDENTITY}"
    exit 1
fi

if [ -z "$OUTPUT_PKG_PATH" ]; then
    bundle_parent_directory="$(dirname -- "$APP_BUNDLE_PATH")"
    OUTPUT_PKG_PATH="${bundle_parent_directory}/Astronomical-${application_version}-app-store.pkg"
fi
case "$OUTPUT_PKG_PATH" in
    /*) ;;
    *) OUTPUT_PKG_PATH="$(pwd -P)/${OUTPUT_PKG_PATH}" ;;
esac

printf '%s operation=package-app-store-pkg status=start identity_present=true\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')"
xcrun productbuild \
    --component "$APP_BUNDLE_PATH" /Applications \
    --sign "$INSTALLER_IDENTITY" \
    "$OUTPUT_PKG_PATH"
pkgutil --check-signature "$OUTPUT_PKG_PATH"
printf '%s operation=package-app-store-pkg status=success output=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$OUTPUT_PKG_PATH"
printf '%s\n' "Signed App Store installer package: ${OUTPUT_PKG_PATH}"
printf '%s\n' "Upload it to App Store Connect to submit for review."
