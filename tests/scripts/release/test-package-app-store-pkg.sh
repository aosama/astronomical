#!/usr/bin/env sh

# Verifies the App Store installer packaging contract: only app-store channel
# bundles are accepted, the installer identity must exist, and the package is
# built and signature-checked through productbuild.

set -eu

SUBJECT_TIMEOUT_SECONDS=30
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe packaging test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-app-store-packager.XXXXXX")"
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    fixture_bundle="${SANDBOX_DIRECTORY}/Astronomical.app"
    mkdir -p "$fake_command_directory" \
        "${fixture_bundle}/Contents/MacOS" \
        "${fixture_bundle}/Contents/Resources"
    cp "${repository_root}/scripts/internal/package-app-store-pkg.sh" \
        "${SANDBOX_DIRECTORY}/package-app-store-pkg.sh"
    chmod +x "${SANDBOX_DIRECTORY}/package-app-store-pkg.sh"

    printf 'fixture-menu' > "${fixture_bundle}/Contents/MacOS/astronomical-menu"
    printf 'fixture-daemon' > "${fixture_bundle}/Contents/MacOS/astronomicald"
    printf 'fixture-worker' > "${fixture_bundle}/Contents/MacOS/astronomical-inference-worker"
    chmod +x "${fixture_bundle}/Contents/MacOS/"*
    {
        printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
        printf '%s\n' '<plist version="1.0"><dict>'
        printf '%s\n' '<key>AstronomicalChannel</key><string>app-store</string>'
        printf '%s\n' '<key>CFBundleShortVersionString</key><string>0.3.0</string>'
        printf '%s\n' '</dict></plist>'
    } > "${fixture_bundle}/Contents/Info.plist"

    cat > "${fake_command_directory}/plutil" <<'PLUTIL'
#!/usr/bin/env sh
if [ "${1:-}" != "-extract" ]; then exit 1; fi
# The channel answer depends on which bundle was passed, so the reject case
# can present a non-App-Store bundle to the packager.
case "${6:-}" in
    *Direct.app*)
        case "${2:-}" in
            AstronomicalChannel) printf '%s\n' stable ;;
            CFBundleShortVersionString) printf '%s\n' 0.3.0 ;;
            *) exit 1 ;;
        esac
        ;;
    *)
        case "${2:-}" in
            AstronomicalChannel) printf '%s\n' app-store ;;
            CFBundleShortVersionString) printf '%s\n' 0.3.0 ;;
            *) exit 1 ;;
        esac
        ;;
esac
PLUTIL
    cat > "${fake_command_directory}/security" <<'SECURITY'
#!/usr/bin/env sh
if [ "${1:-}" = "find-identity" ]; then
    printf '%s\n' '  1) ABCDEF1234 "3rd Party Mac Developer Installer: Example (ABCDE12345)"'
    printf '%s\n' '     1 valid identities found'
    exit 0
fi
exit 1
SECURITY
    cat > "${fake_command_directory}/xcrun" <<'XCRUN'
#!/usr/bin/env sh
if [ "${1:-}" != "productbuild" ]; then exit 1; fi
printf '%s\n' "$*" >> "${FAKE_PRODUCTBUILD_LOG:?}"
# find the trailing output package path (last argument)
output_path=""
prev=""
for argument in "$@"; do
    output_path="$argument"
done
mkdir -p "$(dirname -- "$output_path")"
printf '%s\n' 'fixture-pkg' > "$output_path"
exit 0
XCRUN
    cat > "${fake_command_directory}/pkgutil" <<'PKGUTIL'
#!/usr/bin/env sh
printf '%s\n' "Package \"fixture\": signed with 3rd Party Mac Developer Installer"
exit 0
PKGUTIL
    chmod +x "${fake_command_directory}"/*

    run_packager() {
        (CDPATH='' cd -- "$SANDBOX_DIRECTORY" && \
            FAKE_PRODUCTBUILD_LOG="${SANDBOX_DIRECTORY}/productbuild.log" \
            PATH="${fake_command_directory}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
            "${SANDBOX_DIRECTORY}/package-app-store-pkg.sh" "$@")
    }

    printf '%s\n' '[app-store-packager-test] case=rejects-non-app-store-bundle status=start'
    non_store_bundle="${SANDBOX_DIRECTORY}/Direct.app"
    mkdir -p "${non_store_bundle}/Contents"
    cp "${fixture_bundle}/Contents/Info.plist" "${non_store_bundle}/Contents/Info.plist"
    sed -i '' 's/<string>app-store<\/string>/<string>stable<\/string>/' \
        "${non_store_bundle}/Contents/Info.plist"
    if run_packager --app-bundle "$non_store_bundle" \
        --installer-identity "3rd Party Mac Developer Installer: Example (ABCDE12345)" \
        > "${SANDBOX_DIRECTORY}/reject.log" 2>&1; then
        print_error "packager accepted a non-App-Store bundle"
        exit 1
    fi
    grep -F "only App Store channel bundles can be packaged" \
        "${SANDBOX_DIRECTORY}/reject.log" >/dev/null || {
        print_error "packager did not explain the channel rejection"
        exit 1
    }
    printf '%s\n' '[app-store-packager-test] case=rejects-non-app-store-bundle status=success'

    printf '%s\n' '[app-store-packager-test] case=rejects-missing-installer-identity status=start'
    sed -i '' '/3rd Party Mac Developer Installer/d' "${fake_command_directory}/security"
    if run_packager --app-bundle "$fixture_bundle" \
        --installer-identity "3rd Party Mac Developer Installer: Example (ABCDE12345)" \
        > "${SANDBOX_DIRECTORY}/identity.log" 2>&1; then
        print_error "packager accepted a missing installer identity"
        exit 1
    fi
    grep -F "installer identity is not available as a valid code signing identity" \
        "${SANDBOX_DIRECTORY}/identity.log" >/dev/null || {
        print_error "packager did not explain the identity rejection"
        exit 1
    }
    printf '%s\n' '[app-store-packager-test] case=rejects-missing-installer-identity status=success'

    printf '%s\n' '[app-store-packager-test] case=packages-and-signs status=start'
    # restore the identity fixture for the successful case
    sed -i '' 's/     1 valid identities found/     1 valid identities found/' "${fake_command_directory}/security"
    cat > "${fake_command_directory}/security" <<'SECURITY'
#!/usr/bin/env sh
if [ "${1:-}" = "find-identity" ]; then
    printf '%s\n' '  1) ABCDEF1234 "3rd Party Mac Developer Installer: Example (ABCDE12345)"'
    printf '%s\n' '     1 valid identities found'
    exit 0
fi
exit 1
SECURITY
    run_packager --app-bundle "$fixture_bundle" \
        --installer-identity "3rd Party Mac Developer Installer: Example (ABCDE12345)" \
        --output-pkg "${SANDBOX_DIRECTORY}/out/Astronomical-app-store.pkg"
    [ -s "${SANDBOX_DIRECTORY}/out/Astronomical-app-store.pkg" ] || {
        print_error "signed installer package was not produced"
        exit 1
    }
    grep -F -- '--component' "${SANDBOX_DIRECTORY}/productbuild.log" >/dev/null || {
        print_error "productbuild was not invoked with the app component"
        exit 1
    }
    grep -F -- '--sign 3rd Party Mac Developer Installer: Example (ABCDE12345)' \
        "${SANDBOX_DIRECTORY}/productbuild.log" >/dev/null || {
        print_error "productbuild was not invoked with the installer identity"
        exit 1
    }
    printf '%s\n' '[app-store-packager-test] case=packages-and-signs status=success'
}

main "$@"
