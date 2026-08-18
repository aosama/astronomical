#!/usr/bin/env sh

# Exercises Developer ID validation without accessing the operator's Keychain.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=30
SANDBOX_DIRECTORY=""

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in /|.|..) ;; *) rm -rf "$SANDBOX_DIRECTORY" ;; esac
    fi
}
trap cleanup 0

create_fixture_app() {
    app_bundle="$1"
    sparkle_version="${app_bundle}/Contents/Frameworks/Sparkle.framework/Versions/B"
    mkdir -p "${app_bundle}/Contents/MacOS" \
        "${sparkle_version}/Updater.app" \
        "${sparkle_version}/XPCServices/Downloader.xpc" \
        "${sparkle_version}/XPCServices/Installer.xpc"
    printf '%s\n' fixture > "${sparkle_version}/Autoupdate"
    for executable_name in astronomical-menu astronomicald astronomical-inference-worker; do
        printf '%s\n' fixture > "${app_bundle}/Contents/MacOS/${executable_name}"
    done
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-distribution-validator.XXXXXX")"
    fake_bin="${SANDBOX_DIRECTORY}/fake-bin"
    fixture_app="${SANDBOX_DIRECTORY}/Astronomical.app"
    mkdir -p "$fake_bin"
    create_fixture_app "$fixture_app"
    cat > "${fake_bin}/codesign" <<'CODESIGN'
#!/usr/bin/env sh
for command_argument in "$@"; do code_object_path="$command_argument"; done
if [ "${1:-}" = "--verify" ]; then [ "${FAKE_DEEP_REJECTED:-false}" != "true" ]; exit; fi
case "$code_object_path" in
    *"${FAKE_REJECT_OBJECT:-path-that-cannot-match}"*) printf '%s\n' 'Authority=adhoc' ;;
    *) printf '%s\n' 'Authority=Developer ID Application: Example (ABCDE12345)' ;;
esac
printf 'TeamIdentifier=%s\n' "${FAKE_TEAM_ID:-ABCDE12345}"
printf 'CodeDirectory flags=0x10000%s\n' "${FAKE_RUNTIME_MARKER-(runtime)}"
[ "${FAKE_TIMESTAMP_MISSING:-false}" = "true" ] || printf '%s\n' 'Timestamp=Aug 18, 2026 at 12:00:00 PM'
CODESIGN
    chmod +x "${fake_bin}/codesign"

    printf '%s\n' '[distribution-validator-test] case=valid-developer-id-bundle status=start'
    PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$fixture_app" --team-id ABCDE12345
    printf '%s\n' '[distribution-validator-test] case=valid-developer-id-bundle status=success'

    printf '%s\n' '[distribution-validator-test] case=wrong-team-is-rejected status=start'
    if FAKE_TEAM_ID=WRONGTEAM PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$fixture_app" --team-id ABCDE12345 >/dev/null 2>&1; then
        printf '%s\n' 'Error: validator accepted the wrong Team ID' >&2
        exit 1
    fi
    printf '%s\n' '[distribution-validator-test] case=wrong-team-is-rejected status=success'

    printf '%s\n' '[distribution-validator-test] case=missing-runtime-is-rejected status=start'
    if FAKE_RUNTIME_MARKER='' PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$fixture_app" --team-id ABCDE12345 >/dev/null 2>&1; then
        printf '%s\n' 'Error: validator accepted a bundle without Hardened Runtime' >&2
        exit 1
    fi
    printf '%s\n' '[distribution-validator-test] case=missing-runtime-is-rejected status=success'

    printf '%s\n' '[distribution-validator-test] case=missing-timestamp-is-rejected status=start'
    if FAKE_TIMESTAMP_MISSING=true PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$fixture_app" --team-id ABCDE12345 >/dev/null 2>&1; then
        printf '%s\n' 'Error: validator accepted a signature without a secure timestamp' >&2
        exit 1
    fi
    printf '%s\n' '[distribution-validator-test] case=missing-timestamp-is-rejected status=success'

    printf '%s\n' '[distribution-validator-test] case=broken-deep-seal-is-rejected status=start'
    if FAKE_DEEP_REJECTED=true PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$fixture_app" --team-id ABCDE12345 >/dev/null 2>&1; then
        printf '%s\n' 'Error: validator accepted a broken application seal' >&2
        exit 1
    fi
    printf '%s\n' '[distribution-validator-test] case=broken-deep-seal-is-rejected status=success'

    printf '%s\n' '[distribution-validator-test] case=unexpected-code-object-is-validated status=start'
    unexpected_code_object="${fixture_app}/Contents/MacOS/unexpected-helper"
    printf '%s\n' fixture > "$unexpected_code_object"
    chmod +x "$unexpected_code_object"
    if FAKE_REJECT_OBJECT=unexpected-helper PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$fixture_app" --team-id ABCDE12345 >/dev/null 2>&1; then
        printf '%s\n' 'Error: validator ignored an unexpected executable code object' >&2
        exit 1
    fi
    printf '%s\n' '[distribution-validator-test] case=unexpected-code-object-is-validated status=success'
}

main "$@"
