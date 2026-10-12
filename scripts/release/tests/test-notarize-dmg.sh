#!/usr/bin/env sh

# Proves notarization ordering and fail-closed behavior with command doubles.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=30
SANDBOX_DIRECTORY=""

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in /|.|..) ;; *) rm -rf "$SANDBOX_DIRECTORY" ;; esac
    fi
}
trap cleanup 0

write_logged_command() {
    command_path="$1"
    command_name="$2"
    cat > "$command_path" <<COMMAND
#!/usr/bin/env sh
printf '%s %s\n' '${command_name}' "\$*" >> "\${FAKE_NOTARY_LOG:?}"
exit 0
COMMAND
    chmod +x "$command_path"
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-notary-test.XXXXXX")"
    fake_bin="${SANDBOX_DIRECTORY}/fake-bin"
    fixture_dmg="${SANDBOX_DIRECTORY}/Astronomical.dmg"
    notary_log="${SANDBOX_DIRECTORY}/notary.log"
    mkdir -p "$fake_bin"
    printf '%s\n' fixture > "$fixture_dmg"
    write_logged_command "${fake_bin}/codesign" codesign
    write_logged_command "${fake_bin}/spctl" spctl
    cat > "${fake_bin}/xcrun" <<'XCRUN'
#!/usr/bin/env sh
printf '%s %s\n' xcrun "$*" >> "${FAKE_NOTARY_LOG:?}"
[ "${FAKE_NOTARY_REJECT:-false}" != "true" ]
XCRUN
    chmod +x "${fake_bin}/xcrun"

    printf '%s\n' '[notary-test] case=sign-submit-staple-assess-order status=start'
    FAKE_NOTARY_LOG="$notary_log" PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/release/notarize-dmg.sh" --dmg "$fixture_dmg" \
        --signing-identity "Developer ID Application: Example (ABCDE12345)" \
        --notary-profile "Fixture Notarization"
    expected_order="$(printf '%s\n' codesign xcrun xcrun xcrun spctl)"
    actual_order="$(while IFS= read -r log_line; do printf '%s\n' "${log_line%% *}"; done < "$notary_log")"
    [ "$actual_order" = "$expected_order" ] || {
        printf '%s\n' 'Error: notarization operations ran out of order' >&2
        exit 1
    }
    printf '%s\n' '[notary-test] case=sign-submit-staple-assess-order status=success'

    printf '%s\n' '[notary-test] case=rejection-prevents-stapling status=start'
    : > "$notary_log"
    if FAKE_NOTARY_REJECT=true FAKE_NOTARY_LOG="$notary_log" PATH="${fake_bin}:${PATH}" \
        timeout "$SUBJECT_TIMEOUT_SECONDS" "${repository_root}/scripts/release/notarize-dmg.sh" \
        --dmg "$fixture_dmg" --signing-identity "Developer ID Application: Example (ABCDE12345)" \
        --notary-profile "Fixture Notarization" >/dev/null 2>&1; then
        printf '%s\n' 'Error: rejected notarization unexpectedly succeeded' >&2
        exit 1
    fi
    [ "$(wc -l < "$notary_log" | tr -d ' ')" = "2" ] || {
        printf '%s\n' 'Error: rejected notarization continued to later operations' >&2
        exit 1
    }
    printf '%s\n' '[notary-test] case=rejection-prevents-stapling status=success'
}

main "$@"
