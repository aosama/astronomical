#!/usr/bin/env sh

# Exercises the notarized installation journey with command doubles and fictional paths.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=30
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in /|.|..) ;; *) rm -rf "$SANDBOX_DIRECTORY" ;; esac
    fi
}
trap cleanup 0

write_logged_success_command() {
    command_path="$1"
    command_name="$2"
    cat > "$command_path" <<COMMAND
#!/usr/bin/env sh
printf '%s %s\n' '${command_name}' "\$*" >> "\${FAKE_DMG_VALIDATION_LOG:?}"
exit 0
COMMAND
    chmod +x "$command_path"
}

main() {
    for required_command in mktemp timeout; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-dmg-validator-test.XXXXXX")"
    fake_bin="${SANDBOX_DIRECTORY}/fake-bin"
    fixture_dmg="${SANDBOX_DIRECTORY}/Astronomical.dmg"
    validation_log="${SANDBOX_DIRECTORY}/validation.log"
    mkdir -p "$fake_bin"
    printf '%s\n' fixture > "$fixture_dmg"

    write_logged_success_command "${fake_bin}/codesign" codesign
    write_logged_success_command "${fake_bin}/spctl" spctl
    write_logged_success_command "${fake_bin}/ditto" ditto
    cat > "${fake_bin}/xcrun" <<'XCRUN'
#!/usr/bin/env sh
printf '%s %s\n' xcrun "$*" >> "${FAKE_DMG_VALIDATION_LOG:?}"
[ "${FAKE_STAPLE_REJECTED:-false}" != "true" ]
XCRUN
    cat > "${fake_bin}/hdiutil" <<'HDIUTIL'
#!/usr/bin/env sh
printf '%s %s\n' hdiutil "$*" >> "${FAKE_DMG_VALIDATION_LOG:?}"
if [ "${1:-}" = "attach" ]; then
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-mountpoint" ]; then mount_point="$2"; break; fi
        shift
    done
    mkdir -p "${mount_point:?}/Astronomical.app" "${mount_point}/.background"
    ln -s /Applications "${mount_point}/Applications"
    printf '%s\n' layout > "${mount_point}/.DS_Store"
    printf '%s\n' background > "${mount_point}/.background/background.png"
fi
HDIUTIL
    chmod +x "${fake_bin}/xcrun" "${fake_bin}/hdiutil"

    printf '%s\n' '[dmg-validator-test] case=complete-installation-journey status=start'
    FAKE_DMG_VALIDATION_LOG="$validation_log" PATH="${fake_bin}:${PATH}" \
        timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/release/validate-dmg.sh" --dmg "$fixture_dmg"
    [ "$(grep -c '^codesign ' "$validation_log")" = "2" ] || {
        print_error "validator did not verify both mounted and installed app copies"
        exit 1
    }
    [ "$(grep -c '^spctl ' "$validation_log")" = "3" ] || {
        print_error "validator did not assess the DMG and both app copies"
        exit 1
    }
    printf '%s\n' '[dmg-validator-test] case=complete-installation-journey status=success'

    printf '%s\n' '[dmg-validator-test] case=rejected-ticket-prevents-mount status=start'
    : > "$validation_log"
    if FAKE_STAPLE_REJECTED=true FAKE_DMG_VALIDATION_LOG="$validation_log" \
        PATH="${fake_bin}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
        "${repository_root}/scripts/release/validate-dmg.sh" --dmg "$fixture_dmg" >/dev/null 2>&1; then
        print_error "validator accepted a rejected stapling ticket"
        exit 1
    fi
    [ "$(grep -c '^hdiutil attach ' "$validation_log" || true)" = "0" ] || {
        print_error "validator mounted a DMG after stapling validation failed"
        exit 1
    }
    printf '%s\n' '[dmg-validator-test] case=rejected-ticket-prevents-mount status=success'
}

main "$@"
