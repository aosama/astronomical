#!/usr/bin/env sh

# Signs, notarizes, staples, and validates one immutable macOS distribution image.

set -eu

readonly NOTARIZATION_TIMEOUT_SECONDS=1200
DMG_PATH=""
SIGNING_IDENTITY=""
NOTARY_PROFILE=""

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

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --dmg) [ "$#" -ge 2 ] || { print_error "--dmg requires a path"; exit 2; }; DMG_PATH="$2"; shift 2 ;;
            --signing-identity) [ "$#" -ge 2 ] || { print_error "--signing-identity requires a value"; exit 2; }; SIGNING_IDENTITY="$2"; shift 2 ;;
            --notary-profile) [ "$#" -ge 2 ] || { print_error "--notary-profile requires a value"; exit 2; }; NOTARY_PROFILE="$2"; shift 2 ;;
            *) print_error "unrecognized argument: $1"; exit 2 ;;
        esac
    done
    [ -s "$DMG_PATH" ] || { print_error "--dmg must name a nonempty disk image"; exit 2; }
    [ -n "$SIGNING_IDENTITY" ] || { print_error "--signing-identity is required"; exit 2; }
    [ -n "$NOTARY_PROFILE" ] || { print_error "--notary-profile is required"; exit 2; }
    for required_command in codesign spctl xcrun; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 1
        }
    done

    run_step "sign-dmg" codesign --force --sign "$SIGNING_IDENTITY" --timestamp "$DMG_PATH"
    run_step "submit-dmg-for-notarization" xcrun notarytool submit "$DMG_PATH" \
        --keychain-profile "$NOTARY_PROFILE" --wait --timeout "${NOTARIZATION_TIMEOUT_SECONDS}s"
    run_step "staple-notarization-ticket" xcrun stapler staple "$DMG_PATH"
    run_step "validate-stapled-ticket" xcrun stapler validate "$DMG_PATH"
    run_step "assess-notarized-dmg" spctl --assess --type open \
        --context context:primary-signature --verbose=4 "$DMG_PATH"
}

main "$@"
