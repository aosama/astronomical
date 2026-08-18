#!/usr/bin/env sh

# Release-only entry point for building a clean Stable application candidate.

set -eu

SIGNING_IDENTITY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/release/build-stable-app.sh [--signing-identity NAME]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --signing-identity)
            [ "$#" -ge 2 ] || { print_error "--signing-identity requires a value"; exit 2; }
            [ -z "$SIGNING_IDENTITY" ] || { print_error "--signing-identity may be supplied only once"; exit 2; }
            SIGNING_IDENTITY="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            print_error "unrecognized Stable build argument: $1"
            print_usage >&2
            exit 2
            ;;
    esac
done

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
if [ -n "$SIGNING_IDENTITY" ]; then
    exec "${repository_root}/scripts/internal/build-macos-app.sh" \
        --channel stable --signing-identity "$SIGNING_IDENTITY"
fi
exec "${repository_root}/scripts/internal/build-macos-app.sh" --channel stable
