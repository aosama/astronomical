#!/usr/bin/env sh

# Produces the content fingerprint used to coordinate builds that would create
# the same pinned MLX and MLX-C native runtime state.

set -eu

FINGERPRINT_MANIFEST=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${FINGERPRINT_MANIFEST:-}" ] && [ -f "$FINGERPRINT_MANIFEST" ]; then
        rm -f "$FINGERPRINT_MANIFEST"
    fi
}
trap cleanup 0

main() {
    if [ "$#" -gt 1 ]; then
        print_error "Usage: scripts/native-build-cache-fingerprint.sh [repository-root]"
        exit 2
    fi
    command -v git >/dev/null 2>&1 || {
        print_error "git is required to fingerprint native build inputs"
        exit 2
    }

    if [ "$#" -eq 1 ]; then
        repository_root="$1"
    else
        repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    fi
    repository_root="$(CDPATH='' cd -- "$repository_root" && pwd -P)" || {
        print_error "repository root is unavailable: ${repository_root}"
        exit 2
    }
    git -C "$repository_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        print_error "repository root is not a Git worktree: ${repository_root}"
        exit 2
    }

    printf '%s\n' '[native-build-cache-fingerprint] status=start' >&2
    FINGERPRINT_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/astronomical-native-fingerprint.XXXXXX")"
    git -C "$repository_root" ls-files --stage -- \
        crates/runtime-integration/Cargo.toml \
        crates/runtime-integration/build.rs \
        crates/runtime-integration/build_bindings.rs \
        crates/runtime-integration/native \
        scripts/native-build-cache-fingerprint.sh \
        third-party/native-dependency-manifest.cmake \
        third-party/pins \
        third-party/patches \
        > "$FINGERPRINT_MANIFEST"
    [ -s "$FINGERPRINT_MANIFEST" ] || {
        print_error "no tracked native build inputs were found"
        exit 1
    }

    native_build_cache_fingerprint="$(git hash-object "$FINGERPRINT_MANIFEST")"
    case "$native_build_cache_fingerprint" in
        ''|*[!0-9a-f]*)
            print_error "git returned an invalid native build fingerprint"
            exit 1
            ;;
    esac
    printf '%s\n' "$native_build_cache_fingerprint"
    printf '%s\n' '[native-build-cache-fingerprint] status=success' >&2
}

main "$@"
