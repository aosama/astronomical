#!/usr/bin/env sh

set -eu

readonly TEST_TIMEOUT_SECONDS=120

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    started_at_seconds="$(date +%s)"
    printf '%s step=macos-menu-contract-tests status=start timeout_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$TEST_TIMEOUT_SECONDS"

    # A verification gate must never touch stored credentials. The menu package's
    # Sparkle dependency is public and resolves anonymously on fresh checkouts,
    # so the credential helpers stay disabled by construction: a fresh worktree
    # no longer triggers the developer keychain item while resolving it.
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0=credential.helper
    export GIT_CONFIG_VALUE_0=

    perl -e 'alarm shift; exec @ARGV' "$TEST_TIMEOUT_SECONDS" \
        swift test --package-path "${repository_root}/apps/astronomical-menu"

    finished_at_seconds="$(date +%s)"
    elapsed_seconds=$((finished_at_seconds - started_at_seconds))
    printf '%s step=macos-menu-contract-tests status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$elapsed_seconds"
}

main "$@"
