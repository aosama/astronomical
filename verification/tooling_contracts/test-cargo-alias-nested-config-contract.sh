#!/usr/bin/env sh

# Proves that every alias in .cargo/config.toml expands identically whether
# cargo runs from the repository or from a git worktree nested under it (for
# example under .claude/worktrees/). Cargo merges config files from every
# ancestor directory: array-valued aliases are joined across files, so a
# nested worktree that sees this file twice runs `check ... check ...`, while
# string-valued aliases are replaced by the nearest file. The contract copies
# the real config into an outer and an inner sandbox directory, lists the
# aliases from both, and requires the listings to match. A detector self-check
# with a synthetic array alias proves the listing comparison still catches the
# doubling. A watchdog bounds the whole contract below the 120-second ceiling.

set -eu

TEST_WATCHDOG_SECONDS=110
WATCHDOG_PID=""
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${WATCHDOG_PID:-}" ]; then
        # Reap the watchdog after killing it, or the shell reports the kill as
        # a stray "Terminated" line after the contract's own success message.
        kill -9 "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe alias-contract sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        print_error "required command is missing: $1"
        exit 2
    }
}

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"

require_command cargo
require_command shellcheck

shellcheck "${repository_root}/verification/tooling_contracts/test-cargo-alias-nested-config-contract.sh"

( sleep "$TEST_WATCHDOG_SECONDS" && kill -9 "$$" ) 2>/dev/null &
WATCHDOG_PID=$!

# The sandbox lives under the temp directory, outside every checkout, so the
# only cargo config files in its ancestry are the two the contract plants.
SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-alias-nested-config.XXXXXX")"

# Lists `name alias: expansion` lines for every alias cargo resolves from
# `listing_directory`; cargo's config ancestry walk starts at the working
# directory, so listing from the inner directory merges both planted files.
list_aliases_from() {
    listing_directory="$1"
    (cd "$listing_directory" && cargo --list) | awk '/ alias: / { print }'
}

# Plants `planted_config_file` at an outer directory and again at a nested
# inner directory under `scenario_root`, then prints the inner listing when it
# differs from the outer one (empty output means the aliases survive nesting).
nested_listing_difference() {
    scenario_root="$1"
    planted_config_file="$2"
    mkdir -p "${scenario_root}/outer/.cargo" "${scenario_root}/outer/inner/.cargo"
    cp "$planted_config_file" "${scenario_root}/outer/.cargo/config.toml"
    cp "$planted_config_file" "${scenario_root}/outer/inner/.cargo/config.toml"
    list_aliases_from "${scenario_root}/outer" >"${scenario_root}/outer-aliases.txt"
    list_aliases_from "${scenario_root}/outer/inner" >"${scenario_root}/inner-aliases.txt"
    diff "${scenario_root}/outer-aliases.txt" "${scenario_root}/inner-aliases.txt" || true
}

# Detector self-check: an array alias must be seen to double, or the
# comparison below proves nothing.
printf '%s\n' '[cargo-alias-nested-config-test] case=detector-self-check status=start'
self_check_config="${SANDBOX_DIRECTORY}/array-alias-config.toml"
printf '%s\n' '[alias]' 'fixture-array-alias = ["check", "--workspace"]' >"$self_check_config"
self_check_difference="$(nested_listing_difference "${SANDBOX_DIRECTORY}/self-check" "$self_check_config")"
case "$self_check_difference" in
    *"check --workspace check --workspace"*) ;;
    *)
        print_error "detector self-check failed: a nested array alias did not double"
        printf '%s\n' "$self_check_difference" >&2
        exit 1
        ;;
esac
printf '%s\n' '[cargo-alias-nested-config-test] case=detector-self-check status=success'

printf '%s\n' '[cargo-alias-nested-config-test] case=repository-aliases-survive-nesting status=start'
repository_difference="$(nested_listing_difference "${SANDBOX_DIRECTORY}/repository" "${repository_root}/.cargo/config.toml")"
[ -z "$repository_difference" ] || {
    print_error "an alias in .cargo/config.toml changes meaning inside a nested worktree; write every alias as one string, not a TOML array"
    printf '%s\n' "$repository_difference" >&2
    exit 1
}
# The outer listing must contain the repository's aliases at all, or an empty
# config would pass the comparison vacuously.
grep -q '^ *verify  *alias: check --workspace --all-targets$' "${SANDBOX_DIRECTORY}/repository/outer-aliases.txt" || {
    print_error "the sandbox listing did not resolve the repository's verify alias"
    cat "${SANDBOX_DIRECTORY}/repository/outer-aliases.txt" >&2
    exit 1
}
printf '%s\n' '[cargo-alias-nested-config-test] case=repository-aliases-survive-nesting status=success'

printf '%s\n' '[cargo-alias-nested-config-test] status=success'
