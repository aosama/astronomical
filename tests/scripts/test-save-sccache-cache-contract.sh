#!/usr/bin/env sh

# Proves scripts/ci/save-sccache-cache.sh saves a new generation when no stored
# primary-key entry exists, skips the save when the stored entry is within the
# entry ceiling, deletes and replaces a stored oversize entry, skips the save
# when the trimmed directory exceeds the directory budget, always removes
# oversize stored entries, and reports the decision through GITHUB_OUTPUT,
# using a gh shim and a fabricated cache directory so the contract never
# touches real GitHub state. A watchdog bounds the whole contract below the
# 120-second test ceiling.

set -eu

TEST_WATCHDOG_SECONDS=110
SANDBOX_DIRECTORY=""
WATCHDOG_PID=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${WATCHDOG_PID:-}" ]; then
        kill -9 "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe sccache-save-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

require_command() {
    command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${command_name}"
        exit 2
    }
}

repository_root="$(cd "$(dirname "$0")/../.." && pwd)"

require_command jq
require_command shellcheck

shellcheck "${repository_root}/scripts/ci/save-sccache-cache.sh"

SANDBOX_DIRECTORY="$(mktemp -d)"
FIXTURE_BIN_DIRECTORY="${SANDBOX_DIRECTORY}/bin"
FIXTURE_CACHES_JSON="${SANDBOX_DIRECTORY}/caches.json"
FIXTURE_REPO_JSON="${SANDBOX_DIRECTORY}/repo.json"
DELETIONS_FILE="${SANDBOX_DIRECTORY}/deletions.txt"
SCRIPT_OUTPUT_FILE="${SANDBOX_DIRECTORY}/script-output.txt"
GITHUB_OUTPUT_FILE="${SANDBOX_DIRECTORY}/github-output.txt"
CACHE_DIRECTORY="${SANDBOX_DIRECTORY}/sccache"
mkdir -p "$FIXTURE_BIN_DIRECTORY" "$CACHE_DIRECTORY"

cat >"$FIXTURE_REPO_JSON" <<'JSON'
{"default_branch": "main"}
JSON

cat >"${FIXTURE_BIN_DIRECTORY}/gh" <<'SH'
#!/bin/sh
# Test double for gh: serves fixture JSON and records cache deletions.
case "$1" in
    api)
        if [ "${GH_SHIM_FAIL_LIST:-0}" = "1" ]; then
            echo "gh: simulated cache listing failure" >&2
            exit 1
        fi
        if [ "$2" = "--paginate" ]; then
            # gh api --paginate <path> --jq <filter>
            exec jq -r "$5" "$GH_SHIM_CACHES_JSON"
        fi
        # gh api <path> --jq <filter>
        exec jq -r "$4" "$GH_SHIM_CACHES_JSON"
        ;;
    cache)
        # gh cache delete <cache-id> -R <repository>
        case ",${GH_SHIM_FAIL_DELETIONS:-}," in
            *,"$3",*)
                echo "gh: simulated delete failure for cache $3" >&2
                exit 1
                ;;
        esac
        printf '%s\n' "$3" >>"$GH_SHIM_DELETIONS_FILE"
        ;;
    *)
        echo "gh shim: unsupported invocation: $*" >&2
        exit 64
        ;;
esac
SH
chmod +x "${FIXTURE_BIN_DIRECTORY}/gh"

# The save script must terminate well inside the 120-second test ceiling even
# when a downstream command hangs; the watchdog kills the whole contract.
( sleep "$TEST_WATCHDOG_SECONDS" && kill -9 "$$" ) 2>/dev/null &
WATCHDOG_PID=$!

write_caches_fixture() {
    fixture_json="$1"
    printf '%s\n' "$fixture_json" >"$FIXTURE_CACHES_JSON"
}

write_directory_fixture() {
    first_file_bytes="$1"
    second_file_bytes="$2"
    rm -rf "$CACHE_DIRECTORY"
    mkdir -p "$CACHE_DIRECTORY"
    head -c "$first_file_bytes" /dev/zero >"${CACHE_DIRECTORY}/compiled-objects-a"
    head -c "$second_file_bytes" /dev/zero >"${CACHE_DIRECTORY}/compiled-objects-b"
}

run_save_script() {
    : >"$DELETIONS_FILE"
    : >"$GITHUB_OUTPUT_FILE"
    PATH="${FIXTURE_BIN_DIRECTORY}:${PATH}" \
    GITHUB_REPOSITORY="fixture/repo" \
    GH_SHIM_CACHES_JSON="$FIXTURE_CACHES_JSON" \
    GH_SHIM_DELETIONS_FILE="$DELETIONS_FILE" \
    GH_SHIM_FAIL_DELETIONS="${GH_SHIM_FAIL_DELETIONS:-}" \
    GH_SHIM_FAIL_LIST="${GH_SHIM_FAIL_LIST:-}" \
    GITHUB_OUTPUT="$GITHUB_OUTPUT_FILE" \
    SCCACHE_DIR="$CACHE_DIRECTORY" \
    SCCACHE_SAVE_MAX_DIRECTORY_BYTES="${SCCACHE_SAVE_MAX_DIRECTORY_BYTES:-1000}" \
    SCCACHE_SAVE_MAX_ENTRY_BYTES="${SCCACHE_SAVE_MAX_ENTRY_BYTES:-2000}" \
        sh "${repository_root}/scripts/ci/save-sccache-cache.sh" \
        "astronomical-v2-sccache-macOS-ARM64-toolchainA-lockA" \
        >"$SCRIPT_OUTPUT_FILE" 2>&1
}

assert_outputs() {
    expected_should_save="$1"
    expected_decision="$2"
    case_description="$3"
    actual_should_save="$(sed -n 's/^should_save=//p' "$GITHUB_OUTPUT_FILE")"
    actual_decision="$(sed -n 's/^decision=//p' "$GITHUB_OUTPUT_FILE")"
    if [ "$actual_should_save" != "$expected_should_save" ] ||
        [ "$actual_decision" != "$expected_decision" ]; then
        print_error "$case_description"
        print_error "expected should_save=$expected_should_save decision=$expected_decision"
        print_error "actual   should_save=$actual_should_save decision=$actual_decision"
        cat "$SCRIPT_OUTPUT_FILE" >&2
        exit 1
    fi
}

assert_directory_bytes_output() {
    expected_directory_bytes="$1"
    case_description="$2"
    actual_directory_bytes="$(sed -n 's/^directory_bytes=//p' "$GITHUB_OUTPUT_FILE")"
    if [ "$actual_directory_bytes" != "$expected_directory_bytes" ]; then
        print_error "$case_description"
        print_error "expected directory_bytes=$expected_directory_bytes"
        print_error "actual   directory_bytes=$actual_directory_bytes"
        cat "$SCRIPT_OUTPUT_FILE" >&2
        exit 1
    fi
}

assert_deletions_match() {
    expected_deletions="$1"
    case_description="$2"
    actual_deletions="$(sort -n "$DELETIONS_FILE" | tr '\n' ' ' | sed 's/ $//')"
    if [ "$actual_deletions" != "$expected_deletions" ]; then
        print_error "$case_description"
        print_error "expected deletions: [$expected_deletions]"
        print_error "actual deletions:   [$actual_deletions]"
        cat "$SCRIPT_OUTPUT_FILE" >&2
        exit 1
    fi
}

PRIMARY_KEY_ENTRY_WITHIN_CEILING='{"actions_caches": [
  {"id": 401, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainA-lockA", "ref": "refs/heads/main", "created_at": "2026-10-05T04:57:40Z", "size_in_bytes": 1500}
]}'
PRIMARY_KEY_ENTRY_OVER_CEILING='{"actions_caches": [
  {"id": 402, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainA-lockA", "ref": "refs/heads/main", "created_at": "2026-10-05T04:57:40Z", "size_in_bytes": 2500}
]}'
NO_PRIMARY_KEY_ENTRY='{"actions_caches": [
  {"id": 403, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainB-lockB", "ref": "refs/heads/main", "created_at": "2026-10-05T04:57:40Z", "size_in_bytes": 1500}
]}'

# Contract 1: a fresh dependency graph with a within-budget directory saves a
# new generation, reports the measured directory bytes, and deletes nothing.
write_directory_fixture 400 400
write_caches_fixture "$NO_PRIMARY_KEY_ENTRY"
run_save_script
assert_outputs "true" "new-generation" "a fresh generation must be saved when no primary-key entry exists"
assert_directory_bytes_output "800" "the save decision must report the measured apparent-size bytes"
assert_deletions_match "" "a fresh generation must not delete any cache"

# Contract 2: a within-ceiling stored primary-key entry suppresses the save so
# every default-branch run does not re-upload an unchanged generation.
write_directory_fixture 400 400
write_caches_fixture "$PRIMARY_KEY_ENTRY_WITHIN_CEILING"
run_save_script
assert_outputs "false" "fresh-entry-within-budget" "a within-ceiling stored entry must suppress the save"
assert_deletions_match "" "a within-ceiling stored entry must not be deleted"

# Contract 3: an oversize stored primary-key entry is deleted and replaced so
# the pre-cap generation cannot be restored forever.
write_directory_fixture 400 400
write_caches_fixture "$PRIMARY_KEY_ENTRY_OVER_CEILING"
run_save_script
assert_outputs "true" "replace-oversize-entry" "an oversize stored entry must be replaced by a fresh save"
assert_deletions_match "402" "the oversize stored entry must be deleted before the replacement save"

# Contract 4: a directory over budget skips the save but still removes the
# oversize stored entry so the prune budget shrinks either way.
write_directory_fixture 700 700
write_caches_fixture "$PRIMARY_KEY_ENTRY_OVER_CEILING"
run_save_script
assert_outputs "false" "directory-over-budget" "a directory over budget must skip the save"
assert_deletions_match "402" "the oversize stored entry must be deleted even when the save is skipped"

# Contract 5: a blocked deletion of the oversize entry must not fall through
# to a colliding save.
write_directory_fixture 400 400
write_caches_fixture "$PRIMARY_KEY_ENTRY_OVER_CEILING"
GH_SHIM_FAIL_DELETIONS="402" run_save_script
unset GH_SHIM_FAIL_DELETIONS
assert_outputs "false" "replace-blocked" "a blocked oversize deletion must skip the save"
assert_deletions_match "" "a failed deletion must be tolerated and retried by the prune pass"

# Contract 6: an unavailable cache listing degrades to a skipped save instead
# of failing the job, matching the read-only fork-token tolerance.
write_directory_fixture 400 400
write_caches_fixture "$NO_PRIMARY_KEY_ENTRY"
GH_SHIM_FAIL_LIST="1" run_save_script
unset GH_SHIM_FAIL_LIST
assert_outputs "false" "cache-list-unavailable" "an unavailable cache listing must skip the save without failing"
assert_deletions_match "" "an unavailable cache listing must not attempt any deletion"

printf '%s\n' "save-sccache-cache contract tests passed"
