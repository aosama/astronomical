#!/usr/bin/env sh

# Proves scripts/ci/prune-ci-caches.sh keeps the newest cache per family and ref
# class, keeps zero non-default-ref sccache entries because sccache saves are
# default-branch-only, deletes sccache entries above the decimal 3 GB family
# ceiling even when they are the newest, never applies the sccache ceiling to
# other families, tolerates delete failures, and deletes nothing in dry-run
# mode, using a gh shim so the contract never touches real GitHub state.

set -eu

SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe prune-test sandbox" ;;
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

SANDBOX_DIRECTORY="$(mktemp -d)"
FIXTURE_BIN_DIRECTORY="${SANDBOX_DIRECTORY}/bin"
FIXTURE_CACHES_JSON="${SANDBOX_DIRECTORY}/caches.json"
FIXTURE_OVERSIZED_CACHES_JSON="${SANDBOX_DIRECTORY}/caches-oversized.json"
FIXTURE_REPO_JSON="${SANDBOX_DIRECTORY}/repo.json"
DELETIONS_FILE="${SANDBOX_DIRECTORY}/deletions.txt"
PRUNE_OUTPUT_FILE="${SANDBOX_DIRECTORY}/prune-output.txt"
mkdir -p "$FIXTURE_BIN_DIRECTORY"

cat >"$FIXTURE_REPO_JSON" <<'JSON'
{"default_branch": "main"}
JSON

# Two sccache generations on main (one stale) and one on a pull-request ref.
# All three sit under the decimal 3 GB ceiling, so the count policy decides:
# the stale main entry is surplus and the pull-request copy is deleted because
# sccache keeps zero non-default-ref entries. One native-build trio on main
# (one stale beyond the keep-two policy) and one swiftpm cache that must
# survive untouched.
cat >"$FIXTURE_CACHES_JSON" <<'JSON'
{"actions_caches": [
  {"id": 101, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainA-lockA", "ref": "refs/heads/main", "created_at": "2026-09-29T03:23:17Z", "size_in_bytes": 2900000000},
  {"id": 102, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainA-lockB", "ref": "refs/heads/main", "created_at": "2026-09-28T22:02:06Z", "size_in_bytes": 2850000000},
  {"id": 103, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainA-lockA", "ref": "refs/pull/851/merge", "created_at": "2026-09-29T03:16:40Z", "size_in_bytes": 2800000000},
  {"id": 201, "key": "astronomical-v2-native-build-macOS-ARM64-identityNew", "ref": "refs/heads/main", "created_at": "2026-09-29T02:56:59Z", "size_in_bytes": 6087355},
  {"id": 202, "key": "astronomical-v2-native-build-macOS-ARM64-identityMid", "ref": "refs/heads/main", "created_at": "2026-09-28T20:00:00Z", "size_in_bytes": 6080000},
  {"id": 203, "key": "astronomical-v2-native-build-macOS-ARM64-identityOld", "ref": "refs/heads/main", "created_at": "2026-09-27T20:00:00Z", "size_in_bytes": 6080000},
  {"id": 301, "key": "astronomical-v2-swiftpm-macOS-ARM64-identity", "ref": "refs/heads/main", "created_at": "2026-09-28T20:10:00Z", "size_in_bytes": 163077969}
]}
JSON

# The oversize fixture pins the size-ceiling policy: a 5.5 GB newest sccache
# entry on main and its 5.5 GB pull-request copy must both be deleted as
# oversize, the newest main entry at exactly decimal 3 GB must survive at the
# inclusive ceiling boundary, and a 6 GB native-build entry must survive
# because the ceiling is family-scoped to sccache.
cat >"$FIXTURE_OVERSIZED_CACHES_JSON" <<'JSON'
{"actions_caches": [
  {"id": 111, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainZ-lockZ", "ref": "refs/heads/main", "created_at": "2026-09-30T03:23:17Z", "size_in_bytes": 5500000000},
  {"id": 112, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainZ-lockY", "ref": "refs/heads/main", "created_at": "2026-09-29T03:23:17Z", "size_in_bytes": 3000000000},
  {"id": 113, "key": "astronomical-v2-sccache-macOS-ARM64-toolchainZ-lockZ", "ref": "refs/pull/900/merge", "created_at": "2026-09-30T03:16:40Z", "size_in_bytes": 5500000001},
  {"id": 401, "key": "astronomical-v2-native-build-macOS-ARM64-identityBig", "ref": "refs/heads/main", "created_at": "2026-09-30T03:23:17Z", "size_in_bytes": 6000000000}
]}
JSON

cat >"${FIXTURE_BIN_DIRECTORY}/gh" <<'SH'
#!/bin/sh
# Test double for gh: serves fixture JSON and records cache deletions.
case "$1" in
    api)
        if [ "$2" = "--paginate" ]; then
            # gh api --paginate <path> --jq <filter>
            exec jq -r "$5" "$GH_SHIM_CACHES_JSON"
        fi
        # gh api <path> --jq <filter>
        exec jq -r "$4" "$GH_SHIM_REPO_JSON"
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

run_prune() {
    : >"$DELETIONS_FILE"
    if [ "${1:-}" = "--dry-run" ]; then
        set -- --dry-run --repo fixture/repo
    else
        set -- --repo fixture/repo
    fi
    PATH="${FIXTURE_BIN_DIRECTORY}:${PATH}" \
    GH_SHIM_CACHES_JSON="${PRUNE_CACHES_FIXTURE:-$FIXTURE_CACHES_JSON}" \
    GH_SHIM_REPO_JSON="$FIXTURE_REPO_JSON" \
    GH_SHIM_DELETIONS_FILE="$DELETIONS_FILE" \
    GH_SHIM_FAIL_DELETIONS="${GH_SHIM_FAIL_DELETIONS:-}" \
        sh "${repository_root}/scripts/ci/prune-ci-caches.sh" "$@" >"$PRUNE_OUTPUT_FILE" 2>&1
}

assert_deletions_match() {
    expected_deletions="$1"
    expectation_description="$2"
    actual_deletions="$(sort -n "$DELETIONS_FILE" | tr '\n' ' ' | sed 's/ $//')"
    if [ "$actual_deletions" != "$expected_deletions" ]; then
        print_error "$expectation_description"
        print_error "expected deletions: [$expected_deletions]"
        print_error "actual deletions:   [$actual_deletions]"
        exit 1
    fi
}

shellcheck "${repository_root}/scripts/ci/prune-ci-caches.sh"

# Contract 1: default run keeps the newest sccache on main, keeps the two
# newest native-build entries, and deletes the stale main sccache, the
# pull-request sccache copy (zero non-default-ref entries kept), and the
# surplus native-build entry.
run_prune ""
assert_deletions_match "102 103 203" "stale main sccache, the pull-request sccache copy, and surplus native-build entries should be deleted"
if grep -q 'cache_id=101\|cache_id=301\|cache_id=201\|cache_id=202' "$PRUNE_OUTPUT_FILE"; then
    print_error "kept caches must not appear as deleted"
    exit 1
fi
if ! grep -q 'status=complete outcome=pruned deleted=3' "$PRUNE_OUTPUT_FILE"; then
    print_error "the run should close with a pruned summary of three deletions"
    cat "$PRUNE_OUTPUT_FILE" >&2
    exit 1
fi

# Contract 2: a failed cache deletion is reported but does not abort the run.
GH_SHIM_FAIL_DELETIONS="102" run_prune ""
unset GH_SHIM_FAIL_DELETIONS
assert_deletions_match "103 203" "a failed deletion must not stop the remaining deletions"
if ! grep -q 'action=delete-failed cache_id=102' "$PRUNE_OUTPUT_FILE"; then
    print_error "the failed deletion should be reported visibly"
    cat "$PRUNE_OUTPUT_FILE" >&2
    exit 1
fi

# Contract 3: dry-run reports what it would delete but performs no deletions.
run_prune "--dry-run"
if [ -s "$DELETIONS_FILE" ]; then
    print_error "dry-run must not delete any cache"
    exit 1
fi
if ! grep -q 'action=would-delete reason=surplus cache_id=102' "$PRUNE_OUTPUT_FILE" ||
    ! grep -q 'action=would-delete reason=surplus cache_id=103' "$PRUNE_OUTPUT_FILE" ||
    ! grep -q 'action=would-delete reason=surplus cache_id=203' "$PRUNE_OUTPUT_FILE"; then
    print_error "dry-run should report the stale sccache, its pull-request copy, and the stale native-build as would-delete"
    cat "$PRUNE_OUTPUT_FILE" >&2
    exit 1
fi

# Contract 4: an oversized sccache generation is deleted even when it is the
# newest on main, its oversized pull-request copy is deleted too, the newest
# main entry at exactly decimal 3 GB survives at the inclusive ceiling
# boundary, and the sccache ceiling never reaches the native-build family.
PRUNE_CACHES_FIXTURE="$FIXTURE_OVERSIZED_CACHES_JSON" run_prune ""
unset PRUNE_CACHES_FIXTURE
assert_deletions_match "111 113" "oversized sccache entries must be deleted regardless of recency"
if grep -q 'cache_id=112\|cache_id=401' "$PRUNE_OUTPUT_FILE"; then
    print_error "the under-ceiling sccache generation and the oversized native-build entry must be kept"
    exit 1
fi
if ! grep -q 'action=deleted reason=oversize cache_id=111' "$PRUNE_OUTPUT_FILE"; then
    print_error "oversized deletions must be attributed with reason=oversize"
    cat "$PRUNE_OUTPUT_FILE" >&2
    exit 1
fi
if ! grep -q 'family=sccache status=complete.*oversized_deleted=2' "$PRUNE_OUTPUT_FILE"; then
    print_error "the family summary must count oversized deletions"
    cat "$PRUNE_OUTPUT_FILE" >&2
    exit 1
fi

# Contract 5: dry-run reports oversized sccache entries but deletes nothing.
PRUNE_CACHES_FIXTURE="$FIXTURE_OVERSIZED_CACHES_JSON" run_prune "--dry-run"
unset PRUNE_CACHES_FIXTURE
if [ -s "$DELETIONS_FILE" ]; then
    print_error "dry-run must not delete any cache"
    exit 1
fi
if ! grep -q 'action=would-delete reason=oversize cache_id=111' "$PRUNE_OUTPUT_FILE"; then
    print_error "dry-run should report the oversized sccache entry as would-delete"
    cat "$PRUNE_OUTPUT_FILE" >&2
    exit 1
fi

printf '%s\n' "prune-ci-caches contract tests passed"
