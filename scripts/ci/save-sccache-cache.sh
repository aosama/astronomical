#!/bin/sh
# Decide whether the sccache-trimmed directory may be uploaded as a new cache
# generation, and clear stored copies of the primary key that can never serve
# within budget. The save gate alone cannot shrink a stored generation: an
# exact-key restore skips the save, so an entry written before the sccache cap
# existed is restored and re-downloaded forever. This script measures the
# directory after sccache enforced its cap, deletes stored oversize copies of
# the primary key, and reports the save decision through GitHub Actions
# outputs so the workflow's save step stays a thin conditional.
#
# Unit accounting: sccache parses its cap as a binary multiple (2700M budgets
# 2.83 decimal GB of tracked entries) while GitHub reports entry sizes in
# decimal bytes, so the directory budget sits below the decimal 3 GB entry
# ceiling to absorb cache-archive overhead.
#
# Usage:
#   scripts/ci/save-sccache-cache.sh <primary-cache-key>
#
# Environment:
#   GITHUB_REPOSITORY                 owner/name repository (required)
#   GH_TOKEN / GITHUB_TOKEN           token used by the gh CLI
#   SCCACHE_DIR                       sccache cache directory to measure
#   SCCACHE_SAVE_MAX_DIRECTORY_BYTES  decimal bytes; a larger directory skips
#                                     the save (default 2900000000)
#   SCCACHE_SAVE_MAX_ENTRY_BYTES      decimal bytes; stored entries above this
#                                     are deleted (default 3000000000)
#   GITHUB_OUTPUT                     GitHub Actions outputs file when run
#                                     inside a workflow

set -eu
# pipefail is Bash/Zsh; the subshell probe keeps this script POSIX-runnable.
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    # shellcheck disable=SC3040
    set -o pipefail
fi

DEFAULT_MAX_DIRECTORY_BYTES=2900000000
DEFAULT_MAX_ENTRY_BYTES=3000000000

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <primary-cache-key>" >&2
    exit 2
fi
PRIMARY_CACHE_KEY="$1"

if [ -z "${GITHUB_REPOSITORY:-}" ]; then
    print_error "GITHUB_REPOSITORY is required"
    exit 2
fi
if [ -z "${SCCACHE_DIR:-}" ]; then
    print_error "SCCACHE_DIR is required"
    exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
    print_error "required tool 'gh' is not on PATH"
    exit 2
fi

MAX_DIRECTORY_BYTES="${SCCACHE_SAVE_MAX_DIRECTORY_BYTES:-$DEFAULT_MAX_DIRECTORY_BYTES}"
MAX_ENTRY_BYTES="${SCCACHE_SAVE_MAX_ENTRY_BYTES:-$DEFAULT_MAX_ENTRY_BYTES}"
case "${MAX_DIRECTORY_BYTES}${MAX_ENTRY_BYTES}" in
    ''|*[!0-9]*)
        print_error "sccache save budgets must be decimal integers"
        exit 2
        ;;
esac

WORK_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "$WORK_DIRECTORY"' EXIT INT TERM

SAVE_STARTED_AT="$(date +%s)"

echo "[sccache-save] status=start repository=$GITHUB_REPOSITORY max_directory_bytes=$MAX_DIRECTORY_BYTES max_entry_bytes=$MAX_ENTRY_BYTES"

emit_output() {
    output_name="$1"
    output_value="$2"
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        printf '%s=%s\n' "$output_name" "$output_value" >>"$GITHUB_OUTPUT"
    fi
    printf '[sccache-save] output=%s value=%s\n' "$output_name" "$output_value"
}

finish_with_decision() {
    decision_reason="$1"
    emit_output should_save false
    emit_output decision "$decision_reason"
    emit_output directory_bytes "$DIRECTORY_BYTES"
    printf '[sccache-save] status=complete decision=%s directory_bytes=%s elapsed_seconds=%s\n' \
        "$decision_reason" "$DIRECTORY_BYTES" "$(( $(date +%s) - SAVE_STARTED_AT ))"
    exit 0
}

# Apparent-size accounting, not allocated blocks: the GitHub cache entry is
# billed by archive bytes, so block-rounding slack would bias the decision.
stat_size_probe_path="$WORK_DIRECTORY/size-probe"
: >"$stat_size_probe_path"
if stat -f %z "$stat_size_probe_path" >/dev/null 2>&1; then
    # BSD stat (macOS runners)
    query_file_size_bytes() {
        xargs -0 stat -f %z
    }
else
    # GNU stat (Linux runners)
    query_file_size_bytes() {
        xargs -0 stat -c %s
    }
fi

DIRECTORY_BYTES="$(
    {
        find "$SCCACHE_DIR" -type f -print0 2>/dev/null |
            query_file_size_bytes 2>/dev/null |
            awk '{ total += $1 } END { printf "%.0f", total + 0 }'
    } || true
)"
case "${DIRECTORY_BYTES:-}" in
    ''|*[!0-9]*) DIRECTORY_BYTES=0 ;;
esac
echo "[sccache-save] directory_bytes=$DIRECTORY_BYTES"

CACHE_LIST_PATH="$WORK_DIRECTORY/caches.tsv"
if ! gh api --paginate "repos/$GITHUB_REPOSITORY/actions/caches" \
        --jq '.actions_caches[] | [.id, .key, .ref, .created_at, .size_in_bytes] | @tsv' \
        >"$CACHE_LIST_PATH"; then
    # Read-only fork tokens cannot list caches; the save is default-branch-only
    # anyway, so a degraded listing skips the save instead of failing the job.
    finish_with_decision cache-list-unavailable
fi

EXACT_KEY_ENTRIES_PATH="$WORK_DIRECTORY/exact-key-entries.tsv"
awk -F'\t' -v primary_key="$PRIMARY_CACHE_KEY" \
    '$2 == primary_key { print $1 "\t" $3 "\t" $5 }' \
    "$CACHE_LIST_PATH" >"$EXACT_KEY_ENTRIES_PATH"
STORED_ENTRY_COUNT="$(awk 'END { print NR }' "$EXACT_KEY_ENTRIES_PATH")"
echo "[sccache-save] stored_exact_key_entries=$STORED_ENTRY_COUNT"

DELETION_BLOCKED=0
DELETED_OVERSIZE_COUNT=0
while IFS="$(printf '\t')" read -r cache_id cache_ref cache_size_bytes; do
    [ -n "$cache_id" ] || continue
    [ "$cache_size_bytes" -gt "$MAX_ENTRY_BYTES" ] || continue
    if gh cache delete "$cache_id" -R "$GITHUB_REPOSITORY" 2>"$WORK_DIRECTORY/delete-error.txt"; then
        DELETED_OVERSIZE_COUNT=$((DELETED_OVERSIZE_COUNT + 1))
        echo "[sccache-save] action=deleted-oversize cache_id=$cache_id ref=$cache_ref size_bytes=$cache_size_bytes"
    else
        # A blocked deletion leaves the oversize entry in place; saving under
        # the same key would collide with it, so the save is skipped and the
        # prune pass owns the retry on a later run.
        DELETION_BLOCKED=1
        echo "[sccache-save] action=delete-failed cache_id=$cache_id ref=$cache_ref" >&2
        sed 's/^/[sccache-save] gh: /' "$WORK_DIRECTORY/delete-error.txt" >&2
    fi
done <"$EXACT_KEY_ENTRIES_PATH"

if [ "$DELETION_BLOCKED" -eq 1 ]; then
    finish_with_decision replace-blocked
fi
if [ "$DIRECTORY_BYTES" -gt "$MAX_DIRECTORY_BYTES" ]; then
    finish_with_decision directory-over-budget
fi
if [ "$STORED_ENTRY_COUNT" -gt 0 ] && [ "$DELETED_OVERSIZE_COUNT" -eq 0 ]; then
    finish_with_decision fresh-entry-within-budget
fi
if [ "$DELETED_OVERSIZE_COUNT" -gt 0 ]; then
    SAVE_DECISION="replace-oversize-entry"
else
    SAVE_DECISION="new-generation"
fi

emit_output should_save true
emit_output decision "$SAVE_DECISION"
emit_output directory_bytes "$DIRECTORY_BYTES"
printf '[sccache-save] status=complete decision=%s directory_bytes=%s deleted_oversize=%s elapsed_seconds=%s\n' \
    "$SAVE_DECISION" "$DIRECTORY_BYTES" "$DELETED_OVERSIZE_COUNT" "$(( $(date +%s) - SAVE_STARTED_AT ))"
