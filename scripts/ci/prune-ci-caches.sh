#!/bin/sh
# Prune surplus GitHub Actions caches so the repository stays under the 10 GB
# cache storage limit. Two sccache generations alone can reach ~6.5 GB, and
# when the limit is exceeded GitHub evicts caches least-recently-used, which
# has evicted the small native-build cache and forced silent multi-minute
# CMake rebuilds in CI. The policy keeps the newest entries per cache family:
# the newest on the default branch, plus the newest across feature-branch refs
# for families that still save on pull requests, so an in-flight pull request
# keeps its warm cache while stale toolchain generations are removed. sccache
# saves are default-branch-only, so a non-default-ref sccache entry can never
# be superseded by a newer generation; keeping one preserves a stale
# multi-gigabyte directory forever, and the family keeps zero of them. On top
# of the count policy, sccache entries above a decimal 3 GB ceiling are
# deleted even when newest: the sccache cap budgets 2.83 decimal GB of tracked
# entries, so a larger generation was never trimmed, cannot restore inside the
# one-minute cache segment timeout, and only hastens the 10 GB eviction that
# kills the native-build cache. Count alone cannot bound the budget because
# each sccache generation inherits the whole previous directory through the
# restore-key prefix chain.
#
# Usage:
#   scripts/ci/prune-ci-caches.sh [--dry-run] [--repo owner/name]
#
# Environment:
#   GITHUB_TOKEN / GH_TOKEN  token used by the gh CLI (CI passes github.token)
#   GITHUB_REPOSITORY        default repository when --repo is not given

set -eu
# pipefail is Bash/Zsh; the subshell probe keeps this script POSIX-runnable.
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    # shellcheck disable=SC3040
    set -o pipefail
fi

CACHE_KEY_PREFIX="astronomical-v2-"
# native-build keeps two entries per ref class because the Xcode runner image
# flip-flops between compatibility identities and both must stay warm.
NATIVE_BUILD_KEEP_COUNT=2
DEFAULT_KEEP_COUNT=1
# Size ceilings are family-scoped because only sccache generations ratchet
# upward; 0 disables the ceiling for families that own bounded artifacts. The
# sccache cap (SCCACHE_CACHE_SIZE=2700M, 2.83 decimal GB of tracked entries)
# plus archive overhead must land under this decimal ceiling.
SCCACHE_MAX_ENTRY_BYTES=3000000000
DEFAULT_MAX_ENTRY_BYTES=0
# sccache saves are default-branch-only, so a non-default-ref sccache entry is
# never superseded and only burns budget; every other family keeps one entry
# across non-default refs so an in-flight pull request stays warm.
SCCACHE_OTHER_REF_KEEP_COUNT=0
DEFAULT_OTHER_REF_KEEP_COUNT=1

DRY_RUN=0
REPOSITORY="${GITHUB_REPOSITORY:-}"

print_usage() {
    echo "usage: $0 [--dry-run] [--repo owner/name]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --repo)
            [ "$#" -ge 2 ] || { print_usage; exit 2; }
            REPOSITORY="$2"
            shift
            ;;
        *)
            print_usage
            exit 2
            ;;
    esac
    shift
done

if [ -z "$REPOSITORY" ]; then
    echo "error: no repository given; pass --repo owner/name or set GITHUB_REPOSITORY" >&2
    exit 2
fi

for required_tool in gh jq; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        echo "error: required tool '$required_tool' is not on PATH" >&2
        exit 2
    fi
done

WORK_DIRECTORY="$(mktemp -d)"
trap 'rm -rf "$WORK_DIRECTORY"' EXIT INT TERM

CACHE_LIST_PATH="$WORK_DIRECTORY/caches.tsv"

PRUNE_STARTED_AT="$(date +%s)"

echo "[cache-prune] status=start repository=$REPOSITORY dry_run=$DRY_RUN"

DEFAULT_BRANCH="$(gh api "repos/$REPOSITORY" --jq .default_branch)"
echo "[cache-prune] default_branch=$DEFAULT_BRANCH"

# One TSV line per cache: id, key, ref, created_at, size_in_bytes.
gh api --paginate "repos/$REPOSITORY/actions/caches" \
    --jq '.actions_caches[] | [.id, .key, .ref, .created_at, .size_in_bytes] | @tsv' \
    >"$CACHE_LIST_PATH"

if [ ! -s "$CACHE_LIST_PATH" ]; then
    echo "[cache-prune] status=complete outcome=no-caches-found"
    exit 0
fi

DEFAULT_BRANCH_REF="refs/heads/$DEFAULT_BRANCH"

keep_count_for_family() {
    family_name="$1"
    if [ "$family_name" = "native-build" ]; then
        echo "$NATIVE_BUILD_KEEP_COUNT"
    else
        echo "$DEFAULT_KEEP_COUNT"
    fi
}

max_entry_bytes_for_family() {
    family_name="$1"
    case "$family_name" in
        sccache) echo "$SCCACHE_MAX_ENTRY_BYTES" ;;
        *) echo "$DEFAULT_MAX_ENTRY_BYTES" ;;
    esac
}

other_ref_keep_count_for_family() {
    family_name="$1"
    if [ "$family_name" = "sccache" ]; then
        echo "$SCCACHE_OTHER_REF_KEEP_COUNT"
    else
        echo "$DEFAULT_OTHER_REF_KEEP_COUNT"
    fi
}

# The family is the token between the astronomical-v2- prefix and the runner
# platform segment (every key embeds -macOS- on this repository's runners).
family_of_key() {
    cache_key="$1"
    stripped_key="${cache_key#"$CACHE_KEY_PREFIX"}"
    family_name="${stripped_key%%-macOS*}"
    if [ "$family_name" = "$stripped_key" ]; then
        family_name="$(printf '%s' "$stripped_key" | cut -d- -f1)"
    fi
    echo "$family_name"
}

TOTAL_DELETED_COUNT=0
TOTAL_FREED_BYTES=0

# created_at is ISO 8601 UTC, so lexicographic ordering is chronological.
FAMILIES="$(cut -f2 "$CACHE_LIST_PATH" | while IFS="$(printf '\t')" read -r cache_key; do
    family_of_key "$cache_key"
done | sort -u)"

for FAMILY in $FAMILIES; do
    FAMILY_STARTED_AT="$(date +%s)"
    FAMILY_LIST_PATH="$WORK_DIRECTORY/family-$FAMILY.tsv"
    grep -F "$(printf '\t%s' "astronomical-v2-$FAMILY-")" "$CACHE_LIST_PATH" >"$FAMILY_LIST_PATH" ||
        continue

    KEEP_COUNT="$(keep_count_for_family "$FAMILY")"
    OTHER_REF_KEEP_COUNT="$(other_ref_keep_count_for_family "$FAMILY")"
    MAX_ENTRY_BYTES="$(max_entry_bytes_for_family "$FAMILY")"
    FAMILY_DELETED_COUNT=0
    FAMILY_OVERSIZED_DELETED_COUNT=0
    FAMILY_FREED_BYTES=0

    # Newest first within each ref class; keep the newest KEEP_COUNT entries on
    # the default branch and the newest OTHER_REF_KEEP_COUNT entries across all
    # other refs. Entries above the family's size ceiling are deleted first,
    # even when they are the newest, because they can never restore within
    # budget.
    DELETION_IDS="$(LC_ALL=C sort -t"$(printf '\t')" -k4,4r "$FAMILY_LIST_PATH" |
        awk -F'\t' -v default_ref="$DEFAULT_BRANCH_REF" -v keep_count="$KEEP_COUNT" -v other_keep_count="$OTHER_REF_KEEP_COUNT" -v size_ceiling="$MAX_ENTRY_BYTES" '
            size_ceiling > 0 && ($5 + 0) > size_ceiling { print $1; next }
            $3 == default_ref { kept_main++; if (kept_main <= keep_count) next }
            $3 != default_ref { kept_other++; if (kept_other <= other_keep_count) next }
            { print $1 }
        ')"

    for CACHE_ID in $DELETION_IDS; do
        CACHE_SIZE_BYTES="$(
            awk -F'\t' -v cache_id="$CACHE_ID" '$1 == cache_id { print $5; exit }' "$FAMILY_LIST_PATH"
        )"
        if [ "$MAX_ENTRY_BYTES" -gt 0 ] && [ "$CACHE_SIZE_BYTES" -gt "$MAX_ENTRY_BYTES" ]; then
            DELETION_REASON="oversize"
        else
            DELETION_REASON="surplus"
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[cache-prune] family=$FAMILY action=would-delete reason=$DELETION_REASON cache_id=$CACHE_ID size_bytes=$CACHE_SIZE_BYTES"
            FAMILY_DELETED_COUNT=$((FAMILY_DELETED_COUNT + 1))
            FAMILY_FREED_BYTES=$((FAMILY_FREED_BYTES + CACHE_SIZE_BYTES))
            if [ "$DELETION_REASON" = "oversize" ]; then
                FAMILY_OVERSIZED_DELETED_COUNT=$((FAMILY_OVERSIZED_DELETED_COUNT + 1))
            fi
            continue
        fi
        if gh cache delete "$CACHE_ID" -R "$REPOSITORY" 2>"$WORK_DIRECTORY/delete-error.txt"; then
            FAMILY_DELETED_COUNT=$((FAMILY_DELETED_COUNT + 1))
            FAMILY_FREED_BYTES=$((FAMILY_FREED_BYTES + CACHE_SIZE_BYTES))
            if [ "$DELETION_REASON" = "oversize" ]; then
                FAMILY_OVERSIZED_DELETED_COUNT=$((FAMILY_OVERSIZED_DELETED_COUNT + 1))
            fi
            echo "[cache-prune] family=$FAMILY action=deleted reason=$DELETION_REASON cache_id=$CACHE_ID size_bytes=$CACHE_SIZE_BYTES"
        else
            # A 404 here means the cache vanished between listing and deletion;
            # warn visibly but keep pruning the rest of the family.
            echo "[cache-prune] family=$FAMILY action=delete-failed cache_id=$CACHE_ID" >&2
            sed 's/^/[cache-prune] gh: /' "$WORK_DIRECTORY/delete-error.txt" >&2
        fi
    done

    FAMILY_ELAPSED_SECONDS=$(( $(date +%s) - FAMILY_STARTED_AT ))
    echo "[cache-prune] family=$FAMILY status=complete deleted=$FAMILY_DELETED_COUNT oversized_deleted=$FAMILY_OVERSIZED_DELETED_COUNT freed_bytes=$FAMILY_FREED_BYTES elapsed_seconds=$FAMILY_ELAPSED_SECONDS"
    TOTAL_DELETED_COUNT=$((TOTAL_DELETED_COUNT + FAMILY_DELETED_COUNT))
    TOTAL_FREED_BYTES=$((TOTAL_FREED_BYTES + FAMILY_FREED_BYTES))
done

PRUNE_ELAPSED_SECONDS=$(( $(date +%s) - PRUNE_STARTED_AT ))
if [ "$DRY_RUN" -eq 1 ]; then
    PRUNE_OUTCOME="dry-run"
else
    PRUNE_OUTCOME="pruned"
fi
echo "[cache-prune] status=complete outcome=$PRUNE_OUTCOME deleted=$TOTAL_DELETED_COUNT freed_bytes=$TOTAL_FREED_BYTES elapsed_seconds=$PRUNE_ELAPSED_SECONDS"
