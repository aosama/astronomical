#!/usr/bin/env sh

# Deletes surplus GitHub Actions caches so the small native-build cache entries
# stop being evicted by the Actions 10 GB repository limit under LRU pressure.
# Retention keeps the newest sccache entry per ref and the two newest
# native-build entries per ref; caches saved on a merge ref are only
# restorable on that ref, so retention counts per ref rather than globally.
# Every failure mode degrades to a skip: fork pull requests carry a read-only
# token, and cache pruning must never fail a CI run.

set -eu
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    # shellcheck disable=SC3040
    set -o pipefail
fi

readonly CACHE_LIST_LIMIT=1000

SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        rm -rf "$SANDBOX_DIRECTORY"
    fi
}
trap cleanup 0

# Selects surplus caches newest-first. stdout carries tab-separated
# id/key/ref deletion lines; exit 3 means selection itself failed and the
# caller must skip rather than delete anything.
# shellcheck disable=SC2016
select_surplus_caches() {
    ruby -rjson -rtime -e '
      begin
        cache_entries = JSON.parse($stdin.read)
        retention_by_owner = { "sccache" => 1, "native-build" => 2 }
        known_owners = %w[cargo-downloads native-archives native-build sccache swiftpm]
        sorted_entries = cache_entries.sort_by { |cache_entry| [Time.iso8601(cache_entry.fetch("createdAt")), cache_entry.fetch("id")] }.reverse
        kept_counts_by_owner_and_ref = Hash.new { |counts, owner_and_ref| counts[owner_and_ref] = 0 }
        sorted_entries.each do |cache_entry|
          cache_key = cache_entry.fetch("key").to_s
          matched_owner = known_owners.find { |known_owner| cache_key.start_with?("astronomical-v2-#{known_owner}-") }
          next unless retention_by_owner.key?(matched_owner)
          owner_and_ref = [matched_owner, cache_entry.fetch("ref").to_s]
          kept_counts_by_owner_and_ref[owner_and_ref] += 1
          next if kept_counts_by_owner_and_ref[owner_and_ref] <= retention_by_owner[matched_owner]
          puts [cache_entry.fetch("id"), cache_key, cache_entry.fetch("ref")].join("\t")
        end
      rescue StandardError => selection_error
        warn "[cache-prune] status=selection-failed reason=#{selection_error.class}"
        exit 3
      end
    '
}

main() {
    dry_run=0
    for script_argument in "$@"; do
        case "$script_argument" in
            --dry-run) dry_run=1 ;;
            *)
                print_error "unknown argument: ${script_argument} (only --dry-run is supported)"
                exit 2
                ;;
        esac
    done

    if ! command -v gh >/dev/null 2>&1; then
        printf '%s\n' "[cache-prune] status=skipped reason=gh-cli-unavailable"
        exit 0
    fi
    if [ -z "${GITHUB_REPOSITORY:-}" ]; then
        printf '%s\n' "[cache-prune] status=skipped reason=repository-unset"
        exit 0
    fi

    prune_started_at="$(date +%s)"
    printf '%s\n' "[cache-prune] status=started repository=${GITHUB_REPOSITORY} dry_run=${dry_run}"

    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-cache-prune.XXXXXX")"
    cache_list_file="${SANDBOX_DIRECTORY}/cache-list.json"
    surplus_cache_file="${SANDBOX_DIRECTORY}/surplus-caches.tsv"

    if ! gh cache list \
        -R "$GITHUB_REPOSITORY" \
        --json id,key,ref,createdAt \
        --limit "$CACHE_LIST_LIMIT" > "$cache_list_file" 2>/dev/null; then
        printf '%s\n' "[cache-prune] status=skipped reason=cache-list-unavailable"
        exit 0
    fi

    if ! select_surplus_caches < "$cache_list_file" > "$surplus_cache_file"; then
        printf '%s\n' "[cache-prune] status=skipped reason=cache-selection-unavailable"
        exit 0
    fi

    deleted_cache_count=0
    failed_deletion_count=0
    while IFS="$(printf '\t')" read -r cache_id cache_key cache_ref; do
        [ -n "$cache_id" ] || continue
        if [ "$dry_run" -eq 1 ]; then
            printf '%s\n' "[cache-prune] status=would-delete cache_id=${cache_id} cache_key=${cache_key} cache_ref=${cache_ref}"
            continue
        fi
        if gh cache delete -R "$GITHUB_REPOSITORY" "$cache_id" >/dev/null 2>&1; then
            printf '%s\n' "[cache-prune] status=deleted cache_id=${cache_id} cache_key=${cache_key} cache_ref=${cache_ref}"
            deleted_cache_count=$((deleted_cache_count + 1))
        else
            printf '%s\n' "[cache-prune] status=delete-failed cache_id=${cache_id} cache_key=${cache_key} cache_ref=${cache_ref}"
            failed_deletion_count=$((failed_deletion_count + 1))
        fi
    done < "$surplus_cache_file"

    prune_elapsed_seconds=$(( $(date +%s) - prune_started_at ))
    printf '%s\n' "[cache-prune] status=complete deleted=${deleted_cache_count} failed=${failed_deletion_count} elapsed_seconds=${prune_elapsed_seconds}"

    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            printf '%s\n' "## Surplus CI cache prune"
            printf '%s\n' ""
            printf '%s\n' "- Deleted caches: ${deleted_cache_count}"
            printf '%s\n' "- Failed deletions: ${failed_deletion_count}"
            if [ "$dry_run" -eq 1 ]; then
                printf '%s\n' "- Mode: dry run (no caches were deleted)"
            fi
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}

main "$@"
