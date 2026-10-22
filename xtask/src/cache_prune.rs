//! `cargo xtask prune-caches` — prunes surplus GitHub Actions caches so the
//! repository stays under the 10 GB cache storage limit. The deletion policy
//! is pure and unit-tested beside this source; the `gh` list and delete
//! calls stay thin edges (issue #1250).

use std::process::Command;

const CACHE_KEY_PREFIX: &str = "astronomical-v2-";
/// native-build keeps two entries per ref class because the Xcode runner
/// image flip-flops between native build identities and both must stay warm.
const NATIVE_BUILD_KEEP_COUNT: usize = 2;
const DEFAULT_KEEP_COUNT: usize = 1;
/// Size ceilings are family-scoped because only sccache generations ratchet
/// upward; 0 disables the ceiling for families that own bounded artifacts.
/// The sccache cap (`SCCACHE_CACHE_SIZE=400M`, 0.419 decimal GB of tracked
/// entries) plus archive overhead must land under this decimal ceiling.
const SCCACHE_MAX_ENTRY_BYTES: u64 = 450_000_000;
const DEFAULT_MAX_ENTRY_BYTES: u64 = 0;
/// sccache saves are default-branch-only, so a non-default-ref sccache entry
/// is never superseded and only burns budget; every other family keeps one
/// entry across non-default refs so an in-flight pull request stays warm.
const SCCACHE_OTHER_REF_KEEP_COUNT: usize = 0;
const DEFAULT_OTHER_REF_KEEP_COUNT: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    pub id: u64,
    pub key: String,
    pub ref_name: String,
    pub created_at: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deletion {
    pub id: u64,
    pub reason: DeletionReason,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeletionReason {
    Oversize,
    Surplus,
}

/// The family is the token between the `astronomical-v2-` prefix and the
/// runner platform segment (every key embeds `-macOS-` on this repository's
/// runners).
pub fn family_of_key(cache_key: &str) -> &str {
    let stripped_key = cache_key
        .strip_prefix(CACHE_KEY_PREFIX)
        .unwrap_or(cache_key);
    match stripped_key.split_once("-macOS") {
        Some((family_name, _)) => family_name,
        None => stripped_key.split('-').next().unwrap_or(stripped_key),
    }
}

fn keep_count_for_family(family_name: &str) -> usize {
    if family_name == "native-build" {
        NATIVE_BUILD_KEEP_COUNT
    } else {
        DEFAULT_KEEP_COUNT
    }
}

fn max_entry_bytes_for_family(family_name: &str) -> u64 {
    if family_name == "sccache" {
        SCCACHE_MAX_ENTRY_BYTES
    } else {
        DEFAULT_MAX_ENTRY_BYTES
    }
}

fn other_ref_keep_count_for_family(family_name: &str) -> usize {
    if family_name == "sccache" {
        SCCACHE_OTHER_REF_KEEP_COUNT
    } else {
        DEFAULT_OTHER_REF_KEEP_COUNT
    }
}

/// The pure deletion plan: entries above the family's size ceiling are
/// deleted first even when newest (they can never restore within budget),
/// then the newest `keep_count` entries on the default branch and the newest
/// `other_ref_keep_count` entries across all other refs are kept, and the
/// remainder is deleted. `created_at` is ISO 8601 UTC, so lexicographic
/// ordering is chronological.
pub fn deletion_plan(entries: &[CacheEntry], default_branch_ref: &str) -> Vec<Deletion> {
    let mut family_names: Vec<&str> = entries
        .iter()
        .map(|cache_entry| family_of_key(&cache_entry.key))
        .collect();
    family_names.sort_unstable();
    family_names.dedup();

    let mut deletions = Vec::new();
    for family_name in family_names {
        let mut family_entries: Vec<&CacheEntry> = entries
            .iter()
            .filter(|cache_entry| family_of_key(&cache_entry.key) == family_name)
            .collect();
        family_entries.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        let mut kept_on_default_branch = 0usize;
        let mut kept_on_other_refs = 0usize;
        for cache_entry in family_entries {
            let size_ceiling = max_entry_bytes_for_family(family_name);
            if size_ceiling > 0 && cache_entry.size_bytes > size_ceiling {
                deletions.push(Deletion {
                    id: cache_entry.id,
                    reason: DeletionReason::Oversize,
                    size_bytes: cache_entry.size_bytes,
                });
                continue;
            }
            if cache_entry.ref_name == default_branch_ref {
                kept_on_default_branch += 1;
                if kept_on_default_branch <= keep_count_for_family(family_name) {
                    continue;
                }
            } else {
                kept_on_other_refs += 1;
                if kept_on_other_refs <= other_ref_keep_count_for_family(family_name) {
                    continue;
                }
            }
            deletions.push(Deletion {
                id: cache_entry.id,
                reason: DeletionReason::Surplus,
                size_bytes: cache_entry.size_bytes,
            });
        }
    }
    deletions
}

/// Parses the `id \t key \t ref \t created_at \t size` TSV the gh jq filter
/// emits.
pub fn parse_cache_list_tsv(tsv_text: &str) -> Vec<CacheEntry> {
    tsv_text
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 5 {
                return None;
            }
            Some(CacheEntry {
                id: fields[0].parse().ok()?,
                key: fields[1].to_owned(),
                ref_name: fields[2].to_owned(),
                created_at: fields[3].to_owned(),
                size_bytes: fields[4].parse().ok()?,
            })
        })
        .collect()
}

pub fn run(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    let started_at = std::time::Instant::now();
    let mut dry_run = false;
    let mut repository = std::env::var("GITHUB_REPOSITORY").ok();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--dry-run" => dry_run = true,
            "--repo" => {
                repository = Some(
                    arguments
                        .next()
                        .ok_or("--repo requires owner/name".to_owned())?,
                );
            }
            other => return Err(format!("unexpected prune-caches argument: {other}")),
        }
    }
    let repository =
        repository.ok_or("no repository given; pass --repo owner/name or set GITHUB_REPOSITORY")?;
    eprintln!("[cache-prune] status=start repository={repository} dry_run={dry_run}");

    let default_branch = gh_output(&[
        "api",
        &format!("repos/{repository}"),
        "--jq",
        ".default_branch",
    ])?
    .trim()
    .to_owned();
    eprintln!("[cache-prune] default_branch={default_branch}");
    let cache_list_text = gh_output(&[
        "api",
        "--paginate",
        &format!("repos/{repository}/actions/caches"),
        "--jq",
        ".actions_caches[] | [.id, .key, .ref, .created_at, .size_in_bytes] | @tsv",
    ])?;
    let cache_entries = parse_cache_list_tsv(&cache_list_text);
    if cache_entries.is_empty() {
        eprintln!("[cache-prune] status=complete outcome=no-caches-found");
        return Ok(());
    }

    let default_branch_ref = format!("refs/heads/{default_branch}");
    let deletions = deletion_plan(&cache_entries, &default_branch_ref);
    let mut total_deleted_count = 0usize;
    let mut total_freed_bytes = 0u64;
    for deletion in &deletions {
        let reason_text = match deletion.reason {
            DeletionReason::Oversize => "oversize",
            DeletionReason::Surplus => "surplus",
        };
        if dry_run {
            eprintln!(
                "[cache-prune] action=would-delete reason={reason_text} cache_id={} size_bytes={}",
                deletion.id, deletion.size_bytes
            );
        } else {
            let delete_status = Command::new("gh")
                .args([
                    "cache",
                    "delete",
                    &deletion.id.to_string(),
                    "-R",
                    &repository,
                ])
                .status()
                .map_err(|error| format!("gh cache delete failed to start: {error}"))?;
            if !delete_status.success() {
                // A vanished cache (404) must not fail the prune; the next
                // run relists and retries.
                eprintln!(
                    "[cache-prune] action=delete-failed cache_id={}",
                    deletion.id
                );
                continue;
            }
            eprintln!(
                "[cache-prune] action=deleted reason={reason_text} cache_id={} size_bytes={}",
                deletion.id, deletion.size_bytes
            );
        }
        total_deleted_count += 1;
        total_freed_bytes += deletion.size_bytes;
    }
    let outcome_text = if dry_run { "dry-run" } else { "pruned" };
    eprintln!(
        "[cache-prune] status=complete outcome={outcome_text} deleted={total_deleted_count} freed_bytes={total_freed_bytes} elapsed_seconds={}",
        started_at.elapsed().as_secs()
    );
    Ok(())
}

fn gh_output(gh_arguments: &[&str]) -> Result<String, String> {
    let gh_output = Command::new("gh")
        .args(gh_arguments)
        .output()
        .map_err(|error| format!("gh failed to start: {error}"))?;
    if !gh_output.status.success() {
        return Err(format!(
            "gh api failed with {}: {}",
            gh_output.status,
            String::from_utf8_lossy(&gh_output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&gh_output.stdout).into_owned())
}
