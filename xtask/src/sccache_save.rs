//! `cargo xtask save-sccache` — decides whether the sccache-trimmed
//! directory may be uploaded as a new cache generation, and clears stored
//! copies of the primary key that can never serve within budget. The
//! decision policy is pure and unit-tested beside this source; the
//! directory walk and `gh` calls stay thin edges (issue #1250). Directory
//! and stored-entry budgets are repository policy, not environment inputs.

use std::path::Path;
use std::process::Command;

/// 600M binary sccache cap = 629,145,600 tracked bytes; the directory trim
/// and the stored-entry refusal share a decimal 680 MB ceiling so the saved
/// entry plus cache-archive overhead stays under the prune's family ceiling.
const MAX_DIRECTORY_BYTES: u64 = 680_000_000;
const MAX_ENTRY_BYTES: u64 = 680_000_000;

/// The save decision the workflow's thin save step consumes. An exact-key
/// restore skips the save step entirely, so this gate exists to shrink the
/// stored stock: oversize stored copies of the primary key are deleted, and
/// a save runs only to replace one of them or record a fresh generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SccacheSaveDecision {
    NewGeneration,
    ReplaceOversizeEntry,
    FreshEntryWithinBudget,
    DirectoryOverBudget,
    ReplaceBlocked,
    CacheListUnavailable,
}

impl SccacheSaveDecision {
    pub fn should_save(self) -> bool {
        matches!(self, Self::NewGeneration | Self::ReplaceOversizeEntry)
    }

    pub fn as_decision_text(self) -> &'static str {
        match self {
            Self::NewGeneration => "new-generation",
            Self::ReplaceOversizeEntry => "replace-oversize-entry",
            Self::FreshEntryWithinBudget => "fresh-entry-within-budget",
            Self::DirectoryOverBudget => "directory-over-budget",
            Self::ReplaceBlocked => "replace-blocked",
            Self::CacheListUnavailable => "cache-list-unavailable",
        }
    }
}

/// The pure gate. `stored_exact_key_entries` counts every stored copy of the
/// primary key before deletion; `oversize_deleted_count` counts the ones
/// this run actually removed; `deletion_blocked` marks a failed delete that
/// leaves an oversize entry colliding with the key.
pub fn save_decision(
    directory_bytes: u64,
    stored_exact_key_entries: usize,
    oversize_deleted_count: usize,
    deletion_blocked: bool,
) -> SccacheSaveDecision {
    if deletion_blocked {
        return SccacheSaveDecision::ReplaceBlocked;
    }
    if directory_bytes > MAX_DIRECTORY_BYTES {
        return SccacheSaveDecision::DirectoryOverBudget;
    }
    if stored_exact_key_entries > 0 && oversize_deleted_count == 0 {
        return SccacheSaveDecision::FreshEntryWithinBudget;
    }
    if oversize_deleted_count > 0 {
        SccacheSaveDecision::ReplaceOversizeEntry
    } else {
        SccacheSaveDecision::NewGeneration
    }
}

/// Apparent-size accounting, not allocated blocks: the GitHub cache entry is
/// billed by archive bytes, so block-rounding slack would bias the decision.
pub fn directory_size_bytes(directory_path: &Path) -> u64 {
    let mut total_bytes = 0u64;
    let mut stack = vec![directory_path.to_path_buf()];
    while let Some(current_path) = stack.pop() {
        let Ok(directory_entries) = std::fs::read_dir(&current_path) else {
            continue;
        };
        for entry in directory_entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if let Ok(metadata) = entry.metadata() {
                total_bytes += metadata.len();
            }
        }
    }
    total_bytes
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StoredEntry {
    id: u64,
    ref_name: String,
    size_bytes: u64,
}

fn stored_entries_for_primary_key(
    cache_list_text: &str,
    primary_cache_key: &str,
) -> Vec<StoredEntry> {
    cache_list_text
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 5 || fields[1] != primary_cache_key {
                return None;
            }
            Some(StoredEntry {
                id: fields[0].parse().ok()?,
                ref_name: fields[2].to_owned(),
                size_bytes: fields[4].parse().ok()?,
            })
        })
        .collect()
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let started_at = std::time::Instant::now();
    let [primary_cache_key] = arguments else {
        return Err("usage: cargo xtask save-sccache <primary-cache-key>".to_owned());
    };
    let github_repository = require_environment_value("GITHUB_REPOSITORY")?;
    let sccache_directory = require_environment_value("SCCACHE_DIR")?;
    eprintln!(
        "[sccache-save] status=start repository={github_repository} max_directory_bytes={MAX_DIRECTORY_BYTES} max_entry_bytes={MAX_ENTRY_BYTES}"
    );

    let directory_bytes = directory_size_bytes(Path::new(&sccache_directory));
    eprintln!("[sccache-save] directory_bytes={directory_bytes}");

    let cache_list_text = match gh_output(&[
        "api",
        "--paginate",
        &format!("repos/{github_repository}/actions/caches"),
        "--jq",
        ".actions_caches[] | [.id, .key, .ref, .created_at, .size_in_bytes] | @tsv",
    ]) {
        Ok(cache_list_text) => cache_list_text,
        Err(_) => {
            // Read-only fork tokens cannot list caches; the save is
            // default-branch-only anyway, so a degraded listing skips the
            // save instead of failing the job.
            return finish_with_decision(
                SccacheSaveDecision::CacheListUnavailable,
                directory_bytes,
                started_at,
            );
        }
    };

    let stored_entries = stored_entries_for_primary_key(&cache_list_text, primary_cache_key);
    eprintln!(
        "[sccache-save] stored_exact_key_entries={}",
        stored_entries.len()
    );

    let mut oversize_deleted_count = 0usize;
    let mut deletion_blocked = false;
    for stored_entry in &stored_entries {
        if stored_entry.size_bytes <= MAX_ENTRY_BYTES {
            continue;
        }
        let delete_status = Command::new("gh")
            .args([
                "cache",
                "delete",
                &stored_entry.id.to_string(),
                "-R",
                &github_repository,
            ])
            .status()
            .map_err(|error| format!("gh cache delete failed to start: {error}"))?;
        if delete_status.success() {
            oversize_deleted_count += 1;
            eprintln!(
                "[sccache-save] action=deleted-oversize cache_id={} ref={} size_bytes={}",
                stored_entry.id, stored_entry.ref_name, stored_entry.size_bytes
            );
        } else {
            // A blocked deletion leaves the oversize entry in place; saving
            // under the same key would collide with it, so the save is
            // skipped and the prune pass owns the retry on a later run.
            deletion_blocked = true;
            eprintln!(
                "[sccache-save] action=delete-failed cache_id={}",
                stored_entry.id
            );
        }
    }

    let decision = if deletion_blocked {
        SccacheSaveDecision::ReplaceBlocked
    } else {
        save_decision(
            directory_bytes,
            stored_entries.len(),
            oversize_deleted_count,
            false,
        )
    };
    finish_with_decision(decision, directory_bytes, started_at)
}

fn finish_with_decision(
    decision: SccacheSaveDecision,
    directory_bytes: u64,
    started_at: std::time::Instant,
) -> Result<(), String> {
    emit_output(
        "should_save",
        if decision.should_save() {
            "true"
        } else {
            "false"
        },
    );
    emit_output("decision", decision.as_decision_text());
    emit_output("directory_bytes", &directory_bytes.to_string());
    eprintln!(
        "[sccache-save] status=complete decision={} directory_bytes={directory_bytes} elapsed_seconds={}",
        decision.as_decision_text(),
        started_at.elapsed().as_secs()
    );
    Ok(())
}

fn emit_output(output_name: &str, output_value: &str) {
    if let Ok(github_output_path) = std::env::var("GITHUB_OUTPUT") {
        use std::io::Write;
        if let Ok(mut github_output) = std::fs::OpenOptions::new()
            .append(true)
            .open(&github_output_path)
        {
            let _ = writeln!(github_output, "{output_name}={output_value}");
        }
    }
    eprintln!("[sccache-save] output={output_name} value={output_value}");
}

fn require_environment_value(variable_name: &str) -> Result<String, String> {
    std::env::var(variable_name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{variable_name} is required"))
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
