//! What CI already did, read back from the artifacts its runs left.
//!
//! A merge's tree matches what its pull request tested whenever the base branch
//! did not move in between, and every CI run on a merge otherwise repeats all of
//! that work for about 12 job-minutes. A passing pull request leaves an artifact
//! named after its tree, and a default-branch run that really ran the macOS lanes
//! leaves one saying which commit it was, because those runs are what publish the
//! caches later pull requests restore. Every doubt answers "unknown", which only
//! costs the full run the push would have had.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The artifact name is this plus the tree id. The workflow writes the same text.
pub const VERIFIED_TREE_ARTIFACT_PREFIX: &str = "verified-tree-";
/// The workflow writes the same text.
pub const DEFAULT_BRANCH_RUN_ARTIFACT: &str = "default-branch-macos-run";
/// A hung lookup must not fail the classifier and with it the required check, and a
/// lookup usually takes under a second.
const GITHUB_LOOKUP_LIMIT: Duration = Duration::from_secs(20);
const GITHUB_LOOKUP_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Whether a listing of artifacts holds a live record of this tree from a branch of
/// this repository. Runs from a fork are refused, because a fork's pull request runs
/// its own copy of the workflow, so its record proves nothing about what ran.
pub fn listing_proves_verified_tree(artifact_listing: &str, tree_id: &str) -> bool {
    let Ok(listing) = serde_json::from_str::<serde_json::Value>(artifact_listing) else {
        return false;
    };
    let Some(artifacts) = listing["artifacts"].as_array() else {
        return false;
    };
    let expected_name = format!("{VERIFIED_TREE_ARTIFACT_PREFIX}{tree_id}");
    for artifact in artifacts {
        if artifact["name"].as_str() == Some(expected_name.as_str())
            && artifact["expired"].as_bool() == Some(false)
            && is_from_this_repository(&artifact["workflow_run"])
        {
            return true;
        }
    }
    false
}

fn is_from_this_repository(workflow_run: &serde_json::Value) -> bool {
    match (
        workflow_run["repository_id"].as_u64(),
        workflow_run["head_repository_id"].as_u64(),
    ) {
        (Some(repository_id), Some(head_repository_id)) => repository_id == head_repository_id,
        _ => false,
    }
}

/// The commit of the newest live record of a default-branch macOS run. Records are
/// compared by creation time because the listing's order is not promised.
pub fn newest_default_branch_run_commit(artifact_listing: &str) -> Option<String> {
    let listing = serde_json::from_str::<serde_json::Value>(artifact_listing).ok()?;
    let mut newest_commit = None;
    let mut newest_created_at = "";
    for artifact in listing["artifacts"].as_array()? {
        if artifact["name"].as_str() != Some(DEFAULT_BRANCH_RUN_ARTIFACT)
            || artifact["expired"].as_bool() != Some(false)
            || !is_from_this_repository(&artifact["workflow_run"])
        {
            continue;
        }
        let (Some(created_at), Some(commit)) = (
            artifact["created_at"].as_str(),
            artifact["workflow_run"]["head_sha"].as_str(),
        ) else {
            continue;
        };
        // RFC 3339 times in one zone sort as text, which is how GitHub writes them.
        if created_at > newest_created_at {
            newest_created_at = created_at;
            newest_commit = Some(commit.to_owned());
        }
    }
    newest_commit
}

/// The git and GitHub edge. Reports what it found, because a run that skipped its
/// macOS lanes should say why in its log.
pub fn is_tree_already_verified(repository_root: &Path, head_sha: &str) -> bool {
    let Some(tree_id) = tree_id_of(repository_root, head_sha) else {
        eprintln!("[ci-change-scope] verified_tree=unknown reason=tree-unreadable");
        return false;
    };
    let artifact_name = format!("{VERIFIED_TREE_ARTIFACT_PREFIX}{tree_id}");
    let is_verified = match artifact_listing_named(&artifact_name) {
        Some(artifact_listing) => listing_proves_verified_tree(&artifact_listing, &tree_id),
        None => false,
    };
    eprintln!("[ci-change-scope] verified_tree={is_verified} tree={tree_id}");
    is_verified
}

/// The commit whose default-branch run last published caches, while that is recent
/// enough for its record to be alive.
pub fn last_default_branch_run_commit() -> Option<String> {
    let last_run_commit =
        newest_default_branch_run_commit(&artifact_listing_named(DEFAULT_BRANCH_RUN_ARTIFACT)?);
    eprintln!("[ci-change-scope] last_default_branch_run={last_run_commit:?}");
    last_run_commit
}

/// Output is read only after the process exits, which is safe because ten artifacts
/// fit well inside a pipe's buffer.
fn artifact_listing_named(artifact_name: &str) -> Option<String> {
    let repository = std::env::var("GITHUB_REPOSITORY").ok()?;
    let mut gh_child = Command::new("gh")
        .arg("api")
        .arg(format!(
            "repos/{repository}/actions/artifacts?name={artifact_name}&per_page=10"
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started_at = Instant::now();
    loop {
        match gh_child.try_wait().ok()? {
            Some(_) => break,
            None if started_at.elapsed() >= GITHUB_LOOKUP_LIMIT => {
                eprintln!("[ci-change-scope] github_lookup=timed-out");
                let _kill_result = gh_child.kill();
                let _wait_result = gh_child.wait();
                return None;
            }
            None => std::thread::sleep(GITHUB_LOOKUP_POLL_INTERVAL),
        }
    }
    let gh_output = gh_child.wait_with_output().ok()?;
    if !gh_output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&gh_output.stdout).into_owned())
}

fn tree_id_of(repository_root: &Path, commit_sha: &str) -> Option<String> {
    let git_output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(["rev-parse", &format!("{commit_sha}^{{tree}}")])
        .output()
        .ok()?;
    if !git_output.status.success() {
        return None;
    }
    let tree_id = String::from_utf8_lossy(&git_output.stdout)
        .trim()
        .to_owned();
    if tree_id.is_empty() {
        None
    } else {
        Some(tree_id)
    }
}

#[cfg(test)]
#[path = "verified_tree_tests.rs"]
mod tests;
