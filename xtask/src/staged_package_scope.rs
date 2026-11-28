//! Which workspace packages a commit's staged files can change the clippy
//! verdict of, so the pre-commit clippy pass lints those packages instead of
//! the whole workspace.
//!
//! Every lint the pass enforces (the 80-line function budget first) judges a
//! package by its own source. A commit that edits one crate cannot change
//! the verdict on a crate that only depends on it while that crate's source
//! is untouched, yet `cargo clippy --workspace` re-lints every dependent: a
//! commit to `model-primitives` re-linted the daemon, the apps and the
//! acceptance tests, 36 of a 53-second hook (measured 2026-11-27). Whether
//! the dependents still compile is CI's question, which builds everything.
//!
//! Anything that can change how every package is linted keeps the whole
//! workspace pass: a manifest, the lock file, `clippy.toml`, the toolchain
//! file or `.cargo/` configuration, and a Rust file that belongs to no
//! workspace package.

use std::path::Path;
use std::process::Command;

/// What the clippy pass must lint for a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ClippyScope {
    /// Every workspace package.
    Workspace,
    /// Only these packages, by name, sorted and without repeats.
    Packages(Vec<String>),
}

/// A workspace package: its name and its directory relative to the
/// repository root, `/`-separated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkspaceMember {
    pub(crate) name: String,
    pub(crate) directory: String,
}

/// Files whose change can alter how every package is linted.
const WORKSPACE_WIDE_FILE_NAMES: [&str; 5] = [
    "Cargo.toml",
    "Cargo.lock",
    "clippy.toml",
    "rust-toolchain",
    "rust-toolchain.toml",
];

/// The scope for `staged_paths` (repository-relative, `/`-separated). No
/// staged Rust file at all keeps the whole workspace, so a run outside a
/// commit lints everything as it always did.
pub(crate) fn clippy_scope(staged_paths: &[String], members: &[WorkspaceMember]) -> ClippyScope {
    let mut package_names = Vec::new();
    for staged_path in staged_paths {
        if changes_every_package(staged_path) {
            return ClippyScope::Workspace;
        }
        if !staged_path.ends_with(".rs") {
            continue;
        }
        match owning_member(staged_path, members) {
            Some(member) => package_names.push(member.name.clone()),
            None => return ClippyScope::Workspace,
        }
    }
    if package_names.is_empty() {
        return ClippyScope::Workspace;
    }
    package_names.sort();
    package_names.dedup();
    ClippyScope::Packages(package_names)
}

fn changes_every_package(staged_path: &str) -> bool {
    let file_name = staged_path.rsplit('/').next().unwrap_or(staged_path);
    WORKSPACE_WIDE_FILE_NAMES.contains(&file_name) || staged_path.starts_with(".cargo/")
}

/// The member whose directory holds `staged_path` most closely: a crate
/// nested inside another crate's folder owns its own files.
fn owning_member<'members>(
    staged_path: &str,
    members: &'members [WorkspaceMember],
) -> Option<&'members WorkspaceMember> {
    let mut owner: Option<&WorkspaceMember> = None;
    for member in members {
        let is_inside = member.directory.is_empty()
            || staged_path
                .strip_prefix(member.directory.as_str())
                .is_some_and(|rest| rest.starts_with('/'));
        let is_closer =
            owner.is_none_or(|current| member.directory.len() > current.directory.len());
        if is_inside && is_closer {
            owner = Some(member);
        }
    }
    owner
}

/// The paths the index holds changes for: added, copied, modified, renamed
/// or deleted, since a deleted file changes its crate too.
pub(crate) fn staged_paths(repository_root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args(["diff", "--cached", "--name-only", "--diff-filter=ACMRD"])
        .current_dir(repository_root)
        .output()
        .map_err(|spawn_error| format!("git could not start: {spawn_error}"))?;
    if !output.status.success() {
        return Err("git could not list the staged files".to_owned());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// The workspace's packages from `cargo metadata`, directories relative to
/// `repository_root`.
pub(crate) fn workspace_members(repository_root: &Path) -> Result<Vec<WorkspaceMember>, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(repository_root)
        .output()
        .map_err(|spawn_error| format!("cargo metadata could not start: {spawn_error}"))?;
    if !output.status.success() {
        return Err("cargo metadata failed".to_owned());
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|parse_error| format!("cargo metadata is not JSON: {parse_error}"))?;
    let workspace_root = metadata["workspace_root"].as_str().unwrap_or_default();
    let mut members = Vec::new();
    for package in metadata["packages"].as_array().into_iter().flatten() {
        let (Some(name), Some(manifest_path)) =
            (package["name"].as_str(), package["manifest_path"].as_str())
        else {
            continue;
        };
        members.push(WorkspaceMember {
            name: name.to_owned(),
            directory: member_directory(workspace_root, manifest_path),
        });
    }
    Ok(members)
}

/// A manifest's folder relative to the workspace root: `""` for the root
/// package itself.
fn member_directory(workspace_root: &str, manifest_path: &str) -> String {
    let manifest_folder = manifest_path
        .strip_suffix("/Cargo.toml")
        .unwrap_or(manifest_path);
    manifest_folder
        .strip_prefix(workspace_root)
        .unwrap_or(manifest_folder)
        .trim_start_matches('/')
        .to_owned()
}
