//! Who a release is: the tag, the version it must match, the commit it must
//! sit on, the repository it publishes to, and where its artifacts live.
//! Every check runs before any build, signing, or GitHub call, so a wrong
//! tag or a dirty tree costs seconds instead of a notarization round trip.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::app_bundle::{daemon_package_identity, github_pages_feed_url};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PublicationRequest {
    pub(crate) release_tag: String,
    pub(crate) notes_file: Option<PathBuf>,
    pub(crate) output_directory: Option<PathBuf>,
    pub(crate) is_publish: bool,
    pub(crate) signing_identity: Option<String>,
    pub(crate) team_identifier: Option<String>,
    pub(crate) notary_profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReleaseIdentity {
    pub(crate) release_tag: String,
    pub(crate) release_version: String,
    pub(crate) release_commit: String,
    /// `owner/name` of the GitHub repository the release publishes to.
    pub(crate) repository_slug: String,
    pub(crate) update_feed_url: String,
    pub(crate) output_directory: PathBuf,
}

/// `vMAJOR.MINOR.PATCH` with numeric components; returns `MAJOR.MINOR.PATCH`.
pub(crate) fn parse_release_version(release_tag: &str) -> Result<String, String> {
    let Some(release_version) = release_tag.strip_prefix('v') else {
        return Err("release tag must use vMAJOR.MINOR.PATCH".to_owned());
    };
    let version_components: Vec<&str> = release_version.split('.').collect();
    if version_components.len() != 3 {
        return Err("release tag must use vMAJOR.MINOR.PATCH".to_owned());
    }
    for version_component in version_components {
        if version_component.is_empty() || !version_component.bytes().all(is_ascii_digit_byte) {
            return Err("release tag must use numeric MAJOR.MINOR.PATCH components".to_owned());
        }
    }
    Ok(release_version.to_owned())
}

fn is_ascii_digit_byte(candidate_byte: u8) -> bool {
    candidate_byte.is_ascii_digit()
}

/// `owner/name` from the workspace's `https://github.com/owner/name[.git]`.
pub(crate) fn repository_slug(repository_url: &str) -> Result<String, String> {
    github_pages_feed_url(repository_url)?;
    let Some(repository_path) = repository_url.strip_prefix("https://github.com/") else {
        return Err("workspace repository must be an HTTPS GitHub URL".to_owned());
    };
    Ok(repository_path
        .strip_suffix(".git")
        .unwrap_or(repository_path)
        .to_owned())
}

pub(crate) fn resolve_release_identity(
    repository_root: &Path,
    publication_request: &PublicationRequest,
) -> Result<ReleaseIdentity, String> {
    let release_version = parse_release_version(&publication_request.release_tag)?;
    if !publication_request.is_publish {
        require_release_notes(publication_request.notes_file.as_deref())?;
    }
    let cargo_metadata = command_stdout(
        repository_root,
        "cargo",
        &["metadata", "--no-deps", "--format-version", "1"],
    )?;
    let (workspace_version, repository_url) = daemon_package_identity(&cargo_metadata)?;
    if release_version != workspace_version {
        return Err(format!(
            "release tag {} does not match workspace version {workspace_version}",
            publication_request.release_tag
        ));
    }
    if !command_stdout(
        repository_root,
        "git",
        &["status", "--porcelain", "--untracked-files=normal"],
    )?
    .is_empty()
    {
        return Err("release preparation requires a clean worktree".to_owned());
    }
    let release_commit = command_stdout(repository_root, "git", &["rev-parse", "HEAD"])?;
    let tagged_commit = command_stdout(
        repository_root,
        "git",
        &[
            "rev-parse",
            &format!("{}^{{commit}}", publication_request.release_tag),
        ],
    )?;
    if tagged_commit != release_commit {
        return Err(format!(
            "release tag {} must identify the current commit",
            publication_request.release_tag
        ));
    }
    let output_directory = match &publication_request.output_directory {
        Some(output_directory) => output_directory.clone(),
        None => repository_root
            .join("target/releases")
            .join(&publication_request.release_tag),
    };
    check_output_directory(&output_directory, publication_request.is_publish)?;
    Ok(ReleaseIdentity {
        release_tag: publication_request.release_tag.clone(),
        release_version,
        release_commit,
        repository_slug: repository_slug(&repository_url)?,
        update_feed_url: github_pages_feed_url(&repository_url)?,
        output_directory,
    })
}

fn require_release_notes(notes_file: Option<&Path>) -> Result<(), String> {
    let Some(notes_file) = notes_file else {
        return Err("--notes-file is required while preparing a release".to_owned());
    };
    match std::fs::metadata(notes_file) {
        Ok(notes_metadata) if notes_metadata.is_file() && notes_metadata.len() > 0 => Ok(()),
        _ => Err(format!(
            "release notes are unavailable or empty: {}",
            notes_file.display()
        )),
    }
}

/// Preparing never overwrites a prepared directory; publishing only ever
/// uploads one that already exists, so the reviewed bytes are the bytes
/// that ship.
pub(crate) fn check_output_directory(
    output_directory: &Path,
    is_publish: bool,
) -> Result<(), String> {
    if is_publish && !output_directory.is_dir() {
        return Err(format!(
            "prepared release directory is unavailable: {}",
            output_directory.display()
        ));
    }
    if !is_publish && output_directory.exists() {
        return Err(format!(
            "release output already exists: {}",
            output_directory.display()
        ));
    }
    Ok(())
}

pub(crate) fn command_stdout(
    working_directory: &Path,
    program: &str,
    program_arguments: &[&str],
) -> Result<String, String> {
    match Command::new(program)
        .args(program_arguments)
        .current_dir(working_directory)
        .output()
    {
        Ok(program_output) if program_output.status.success() => {
            Ok(String::from_utf8_lossy(&program_output.stdout)
                .trim()
                .to_owned())
        }
        Ok(program_output) => Err(format!(
            "{program} {} failed: {}",
            program_arguments.join(" "),
            String::from_utf8_lossy(&program_output.stderr).trim()
        )),
        Err(spawn_error) => Err(format!("{program} could not start: {spawn_error}")),
    }
}
