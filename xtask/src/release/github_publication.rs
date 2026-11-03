//! Publishes a prepared release: re-hash the reviewed artifacts against the
//! manifest, prove the remote tag resolves to the prepared commit, create or
//! resume a draft GitHub Release, repair its asset if needed, publish it,
//! and stage the signed appcast for the GitHub Pages deployment.
//!
//! The run is resumable: every decision reads GitHub's current state first,
//! so a run interrupted after the draft or the upload continues instead of
//! duplicating anything. All decisions are pure functions over `gh` JSON.

use std::path::Path;
use std::process::Command;

use super::release_identity::{ReleaseIdentity, command_stdout};
use super::release_manifest::{MANIFEST_FILE_NAME, ReleaseManifest, verify_artifact};
use crate::step_log::start_step;

/// Annotated tags can point at tags; a deeper chain than this is treated as
/// hostile rather than followed forever.
const MAXIMUM_TAG_NESTING: usize = 8;

/// What to do with the draft release's asset before publishing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DraftAssetAction {
    Upload,
    ReplaceMismatchedUpload,
    KeepVerifiedUpload,
}

pub(crate) fn load_prepared_release(
    release_identity: &ReleaseIdentity,
) -> Result<ReleaseManifest, String> {
    let manifest_text =
        std::fs::read_to_string(release_identity.output_directory.join(MANIFEST_FILE_NAME))
            .unwrap_or_default();
    let release_manifest = ReleaseManifest::from_json_text(&manifest_text)?;
    if release_manifest.release_tag != release_identity.release_tag {
        return Err("prepared release tag does not match".to_owned());
    }
    if release_manifest.release_commit != release_identity.release_commit {
        return Err("prepared release commit does not match HEAD".to_owned());
    }
    if release_manifest.release_version != release_identity.release_version {
        return Err("prepared release version does not match".to_owned());
    }
    let output_directory = &release_identity.output_directory;
    verify_artifact(
        &output_directory.join(&release_manifest.disk_image.artifact_name),
        &release_manifest.disk_image,
        "DMG",
    )?;
    verify_artifact(
        &output_directory.join(&release_manifest.release_notes.artifact_name),
        &release_manifest.release_notes,
        "release-notes",
    )?;
    verify_artifact(
        &output_directory.join(&release_manifest.appcast.artifact_name),
        &release_manifest.appcast,
        "appcast",
    )?;
    Ok(release_manifest)
}

pub(crate) fn publish_release(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    release_manifest: &ReleaseManifest,
) -> Result<(), String> {
    let tag_step = start_step("verify-remote-release-tag");
    verify_remote_release_tag(repository_root, release_identity)?;
    tag_step.finish("success");
    let draft_step = start_step("create-draft-github-release");
    create_draft_release_if_missing(repository_root, release_identity, release_manifest)?;
    draft_step.finish("success");
    let publish_step = start_step("verify-and-publish-github-release");
    publish_verified_draft(repository_root, release_identity, release_manifest)?;
    publish_step.finish("success");
    let appcast_step = start_step("stage-signed-pages-appcast");
    stage_pages_appcast(
        repository_root,
        &release_identity
            .output_directory
            .join(&release_manifest.appcast.artifact_name),
    )?;
    appcast_step.finish("success");
    Ok(())
}

fn verify_remote_release_tag(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
) -> Result<(), String> {
    let mut reference_document = gh_json(
        repository_root,
        &[
            "api",
            &format!(
                "repos/{}/git/ref/tags/{}",
                release_identity.repository_slug, release_identity.release_tag
            ),
        ],
    )?;
    for _ in 0..=MAXIMUM_TAG_NESTING {
        match next_tag_object(&reference_document)? {
            TagObject::Commit(commit_sha) if commit_sha == release_identity.release_commit => {
                return Ok(());
            }
            TagObject::Commit(_) => {
                return Err("remote release tag does not resolve to the prepared commit".to_owned());
            }
            TagObject::AnnotatedTag(tag_sha) => {
                reference_document = gh_json(
                    repository_root,
                    &[
                        "api",
                        &format!(
                            "repos/{}/git/tags/{tag_sha}",
                            release_identity.repository_slug
                        ),
                    ],
                )?;
            }
        }
    }
    Err("remote release tag nesting exceeds the safety limit".to_owned())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TagObject {
    Commit(String),
    AnnotatedTag(String),
}

pub(crate) fn next_tag_object(reference_document: &serde_json::Value) -> Result<TagObject, String> {
    let tag_target = &reference_document["object"];
    match (tag_target["type"].as_str(), tag_target["sha"].as_str()) {
        (Some("commit"), Some(commit_sha)) => Ok(TagObject::Commit(commit_sha.to_owned())),
        (Some("tag"), Some(tag_sha)) => Ok(TagObject::AnnotatedTag(tag_sha.to_owned())),
        _ => Err("remote release tag does not resolve to the prepared commit".to_owned()),
    }
}

fn create_draft_release_if_missing(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    release_manifest: &ReleaseManifest,
) -> Result<(), String> {
    let view_status = Command::new("gh")
        .args([
            "release",
            "view",
            &release_identity.release_tag,
            "--repo",
            &release_identity.repository_slug,
        ])
        .current_dir(repository_root)
        .output();
    if matches!(view_status, Ok(view_output) if view_output.status.success()) {
        println!(
            "Resuming matching GitHub Release {}.",
            release_identity.release_tag
        );
        return Ok(());
    }
    let output_directory = &release_identity.output_directory;
    run_gh(
        repository_root,
        &[
            "release",
            "create",
            &release_identity.release_tag,
            &output_directory
                .join(&release_manifest.disk_image.artifact_name)
                .display()
                .to_string(),
            "--repo",
            &release_identity.repository_slug,
            "--verify-tag",
            "--draft",
            "--title",
            &format!("Astronomical {}", release_identity.release_version),
            "--notes-file",
            &output_directory
                .join(&release_manifest.release_notes.artifact_name)
                .display()
                .to_string(),
        ],
    )
}

fn publish_verified_draft(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    release_manifest: &ReleaseManifest,
) -> Result<(), String> {
    let notes_text = std::fs::read_to_string(
        release_identity
            .output_directory
            .join(&release_manifest.release_notes.artifact_name),
    )
    .unwrap_or_default();
    let mut release_state = load_release_state(repository_root, release_identity, &notes_text)?;
    if release_state["isDraft"].as_bool() == Some(true) {
        if release_state["isImmutable"].as_bool() != Some(false) {
            return Err("draft GitHub Release is unexpectedly immutable".to_owned());
        }
        repair_draft_asset(
            repository_root,
            release_identity,
            release_manifest,
            &release_state,
        )?;
        release_state = load_release_state(repository_root, release_identity, &notes_text)?;
        if let Some(asset_problem) =
            uploaded_asset_problems(&release_state, release_manifest).first()
        {
            return Err(asset_problem.clone());
        }
        run_gh(
            repository_root,
            &[
                "release",
                "edit",
                &release_identity.release_tag,
                "--repo",
                &release_identity.repository_slug,
                "--draft=false",
            ],
        )?;
        release_state = load_release_state(repository_root, release_identity, &notes_text)?;
    }
    if release_state["isDraft"].as_bool() != Some(false) {
        return Err("GitHub Release is still a draft".to_owned());
    }
    match uploaded_asset_problems(&release_state, release_manifest).first() {
        Some(asset_problem) => Err(asset_problem.clone()),
        None => Ok(()),
    }
}

fn repair_draft_asset(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    release_manifest: &ReleaseManifest,
    release_state: &serde_json::Value,
) -> Result<(), String> {
    let disk_image = release_identity
        .output_directory
        .join(&release_manifest.disk_image.artifact_name)
        .display()
        .to_string();
    let upload_arguments = [
        "release",
        "upload",
        &release_identity.release_tag,
        &disk_image,
        "--repo",
        &release_identity.repository_slug,
    ];
    match draft_asset_action(release_state, release_manifest)? {
        DraftAssetAction::Upload => run_gh(repository_root, &upload_arguments),
        DraftAssetAction::ReplaceMismatchedUpload => {
            let mut clobber_arguments = upload_arguments.to_vec();
            clobber_arguments.push("--clobber");
            run_gh(repository_root, &clobber_arguments)
        }
        DraftAssetAction::KeepVerifiedUpload => Ok(()),
    }
}

/// A draft holds either nothing (upload), exactly the expected asset with
/// the right bytes (keep), or exactly the expected asset with wrong or
/// incomplete bytes (replace). Anything else is refused.
pub(crate) fn draft_asset_action(
    release_state: &serde_json::Value,
    release_manifest: &ReleaseManifest,
) -> Result<DraftAssetAction, String> {
    let release_assets = release_state["assets"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if release_assets.is_empty() {
        return Ok(DraftAssetAction::Upload);
    }
    if release_assets.len() != 1
        || release_assets[0]["name"].as_str()
            != Some(release_manifest.disk_image.artifact_name.as_str())
    {
        return Err("draft GitHub Release contains unexpected assets".to_owned());
    }
    if asset_matches_manifest(&release_assets[0], release_manifest) {
        Ok(DraftAssetAction::KeepVerifiedUpload)
    } else {
        Ok(DraftAssetAction::ReplaceMismatchedUpload)
    }
}

/// The uploaded asset must be the only asset, by name, and carry the exact
/// digest and size of the reviewed DMG.
pub(crate) fn uploaded_asset_problems(
    release_state: &serde_json::Value,
    release_manifest: &ReleaseManifest,
) -> Vec<String> {
    let release_assets = release_state["assets"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let expected_name = release_manifest.disk_image.artifact_name.as_str();
    let mut matching_assets = Vec::new();
    for release_asset in &release_assets {
        if release_asset["name"].as_str() == Some(expected_name) {
            matching_assets.push(release_asset.clone());
        }
    }
    if matching_assets.len() != 1 {
        return vec!["GitHub Release must contain exactly one expected DMG".to_owned()];
    }
    let mut asset_problems = Vec::new();
    if release_assets.len() != 1 {
        asset_problems.push("GitHub Release contains unexpected assets".to_owned());
    }
    let uploaded_asset = &matching_assets[0];
    if uploaded_asset["digest"].as_str()
        != Some(format!("sha256:{}", release_manifest.disk_image.sha256).as_str())
    {
        asset_problems.push("GitHub Release asset digest does not match prepared DMG".to_owned());
    }
    if uploaded_asset["size"].as_u64() != Some(release_manifest.disk_image.size_bytes) {
        asset_problems.push("GitHub Release asset size does not match prepared DMG".to_owned());
    }
    if uploaded_asset["state"].as_str() != Some("uploaded") {
        asset_problems.push("GitHub Release asset upload is incomplete".to_owned());
    }
    asset_problems
}

fn asset_matches_manifest(
    release_asset: &serde_json::Value,
    release_manifest: &ReleaseManifest,
) -> bool {
    release_asset["digest"].as_str()
        == Some(format!("sha256:{}", release_manifest.disk_image.sha256).as_str())
        && release_asset["size"].as_u64() == Some(release_manifest.disk_image.size_bytes)
        && release_asset["state"].as_str() == Some("uploaded")
}

fn load_release_state(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    notes_text: &str,
) -> Result<serde_json::Value, String> {
    let release_state = gh_json(
        repository_root,
        &[
            "release",
            "view",
            &release_identity.release_tag,
            "--repo",
            &release_identity.repository_slug,
            "--json",
            "assets,body,isDraft,isImmutable,isPrerelease,name,tagName",
        ],
    )?;
    match release_state_problems(&release_state, release_identity, notes_text).first() {
        Some(state_problem) => Err(state_problem.clone()),
        None => Ok(release_state),
    }
}

/// An existing release must be this release: same tag, title, and notes,
/// and never a prerelease.
pub(crate) fn release_state_problems(
    release_state: &serde_json::Value,
    release_identity: &ReleaseIdentity,
    notes_text: &str,
) -> Vec<String> {
    let mut state_problems = Vec::new();
    if release_state["tagName"].as_str() != Some(release_identity.release_tag.as_str()) {
        state_problems.push("existing GitHub Release has a conflicting tag".to_owned());
    }
    if release_state["name"].as_str()
        != Some(format!("Astronomical {}", release_identity.release_version).as_str())
    {
        state_problems.push("existing GitHub Release has a conflicting title".to_owned());
    }
    if release_state["body"].as_str().map(str::trim_end) != Some(notes_text.trim_end()) {
        state_problems.push("existing GitHub Release has conflicting release notes".to_owned());
    }
    if release_state["isPrerelease"].as_bool() != Some(false) {
        state_problems.push("Stable publication cannot use a prerelease GitHub Release".to_owned());
    }
    state_problems
}

/// The appcast is staged with ordinary read permissions for review and the
/// next main-branch Pages deployment; it is never committed automatically.
fn stage_pages_appcast(repository_root: &Path, appcast: &Path) -> Result<(), String> {
    let pages_appcast = repository_root.join("site/appcast.xml");
    if let Err(copy_error) = std::fs::copy(appcast, &pages_appcast) {
        return Err(format!("could not stage site/appcast.xml: {copy_error}"));
    }
    let readable_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o644);
    match std::fs::set_permissions(&pages_appcast, readable_permissions) {
        Ok(()) => Ok(()),
        Err(permission_error) => Err(format!(
            "could not set site/appcast.xml permissions: {permission_error}"
        )),
    }
}

fn gh_json(repository_root: &Path, gh_arguments: &[&str]) -> Result<serde_json::Value, String> {
    let gh_output = command_stdout(repository_root, "gh", gh_arguments)?;
    match serde_json::from_str(&gh_output) {
        Ok(gh_document) => Ok(gh_document),
        Err(parse_error) => Err(format!(
            "gh {} did not return JSON: {parse_error}",
            gh_arguments.join(" ")
        )),
    }
}

fn run_gh(repository_root: &Path, gh_arguments: &[&str]) -> Result<(), String> {
    match Command::new("gh")
        .args(gh_arguments)
        .current_dir(repository_root)
        .status()
    {
        Ok(exit_status) if exit_status.success() => Ok(()),
        _ => Err(format!("gh {} failed", gh_arguments.join(" "))),
    }
}
