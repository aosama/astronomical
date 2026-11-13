//! Prepares a signed Stable release locally without touching GitHub:
//! package, prove the Developer ID signature, build the DMG, notarize and
//! staple it, prove its install journey, sign the Sparkle appcast, and write
//! the manifest. Everything is assembled in a staging directory under
//! `target/` and renamed into the output directory only when complete, so a
//! failed preparation never leaves something that looks reviewable.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};

use super::disk_image_creation::{DiskImageRequest, create_disk_image};
use super::disk_image_validation::validate_disk_image;
use super::distribution_signature::validate_distribution_app;
use super::notarization::{
    DEFAULT_NOTARIZATION_TIMEOUT_SECONDS, NotarizationRequest, notarize_disk_image,
};
use super::release_identity::ReleaseIdentity;
use super::release_manifest::{MANIFEST_FILE_NAME, ReleaseManifest, record_artifact};
use crate::step_log::start_step;

const APPCAST_TOOL_CANDIDATES: [&str; 2] = [
    "apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin/generate_appcast",
    "apps/astronomical-menu/.build/artifacts/sparkle/bin/generate_appcast",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DistributionCredentials {
    pub(crate) signing_identity: String,
    pub(crate) team_identifier: String,
    pub(crate) notary_profile: String,
}

/// The signing identity must be a Developer ID Application identity the
/// keychain actually holds, and the notary profile must be readable now;
/// both fail before any build starts.
pub(crate) fn validate_distribution_credentials(
    credentials: &DistributionCredentials,
) -> Result<(), String> {
    if !credentials
        .signing_identity
        .starts_with("Developer ID Application:")
    {
        return Err("configured identity must be a Developer ID Application identity".to_owned());
    }
    if credentials.team_identifier.len() < 7 {
        return Err("--team-id is required while preparing a release".to_owned());
    }
    let identity_listing = super::release_identity::command_stdout(
        Path::new("/"),
        "security",
        &["find-identity", "-v", "-p", "codesigning"],
    )?;
    if !identity_listing.contains(&format!("\"{}\"", credentials.signing_identity)) {
        return Err("configured Developer ID Application identity is unavailable".to_owned());
    }
    verify_notary_profile_access(&credentials.notary_profile)
}

/// Enough of Apple's reply to name the cause without flooding the log.
const MAXIMUM_NOTARY_REPLY_CHARACTERS: usize = 600;

/// Asks the notary service for the submission history, which needs both the
/// Keychain profile and Apple's consent to serve this account. The default
/// notarytool profile lives in the Data Protection Keychain, which a locked
/// screen makes unreadable mid-run, but Apple also refuses the request for
/// reasons that have nothing to do with this Mac, such as a program agreement
/// awaiting acceptance. The failure therefore carries Apple's own words: a
/// fixed "unlock this Mac" message once sent a release looking at the
/// Keychain while the cause sat on the developer account.
pub(crate) fn verify_notary_profile_access(notary_profile: &str) -> Result<(), String> {
    let history_attempt = Command::new("xcrun")
        .args([
            "notarytool",
            "history",
            "--keychain-profile",
            notary_profile,
        ])
        .output();
    match history_attempt {
        Ok(history_output) if history_output.status.success() => Ok(()),
        refused_or_unlaunched => Err(notary_check_failure(refused_or_unlaunched)),
    }
}

/// Words a failed profile check: notarytool's reply when it ran and was
/// refused, or the launch error when it never ran.
fn notary_check_failure(history_attempt: std::io::Result<std::process::Output>) -> String {
    match history_attempt {
        Ok(history_output) => notary_profile_access_problem(
            &String::from_utf8_lossy(&history_output.stderr),
            &String::from_utf8_lossy(&history_output.stdout),
        ),
        Err(launch_failure) => {
            format!("notarization profile check could not run xcrun notarytool: {launch_failure}")
        }
    }
}

/// Describes a refused profile check: Apple's reply, then the step that
/// usually clears it. notarytool writes its error to either stream depending
/// on the output format, so the first stream that says anything is quoted.
pub(crate) fn notary_profile_access_problem(
    notary_error_output: &str,
    notary_standard_output: &str,
) -> String {
    let notary_reply = if notary_error_output.trim().is_empty() {
        notary_standard_output.trim()
    } else {
        notary_error_output.trim()
    };
    if notary_reply.is_empty() {
        return "notarization profile check failed and notarytool gave no reason. Unlock this Mac and retry; if access still fails, verify the configured notarytool profile.".to_owned();
    }
    let bounded_notary_reply: String = notary_reply
        .chars()
        .take(MAXIMUM_NOTARY_REPLY_CHARACTERS)
        .collect();
    let remedy = if notary_reply.to_lowercase().contains("agreement") {
        "Sign in to the Apple Developer account, accept the pending agreement, and retry; Apple's services can take a few minutes to recognise it."
    } else {
        "If the reply names no cause, unlock this Mac and retry, then verify the configured notarytool profile."
    };
    format!(
        "notarization profile check failed. notarytool replied: {bounded_notary_reply} {remedy}"
    )
}

/// Keeps the Mac from idle-locking (and so locking the keychain) for as long
/// as this process lives; `-w` ties the assertion to this process id.
pub(crate) fn prevent_idle_lock() -> Result<Child, String> {
    match Command::new("caffeinate")
        .args(["-dimsu", "-w", &std::process::id().to_string()])
        .spawn()
    {
        Ok(caffeinate_process) => {
            println!(
                "{} step=prevent-idle-lock status=enabled",
                crate::disposable_target::local_timestamp()
            );
            Ok(caffeinate_process)
        }
        Err(spawn_error) => Err(format!("caffeinate could not start: {spawn_error}")),
    }
}

pub(crate) fn prepare_release(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    credentials: &DistributionCredentials,
    notes_file: &Path,
) -> Result<PathBuf, String> {
    let staging_directory = StagingDirectory::create(repository_root)?;
    let stable_bundle = build_stable_bundle(repository_root, release_identity, credentials)?;
    let disk_image = staging_directory.directory_path.join(format!(
        "Astronomical-{}-macOS-arm64.dmg",
        release_identity.release_version
    ));
    produce_notarized_disk_image(repository_root, &stable_bundle, &disk_image, credentials)?;
    let notes_copy = staging_directory.directory_path.join(format!(
        "Astronomical-{}-macOS-arm64.md",
        release_identity.release_version
    ));
    if let Err(copy_error) = std::fs::copy(notes_file, &notes_copy) {
        return Err(format!("could not copy the release notes: {copy_error}"));
    }
    let appcast = sign_appcast(
        repository_root,
        release_identity,
        &staging_directory.directory_path,
        &disk_image,
    )?;
    let release_manifest = ReleaseManifest {
        release_tag: release_identity.release_tag.clone(),
        release_commit: release_identity.release_commit.clone(),
        release_version: release_identity.release_version.clone(),
        disk_image: record_artifact(&disk_image)?,
        release_notes: record_artifact(&notes_copy)?,
        appcast: record_artifact(&appcast)?,
    };
    write_text(
        &staging_directory.directory_path.join(MANIFEST_FILE_NAME),
        &release_manifest.to_json_text(),
    )?;
    staging_directory.publish_as(&release_identity.output_directory)?;
    Ok(release_identity
        .output_directory
        .join(&release_manifest.disk_image.artifact_name))
}

fn build_stable_bundle(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    credentials: &DistributionCredentials,
) -> Result<PathBuf, String> {
    let build_step = start_step("build-stable-app");
    let package_arguments = [
        "package".to_owned(),
        "--channel".to_owned(),
        "stable".to_owned(),
        "--signing-identity".to_owned(),
        credentials.signing_identity.clone(),
    ];
    crate::app_bundle::run(repository_root, &package_arguments)?;
    build_step.finish("success");
    let stable_bundle =
        repository_root.join("target/astronomical-macos-stable.noindex/Astronomical.app");
    let plist_document = read_plist_document(&stable_bundle)?;
    if plist_document["CFBundleShortVersionString"].as_str()
        != Some(release_identity.release_version.as_str())
    {
        return Err(format!(
            "built app version does not match {}",
            release_identity.release_tag
        ));
    }
    if plist_document["SUFeedURL"].as_str() != Some(release_identity.update_feed_url.as_str()) {
        return Err("built app update feed does not match this repository".to_owned());
    }
    let signature_step = start_step("validate-developer-id-app");
    validate_distribution_app(&stable_bundle, &credentials.team_identifier)?;
    signature_step.finish("success");
    Ok(stable_bundle)
}

fn produce_notarized_disk_image(
    repository_root: &Path,
    stable_bundle: &Path,
    disk_image: &Path,
    credentials: &DistributionCredentials,
) -> Result<(), String> {
    let disk_image_request = DiskImageRequest {
        app_bundle: stable_bundle.to_path_buf(),
        output_disk_image: disk_image.to_path_buf(),
    };
    create_disk_image(repository_root, &disk_image_request)?;
    verify_notary_profile_access(&credentials.notary_profile)?;
    notarize_disk_image(&NotarizationRequest {
        disk_image: disk_image.to_path_buf(),
        signing_identity: credentials.signing_identity.clone(),
        notary_profile: credentials.notary_profile.clone(),
        timeout_seconds: DEFAULT_NOTARIZATION_TIMEOUT_SECONDS,
    })?;
    validate_disk_image(disk_image)
}

fn sign_appcast(
    repository_root: &Path,
    release_identity: &ReleaseIdentity,
    staging_directory: &Path,
    disk_image: &Path,
) -> Result<PathBuf, String> {
    let appcast_step = start_step("sign-dmg-and-appcast");
    let appcast_tool = locate_appcast_tool(repository_root)?;
    let download_prefix = release_download_prefix(release_identity);
    let appcast = staging_directory.join("appcast.xml");
    let appcast_status = Command::new(appcast_tool)
        .args([
            "--download-url-prefix",
            &download_prefix,
            "--embed-release-notes",
            "--maximum-deltas",
            "0",
            "-o",
        ])
        .arg(&appcast)
        .arg(staging_directory)
        .status();
    if !matches!(appcast_status, Ok(exit_status) if exit_status.success()) {
        return Err("Sparkle could not generate the appcast".to_owned());
    }
    if !matches!(Command::new("xmllint").arg("--noout").arg(&appcast).status(), Ok(exit_status) if exit_status.success())
    {
        return Err("the generated appcast is not well-formed XML".to_owned());
    }
    let appcast_text = std::fs::read_to_string(&appcast).unwrap_or_default();
    let asset_name = disk_image.file_name().map_or(String::new(), os_text);
    if let Some(appcast_problem) =
        appcast_problems(&appcast_text, &download_prefix, &asset_name).first()
    {
        return Err(appcast_problem.clone());
    }
    appcast_step.finish("success");
    Ok(appcast)
}

fn os_text(os_name: &std::ffi::OsStr) -> String {
    os_name.to_string_lossy().into_owned()
}

pub(crate) fn release_download_prefix(release_identity: &ReleaseIdentity) -> String {
    format!(
        "https://github.com/{}/releases/download/{}/",
        release_identity.repository_slug, release_identity.release_tag
    )
}

/// A Sparkle update is only safe when the archive carries its EdDSA
/// signature, the feed itself is signed, and the enclosure points at the
/// exact release asset.
pub(crate) fn appcast_problems(
    appcast_text: &str,
    download_prefix: &str,
    asset_name: &str,
) -> Vec<String> {
    let mut appcast_problems = Vec::new();
    if appcast_text.is_empty() {
        appcast_problems.push("Sparkle did not generate an appcast".to_owned());
    }
    if !appcast_text.contains("sparkle:edSignature=") {
        appcast_problems.push("appcast archive signature is missing".to_owned());
    }
    if !appcast_text.contains("sparkle-signatures:") {
        appcast_problems.push("signed appcast envelope is missing".to_owned());
    }
    if !appcast_text.contains(&format!("{download_prefix}{asset_name}")) {
        appcast_problems.push("appcast release URL is incorrect".to_owned());
    }
    appcast_problems
}

fn locate_appcast_tool(repository_root: &Path) -> Result<PathBuf, String> {
    for appcast_tool_path in APPCAST_TOOL_CANDIDATES {
        let candidate_tool = repository_root.join(appcast_tool_path);
        if candidate_tool.is_file() {
            return Ok(candidate_tool);
        }
    }
    Err("Sparkle generate_appcast is unavailable after the menu build".to_owned())
}

fn read_plist_document(bundle_path: &Path) -> Result<serde_json::Value, String> {
    let plist_json_text = super::release_identity::command_stdout(
        Path::new("/"),
        "plutil",
        &[
            "-convert",
            "json",
            "-o",
            "-",
            &bundle_path
                .join("Contents/Info.plist")
                .display()
                .to_string(),
        ],
    )?;
    match serde_json::from_str(&plist_json_text) {
        Ok(plist_document) => Ok(plist_document),
        Err(parse_error) => Err(format!("Info.plist did not convert to JSON: {parse_error}")),
    }
}

fn write_text(file_path: &Path, file_text: &str) -> Result<(), String> {
    match std::fs::write(file_path, file_text) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!(
            "could not write {}: {write_error}",
            file_path.display()
        )),
    }
}

/// A staging directory under `target/`, on the same filesystem as the
/// output so publishing it is one rename; removed on drop unless published.
struct StagingDirectory {
    directory_path: PathBuf,
    is_published: bool,
}

impl StagingDirectory {
    fn create(repository_root: &Path) -> Result<Self, String> {
        let directory_path = repository_root.join("target").join(format!(
            "astronomical-release-staging.{}",
            std::process::id()
        ));
        match std::fs::create_dir_all(&directory_path) {
            Ok(()) => Ok(Self {
                directory_path,
                is_published: false,
            }),
            Err(create_error) => Err(format!(
                "could not create the release staging directory: {create_error}"
            )),
        }
    }

    fn publish_as(mut self, output_directory: &Path) -> Result<(), String> {
        if let Some(output_parent) = output_directory.parent()
            && let Err(create_error) = std::fs::create_dir_all(output_parent)
        {
            return Err(format!(
                "could not create {}: {create_error}",
                output_parent.display()
            ));
        }
        match std::fs::rename(&self.directory_path, output_directory) {
            Ok(()) => {
                self.is_published = true;
                Ok(())
            }
            Err(rename_error) => Err(format!(
                "could not publish the prepared release: {rename_error}"
            )),
        }
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if !self.is_published {
            let _ = std::fs::remove_dir_all(&self.directory_path);
        }
    }
}
