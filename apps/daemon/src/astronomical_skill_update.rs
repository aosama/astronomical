//! Picks up a newer Astronomical skill from the project's site, so the skill
//! can improve between app releases (issue #2397).
//!
//! `cargo xtask skill-site` publishes the skill's files and a manifest with
//! the skill's version, the oldest app version it suits, and each file's
//! SHA-256. This module reads the manifest and, when it describes a newer
//! skill this app can use, downloads the files, checks each digest, and swaps
//! the whole folder in at once. The compiled-in skill stays the floor: a
//! failed or refused update changes nothing.
//!
//! Trust is HTTPS from the project's own site and nothing more; the digests
//! detect an incomplete download, not a hostile publisher.

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::astronomical_skill::{
    ASTRONOMICAL_SKILL_NAME, bundled_skill_version, downloaded_skill_version,
    record_downloaded_skill_version,
};

/// Where `cargo xtask skill-site` publishes the skill.
pub const SKILL_SITE_BASE_URL: &str = "https://getastronomical.com/skill";
const MANIFEST_FILE_NAME: &str = "manifest.json";
const INCOMING_FOLDER_NAME: &str = "astronomical.incoming";
const OUTGOING_FOLDER_NAME: &str = "astronomical.outgoing";
/// A skill is a short index and a dozen reference files; these bounds refuse
/// a manifest or file that is plainly something else.
const MAXIMUM_SKILL_FILE_COUNT: usize = 64;
const MAXIMUM_SKILL_FILE_BYTES: usize = 256 * 1024;
/// The files are a few kilobytes each; a site slower than this is treated as
/// unreachable and tried again on the next daily look.
const SITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

#[derive(Debug, Deserialize)]
struct SkillManifest {
    skill_version: u64,
    minimum_app_version: String,
    files: BTreeMap<String, String>,
}

/// What one look at the site led to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SkillUpdateOutcome {
    /// The skill on this Mac is at least as new as the published one.
    AlreadyCurrent { installed_version: u64 },
    /// The published skill is for a newer app; this app keeps its own.
    NeedsNewerApp { minimum_app_version: String },
    /// A newer skill was downloaded and is now the one models read.
    Updated { from_version: u64, to_version: u64 },
}

/// Why a look at the site changed nothing.
#[derive(Debug, Error)]
pub enum SkillUpdateError {
    #[error("the skill site could not be reached: {0}")]
    Unreachable(#[from] reqwest::Error),
    #[error("the skill site answered {status} for {url}")]
    Refused { url: String, status: u16 },
    #[error("the skill manifest is not usable: {0}")]
    UnusableManifest(String),
    #[error("{relative_path} did not match the manifest's digest")]
    DigestMismatch { relative_path: String },
    #[error("the downloaded skill could not be stored: {0}")]
    Storage(#[from] io::Error),
}

/// The client for the skill site. Production allows HTTPS only; the tests'
/// stand-in site is plain HTTP on loopback.
pub(crate) fn skill_site_http_client(https_only: bool) -> Result<reqwest::Client, reqwest::Error> {
    // reqwest is built without a TLS provider of its own; see
    // `install_default_tls_crypto_provider`.
    crate::library::crypto_provider::install_default_tls_crypto_provider();
    reqwest::Client::builder()
        .https_only(https_only)
        .connect_timeout(SITE_TIMEOUT)
        .read_timeout(SITE_TIMEOUT)
        .build()
}

/// The client production uses to reach the skill site.
pub fn production_skill_site_http_client() -> Result<reqwest::Client, reqwest::Error> {
    skill_site_http_client(true)
}

/// Looks at the site once and installs a newer skill this app can use.
pub async fn update_skill_from_site(
    http_client: &reqwest::Client,
    site_base_url: &str,
    skill_directory: &Path,
    app_version: &str,
) -> Result<SkillUpdateOutcome, SkillUpdateError> {
    let manifest_bytes = fetch(
        http_client,
        &format!("{site_base_url}/{MANIFEST_FILE_NAME}"),
    )
    .await?;
    let manifest: SkillManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|parse_error| SkillUpdateError::UnusableManifest(parse_error.to_string()))?;
    validate_manifest(&manifest)?;
    let installed_version = installed_skill_version(skill_directory);
    if manifest.skill_version <= installed_version {
        return Ok(SkillUpdateOutcome::AlreadyCurrent { installed_version });
    }
    if !app_is_at_least(app_version, &manifest.minimum_app_version) {
        return Ok(SkillUpdateOutcome::NeedsNewerApp {
            minimum_app_version: manifest.minimum_app_version,
        });
    }
    let incoming_directory = skill_directory.join(INCOMING_FOLDER_NAME);
    let download_result =
        download_skill_files(http_client, site_base_url, &manifest, &incoming_directory).await;
    if let Err(download_error) = download_result {
        // Best effort: the download's own error is the one worth reporting.
        let _ = fs::remove_dir_all(&incoming_directory);
        return Err(download_error);
    }
    swap_in_downloaded_skill(skill_directory, &incoming_directory)?;
    record_downloaded_skill_version(skill_directory, manifest.skill_version)?;
    Ok(SkillUpdateOutcome::Updated {
        from_version: installed_version,
        to_version: manifest.skill_version,
    })
}

/// The newer of the compiled-in skill and a downloaded one.
fn installed_skill_version(skill_directory: &Path) -> u64 {
    let bundled_version = bundled_skill_version();
    match downloaded_skill_version(skill_directory) {
        Some(downloaded_version) if downloaded_version > bundled_version => downloaded_version,
        _ => bundled_version,
    }
}

fn validate_manifest(manifest: &SkillManifest) -> Result<(), SkillUpdateError> {
    if manifest.files.is_empty() || manifest.files.len() > MAXIMUM_SKILL_FILE_COUNT {
        return Err(SkillUpdateError::UnusableManifest(format!(
            "it lists {} files",
            manifest.files.len()
        )));
    }
    if !manifest.files.contains_key("SKILL.md") {
        return Err(SkillUpdateError::UnusableManifest(
            "it has no SKILL.md".to_owned(),
        ));
    }
    for relative_path in manifest.files.keys() {
        if !is_skill_file_path(relative_path) {
            return Err(SkillUpdateError::UnusableManifest(format!(
                "{relative_path} is not a skill file path"
            )));
        }
    }
    Ok(())
}

/// A skill holds `SKILL.md` and Markdown files directly under `references/`.
/// Anything else, above all a path that climbs out of the folder, is refused
/// before a single byte is written.
fn is_skill_file_path(relative_path: &str) -> bool {
    if relative_path == "SKILL.md" {
        return true;
    }
    let Some(reference_file_name) = relative_path.strip_prefix("references/") else {
        return false;
    };
    let Some(reference_name) = reference_file_name.strip_suffix(".md") else {
        return false;
    };
    !reference_name.is_empty() && reference_name.chars().all(is_reference_name_character)
}

fn is_reference_name_character(character: char) -> bool {
    character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
}

/// Whether `app_version` is `minimum_app_version` or later, comparing the
/// dotted numbers one by one. A version that is not dotted numbers suits
/// nothing, so a malformed manifest can never look compatible.
fn app_is_at_least(app_version: &str, minimum_app_version: &str) -> bool {
    match (
        version_numbers(app_version),
        version_numbers(minimum_app_version),
    ) {
        (Some(app_numbers), Some(minimum_numbers)) => app_numbers >= minimum_numbers,
        _ => false,
    }
}

fn version_numbers(version_text: &str) -> Option<Vec<u64>> {
    let mut version_numbers = Vec::new();
    for version_part in version_text.split('.') {
        version_numbers.push(version_part.parse::<u64>().ok()?);
    }
    Some(version_numbers)
}

async fn download_skill_files(
    http_client: &reqwest::Client,
    site_base_url: &str,
    manifest: &SkillManifest,
    incoming_directory: &Path,
) -> Result<(), SkillUpdateError> {
    // A folder left by an interrupted earlier attempt would mix old files in.
    let _ = fs::remove_dir_all(incoming_directory);
    fs::create_dir_all(incoming_directory.join("references"))?;
    for (relative_path, expected_digest) in &manifest.files {
        let file_url = format!("{site_base_url}/{ASTRONOMICAL_SKILL_NAME}/{relative_path}");
        let file_bytes = fetch(http_client, &file_url).await?;
        let actual_digest = lowercase_hex::encode_lowercase_hex(&Sha256::digest(&file_bytes));
        if &actual_digest != expected_digest {
            return Err(SkillUpdateError::DigestMismatch {
                relative_path: relative_path.clone(),
            });
        }
        fs::write(incoming_directory.join(relative_path), file_bytes)?;
    }
    Ok(())
}

/// Replaces the skill folder with the downloaded one through two renames, so
/// an agent reading the skill finds either the old one whole or the new one
/// whole.
fn swap_in_downloaded_skill(skill_directory: &Path, incoming_directory: &Path) -> io::Result<()> {
    let skill_root: PathBuf = skill_directory.join(ASTRONOMICAL_SKILL_NAME);
    let outgoing_directory = skill_directory.join(OUTGOING_FOLDER_NAME);
    let _ = fs::remove_dir_all(&outgoing_directory);
    if skill_root.exists() {
        fs::rename(&skill_root, &outgoing_directory)?;
    }
    if let Err(rename_error) = fs::rename(incoming_directory, &skill_root) {
        // Put the previous skill back so models are never left without one.
        let _ = fs::rename(&outgoing_directory, &skill_root);
        return Err(rename_error);
    }
    let _ = fs::remove_dir_all(&outgoing_directory);
    Ok(())
}

async fn fetch(http_client: &reqwest::Client, url: &str) -> Result<Vec<u8>, SkillUpdateError> {
    let response = http_client.get(url).send().await?;
    if !response.status().is_success() {
        return Err(SkillUpdateError::Refused {
            url: url.to_owned(),
            status: response.status().as_u16(),
        });
    }
    // Refuse before reading when the site declares an oversized body; the
    // check after reading covers a response that declares no length.
    if let Some(declared_bytes) = response.content_length()
        && declared_bytes > MAXIMUM_SKILL_FILE_BYTES as u64
    {
        return Err(SkillUpdateError::UnusableManifest(format!(
            "{url} is {declared_bytes} bytes, more than a skill file can be"
        )));
    }
    let response_bytes = response.bytes().await?;
    if response_bytes.len() > MAXIMUM_SKILL_FILE_BYTES {
        return Err(SkillUpdateError::UnusableManifest(format!(
            "{url} is {} bytes, more than a skill file can be",
            response_bytes.len()
        )));
    }
    Ok(response_bytes.to_vec())
}

/// How long after the engine starts the first look happens: late enough to
/// stay out of the way of startup and the first model load.
const FIRST_LOOK_DELAY: std::time::Duration = std::time::Duration::from_secs(60);
/// The skill changes rarely; once a day keeps it current without making the
/// app chatty on the network.
const LOOK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// Looks at the site shortly after start and then once a day, for as long as
/// the engine runs. The person's switch is read before every look, so turning
/// it off in Observatory stops the next one without a restart.
pub async fn keep_skill_current(
    reloadable_runtime_config: std::sync::Arc<std::sync::RwLock<crate::ResolvedRuntimeConfig>>,
    skill_directory: PathBuf,
    app_version: &'static str,
) {
    let http_client = match production_skill_site_http_client() {
        Ok(http_client) => http_client,
        Err(client_error) => {
            tracing::warn!(error = %client_error, "the skill update check could not start");
            return;
        }
    };
    tokio::time::sleep(FIRST_LOOK_DELAY).await;
    loop {
        if updates_are_switched_on(&reloadable_runtime_config) {
            let look_result = update_skill_from_site(
                &http_client,
                SKILL_SITE_BASE_URL,
                &skill_directory,
                app_version,
            )
            .await;
            match look_result {
                Ok(update_outcome) => tracing::info!(?update_outcome, "looked for a newer skill"),
                Err(update_error) => {
                    tracing::warn!(error = %update_error, "the skill was not updated")
                }
            }
        }
        tokio::time::sleep(LOOK_INTERVAL).await;
    }
}

fn updates_are_switched_on(
    reloadable_runtime_config: &std::sync::RwLock<crate::ResolvedRuntimeConfig>,
) -> bool {
    match reloadable_runtime_config.read() {
        Ok(resolved_runtime_config) => resolved_runtime_config.skill_settings.updates_from_web,
        Err(_) => false,
    }
}

#[cfg(test)]
#[path = "astronomical_skill_update_tests.rs"]
mod astronomical_skill_update_tests;
