//! The two publisher entry points.
//!
//! `cargo xtask release prepare-and-publish` takes the credentials
//! explicitly. `cargo xtask release prepare-signed` resolves the documented
//! credential map first (exactly one Developer ID Application identity, its
//! ten-character Team ID, and the "Astronomical Notarization" profile) and
//! then runs the same publisher in this process. Typed port of the retired
//! `prepare-and-publish.sh` and `prepare-signed-release.sh`.
//!
//! Without `--publish` nothing on GitHub changes; `--publish` uploads only
//! the exact directory a previous run prepared and someone reviewed.

use std::path::Path;

use super::github_publication::{load_prepared_release, publish_release};
use super::release_identity::{PublicationRequest, command_stdout, resolve_release_identity};
use super::release_preparation::{
    DistributionCredentials, prepare_release, prevent_idle_lock, validate_distribution_credentials,
};
use crate::step_log::start_step;

pub(crate) const RELEASE_NOTARY_PROFILE: &str = "Astronomical Notarization";

pub(crate) fn run_prepare_and_publish(
    repository_root: &Path,
    publication_request: &PublicationRequest,
) -> Result<(), String> {
    let identity_step = start_step("validate-release-identity");
    let release_identity = resolve_release_identity(repository_root, publication_request)?;
    identity_step.finish("success");
    if publication_request.is_publish {
        let release_manifest = load_prepared_release(&release_identity)?;
        publish_release(repository_root, &release_identity, &release_manifest)?;
        println!(
            "Published GitHub Release {}; review and commit site/appcast.xml to activate the update feed.",
            release_identity.release_tag
        );
        return Ok(());
    }
    let credentials = explicit_credentials(publication_request)?;
    let Some(notes_file) = &publication_request.notes_file else {
        return Err("--notes-file is required while preparing a release".to_owned());
    };
    // Held for the whole preparation; caffeinate exits with this process.
    let _idle_lock_prevention = prevent_idle_lock()?;
    validate_distribution_credentials(&credentials)?;
    let prepared_disk_image =
        prepare_release(repository_root, &release_identity, &credentials, notes_file)?;
    println!("Prepared signed release: {}", prepared_disk_image.display());
    println!(
        "No GitHub state changed. Review this exact directory, then re-run with the same output directory and --publish."
    );
    Ok(())
}

fn explicit_credentials(
    publication_request: &PublicationRequest,
) -> Result<DistributionCredentials, String> {
    let (Some(signing_identity), Some(team_identifier), Some(notary_profile)) = (
        &publication_request.signing_identity,
        &publication_request.team_identifier,
        &publication_request.notary_profile,
    ) else {
        return Err("--signing-identity, --team-id, and --notary-profile are required while preparing a release".to_owned());
    };
    Ok(DistributionCredentials {
        signing_identity: signing_identity.clone(),
        team_identifier: team_identifier.clone(),
        notary_profile: notary_profile.clone(),
    })
}

/// Resolves the credential map, then runs the publisher with it.
pub(crate) fn run_prepare_signed(
    repository_root: &Path,
    publication_request: &PublicationRequest,
) -> Result<(), String> {
    let identity_listing = command_stdout(
        repository_root,
        "security",
        &["find-identity", "-v", "-p", "codesigning"],
    )?;
    let (signing_identity, team_identifier) = resolve_credential_map(&identity_listing)?;
    println!(
        "[prepare-signed-release] resolved credential map; preparing {}",
        publication_request.release_tag
    );
    let mut credentialed_request = publication_request.clone();
    credentialed_request.signing_identity = Some(signing_identity);
    credentialed_request.team_identifier = Some(team_identifier);
    credentialed_request.notary_profile = Some(RELEASE_NOTARY_PROFILE.to_owned());
    run_prepare_and_publish(repository_root, &credentialed_request)
}

/// Exactly one Developer ID Application identity, and the ten-character
/// Team ID in its trailing parentheses. Zero or several identities fail
/// closed, because signing with the wrong one would poison the release.
pub(crate) fn resolve_credential_map(identity_listing: &str) -> Result<(String, String), String> {
    let mut developer_identities = Vec::new();
    for listing_line in identity_listing.lines() {
        let Some(quote_start) = listing_line.find("\"Developer ID Application:") else {
            continue;
        };
        let quoted_identity = &listing_line[quote_start + 1..];
        if let Some(quote_end) = quoted_identity.find('"') {
            developer_identities.push(quoted_identity[..quote_end].to_owned());
        }
    }
    developer_identities.dedup();
    let [signing_identity] = developer_identities.as_slice() else {
        return Err("exactly one Developer ID Application identity is required".to_owned());
    };
    let team_identifier = signing_identity
        .rsplit_once('(')
        .and_then(team_identifier_in_parentheses)
        .unwrap_or_default();
    if team_identifier.len() != 10 {
        return Err("Developer ID identity does not contain a 10-character Team ID".to_owned());
    }
    Ok((signing_identity.clone(), team_identifier))
}

fn team_identifier_in_parentheses(identity_halves: (&str, &str)) -> Option<String> {
    let team_identifier = identity_halves.1.strip_suffix(')')?;
    if team_identifier.bytes().all(is_ascii_alphanumeric_byte) {
        Some(team_identifier.to_owned())
    } else {
        None
    }
}

fn is_ascii_alphanumeric_byte(candidate_byte: u8) -> bool {
    candidate_byte.is_ascii_alphanumeric()
}
