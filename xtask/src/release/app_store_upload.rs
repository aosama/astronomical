//! `cargo xtask release app-store-upload --package PATH --api-key ID
//! --api-issuer ID [--upload]`: asks App Store Connect to validate a store
//! installer package and, only with `--upload`, sends it.
//!
//! Validation is the default because it is free of consequences: Apple
//! checks the package and lists what is wrong with it without creating a
//! build. An upload cannot be taken back, since a build number can be used
//! only once per version, so it must be asked for by name and still runs
//! only after validation has passed in the same invocation.
//!
//! Authentication is an App Store Connect API key. Its private half stays
//! where Apple's tool looks for it, outside the repository; this command is
//! given only the key's identifier and the issuer's, neither of which is a
//! secret.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::system_step::run_system_step;

/// The installer certificate App Store Connect accepts packages signed with.
const STORE_INSTALLER_CERTIFICATE_NAME: &str = "3rd Party Mac Developer Installer";
/// The folders, beneath the home directory, where Apple's upload tool looks
/// for a private key named `AuthKey_<ID>.p8`.
const PRIVATE_KEY_FOLDERS: [&str; 3] = [
    ".appstoreconnect/private_keys",
    ".private_keys",
    "private_keys",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppStoreUploadRequest {
    pub(crate) package: PathBuf,
    pub(crate) api_key_identifier: String,
    pub(crate) api_issuer_identifier: String,
    pub(crate) should_upload: bool,
}

/// What App Store Connect is asked to do with the package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreSubmissionStep {
    Validate,
    Upload,
}

pub(crate) fn validate_and_upload_for_app_store(
    upload_request: &AppStoreUploadRequest,
) -> Result<(), String> {
    if !upload_request.package.is_file() {
        return Err(format!(
            "store package not found: {}",
            upload_request.package.display()
        ));
    }
    let signature_report = package_signature_report(&upload_request.package)?;
    if let Some(signature_problem) = store_package_signature_problem(&signature_report) {
        return Err(signature_problem);
    }
    let home_directory = home_directory()?;
    if installed_private_key(&home_directory, &upload_request.api_key_identifier).is_none() {
        return Err(missing_private_key_message(
            &upload_request.api_key_identifier,
        ));
    }
    run_system_step(
        "validate-app-store-package",
        Command::new("xcrun").args(submission_arguments(
            upload_request,
            StoreSubmissionStep::Validate,
        )),
    )?;
    if !upload_request.should_upload {
        println!(
            "App Store Connect accepted the package in validation. Nothing was uploaded; add --upload to send this build, which cannot be taken back."
        );
        return Ok(());
    }
    run_system_step(
        "upload-app-store-package",
        Command::new("xcrun").args(submission_arguments(
            upload_request,
            StoreSubmissionStep::Upload,
        )),
    )?;
    println!("Uploaded to App Store Connect. The build appears there once Apple has processed it.");
    Ok(())
}

/// The arguments after `xcrun` for one step. Apple's tool prints its own
/// findings, so its output is left to stream through untouched.
pub(crate) fn submission_arguments(
    upload_request: &AppStoreUploadRequest,
    submission_step: StoreSubmissionStep,
) -> Vec<String> {
    let step_flag = match submission_step {
        StoreSubmissionStep::Validate => "--validate-app",
        StoreSubmissionStep::Upload => "--upload-package",
    };
    vec![
        "altool".to_owned(),
        step_flag.to_owned(),
        upload_request.package.display().to_string(),
        "--api-key".to_owned(),
        upload_request.api_key_identifier.clone(),
        "--api-issuer".to_owned(),
        upload_request.api_issuer_identifier.clone(),
    ]
}

/// A package signed with anything but the store installer certificate, or
/// not signed at all, would be refused after the whole upload; it is cheaper
/// to refuse it here.
pub(crate) fn store_package_signature_problem(signature_report: &str) -> Option<String> {
    if signature_report.contains(STORE_INSTALLER_CERTIFICATE_NAME) {
        return None;
    }
    Some(format!(
        "this is not a store package: it is not signed with a \"{STORE_INSTALLER_CERTIFICATE_NAME}\" certificate. Build one with cargo xtask release app-store-pkg"
    ))
}

/// Where the private key for `api_key_identifier` would be, in the order
/// Apple's tool searches.
pub(crate) fn private_key_candidates(
    home_directory: &Path,
    api_key_identifier: &str,
) -> Vec<PathBuf> {
    let mut private_key_candidates = Vec::new();
    for private_key_folder in PRIVATE_KEY_FOLDERS {
        private_key_candidates.push(
            home_directory
                .join(private_key_folder)
                .join(format!("AuthKey_{api_key_identifier}.p8")),
        );
    }
    private_key_candidates
}

fn installed_private_key(home_directory: &Path, api_key_identifier: &str) -> Option<PathBuf> {
    private_key_candidates(home_directory, api_key_identifier)
        .into_iter()
        .find(|private_key_candidate| private_key_candidate.is_file())
}

/// Says where the key belongs, since Apple's tool reports a missing key only
/// as a failed sign-in.
pub(crate) fn missing_private_key_message(api_key_identifier: &str) -> String {
    format!(
        "no private key for API key {api_key_identifier} was found. Download it once from App Store Connect (Users and Access, Integrations, App Store Connect API) and save it as ~/{}/AuthKey_{api_key_identifier}.p8",
        PRIVATE_KEY_FOLDERS[0]
    )
}

fn package_signature_report(package: &Path) -> Result<String, String> {
    match Command::new("pkgutil")
        .arg("--check-signature")
        .arg(package)
        .output()
    {
        // An unsigned package makes pkgutil exit non-zero while still
        // describing it, so the text is read whatever the exit status.
        Ok(signature_output) => Ok(String::from_utf8_lossy(&signature_output.stdout).into_owned()),
        Err(spawn_error) => Err(format!("pkgutil could not start: {spawn_error}")),
    }
}

fn home_directory() -> Result<PathBuf, String> {
    match std::env::var_os("HOME") {
        Some(home_directory) => Ok(PathBuf::from(home_directory)),
        None => Err("the home directory is unknown, so the API key cannot be located".to_owned()),
    }
}

#[cfg(test)]
#[path = "app_store_upload_tests.rs"]
mod app_store_upload_tests;
