//! `cargo xtask release app-store-pkg --app-bundle PATH --installer-identity
//! NAME [--output-pkg PATH]`: wraps a signed App Store channel bundle in the
//! signed installer package App Store Connect accepts. Typed port of the
//! retired `package-app-store-pkg.sh`.
//!
//! The package is signed with the Mac Developer Installer certificate; App
//! Store Connect rejects unsigned or Developer ID-signed packages, so the
//! identity is checked against the keychain before anything is built.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::system_step::run_system_step;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppStorePackageRequest {
    pub(crate) app_bundle: PathBuf,
    pub(crate) installer_identity: String,
    pub(crate) output_package: Option<PathBuf>,
}

pub(crate) fn package_for_app_store(
    package_request: &AppStorePackageRequest,
) -> Result<(), String> {
    if !package_request.app_bundle.is_dir() {
        return Err(format!(
            "app bundle not found: {}",
            package_request.app_bundle.display()
        ));
    }
    let plist_document = read_bundle_plist(&package_request.app_bundle)?;
    if let Some(channel_problem) = app_store_channel_problem(&plist_document) {
        return Err(channel_problem);
    }
    let Some(application_version) = plist_document["CFBundleShortVersionString"].as_str() else {
        return Err("the bundle does not declare its version".to_owned());
    };
    let identity_listing = match Command::new("security")
        .args(["find-identity", "-v"])
        .output()
    {
        Ok(identity_output) => String::from_utf8_lossy(&identity_output.stdout).into_owned(),
        Err(spawn_error) => return Err(format!("security could not start: {spawn_error}")),
    };
    if !identity_is_listed(&identity_listing, &package_request.installer_identity) {
        return Err(format!(
            "installer identity is not available as a valid code signing identity: {}",
            package_request.installer_identity
        ));
    }
    let output_package = resolved_output_package(package_request, application_version)?;
    build_and_check_package(package_request, &output_package)?;
    println!(
        "Signed App Store installer package: {}",
        output_package.display()
    );
    println!(
        "Validate it with cargo xtask release app-store-upload --package PATH --api-key ID --api-issuer ID; add --upload to send it."
    );
    Ok(())
}

fn build_and_check_package(
    package_request: &AppStorePackageRequest,
    output_package: &Path,
) -> Result<(), String> {
    run_system_step(
        "package-app-store-pkg",
        Command::new("xcrun")
            .args(["productbuild", "--component"])
            .arg(&package_request.app_bundle)
            .args([
                "/Applications",
                "--sign",
                &package_request.installer_identity,
            ])
            .arg(output_package),
    )?;
    run_system_step(
        "check-package-signature",
        Command::new("pkgutil")
            .arg("--check-signature")
            .arg(output_package),
    )
}

/// Only App Store channel bundles carry the sandbox and Application Support
/// state root App Review requires.
pub(crate) fn app_store_channel_problem(plist_document: &serde_json::Value) -> Option<String> {
    match plist_document["AstronomicalChannel"].as_str() {
        Some("app-store") => None,
        Some(declared_channel) => Some(format!(
            "only App Store channel bundles can be packaged for App Store Connect; this bundle declares {declared_channel}"
        )),
        None => Some("bundle does not declare an Astronomical channel".to_owned()),
    }
}

/// `security find-identity -v` quotes each identity's full name; matching
/// the quoted name avoids accepting an identity whose name merely contains
/// the requested one.
pub(crate) fn identity_is_listed(identity_listing: &str, installer_identity: &str) -> bool {
    !installer_identity.is_empty()
        && identity_listing.contains(&format!("\"{installer_identity}\""))
}

/// Beside the bundle by default; a relative path is anchored to the
/// working directory so the success line always names an absolute path.
pub(crate) fn default_output_package(app_bundle: &Path, application_version: &str) -> PathBuf {
    let package_name = format!("Astronomical-{application_version}-app-store.pkg");
    match app_bundle.parent() {
        Some(bundle_parent) => bundle_parent.join(package_name),
        None => PathBuf::from(package_name),
    }
}

fn resolved_output_package(
    package_request: &AppStorePackageRequest,
    application_version: &str,
) -> Result<PathBuf, String> {
    let output_package = match &package_request.output_package {
        Some(output_package) => output_package.clone(),
        None => default_output_package(&package_request.app_bundle, application_version),
    };
    if output_package.is_absolute() {
        return Ok(output_package);
    }
    match std::env::current_dir() {
        Ok(working_directory) => Ok(working_directory.join(output_package)),
        Err(directory_error) => Err(format!(
            "could not resolve the working directory: {directory_error}"
        )),
    }
}

fn read_bundle_plist(app_bundle: &Path) -> Result<serde_json::Value, String> {
    let info_plist = app_bundle.join("Contents/Info.plist");
    let plist_output = Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(&info_plist)
        .output();
    let plist_json_text = match plist_output {
        Ok(plist_output) if plist_output.status.success() => {
            String::from_utf8_lossy(&plist_output.stdout).into_owned()
        }
        _ => {
            return Err(format!(
                "bundle Info.plist is unavailable: {}",
                info_plist.display()
            ));
        }
    };
    match serde_json::from_str(&plist_json_text) {
        Ok(plist_document) => Ok(plist_document),
        Err(parse_error) => Err(format!("Info.plist did not convert to JSON: {parse_error}")),
    }
}
