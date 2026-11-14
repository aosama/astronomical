//! Argument parsing for `cargo xtask release`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::app_store_package::{AppStorePackageRequest, package_for_app_store};
use super::app_store_upload::{AppStoreUploadRequest, validate_and_upload_for_app_store};
use super::disk_image_creation::{DiskImageRequest, create_disk_image};
use super::disk_image_validation::validate_disk_image;
use super::distribution_signature::validate_distribution_app;
use super::notarization::{NotarizationRequest, notarize_disk_image, parse_timeout_seconds};
use super::publication_entry::{run_prepare_and_publish, run_prepare_signed};
use super::release_identity::PublicationRequest;
use super::sparkle_acceptance::accept_sparkle_update;
use super::stable_installation::{
    InstallRequest, build_and_install_stable, default_stable_bundle, install_stable,
};

pub(crate) const RELEASE_USAGE: &str = "usage: cargo xtask release create-dmg --app-bundle PATH --output PATH | release validate-dmg --dmg PATH | release validate-distribution-app --app-bundle PATH --team-id ID | release notarize-dmg --dmg PATH --signing-identity NAME --notary-profile PROFILE [--timeout-seconds N] | release install-stable [--app-bundle PATH] [--dry-run] | release build-and-install-stable [--dry-run] | release app-store-pkg --app-bundle PATH --installer-identity NAME [--output-pkg PATH] | release app-store-upload --package PATH --api-key ID --api-issuer ID [--upload] | release prepare-and-publish --tag vX.Y.Z (--notes-file PATH --signing-identity NAME --team-id ID --notary-profile NAME | --publish) [--output-directory PATH] | release prepare-signed --tag vX.Y.Z [--notes-file PATH] [--output-directory PATH] [--publish] | release accept-sparkle-update";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReleaseCommand {
    CreateDiskImage(DiskImageRequest),
    ValidateDiskImage(PathBuf),
    ValidateDistributionApp {
        app_bundle: PathBuf,
        team_identifier: String,
    },
    NotarizeDiskImage(NotarizationRequest),
    InstallStable {
        app_bundle: Option<PathBuf>,
        is_dry_run: bool,
    },
    BuildAndInstallStable {
        is_dry_run: bool,
    },
    AppStore(AppStoreCommand),
    PrepareAndPublish(PublicationRequest),
    PrepareSigned(PublicationRequest),
    AcceptSparkleUpdate,
}

/// Routes the subcommand to its own parser, so each subcommand's flags and
/// requirements live in one small function rather than one long match.
pub(crate) fn parse_release_command(arguments: &[String]) -> Result<ReleaseCommand, String> {
    let Some((subcommand, subcommand_arguments)) = arguments.split_first() else {
        return Err(RELEASE_USAGE.to_owned());
    };
    match subcommand.as_str() {
        "create-dmg" => parse_disk_image_creation_command(subcommand_arguments),
        "validate-dmg" => parse_disk_image_validation_command(subcommand_arguments),
        "validate-distribution-app" => parse_distribution_validation_command(subcommand_arguments),
        "notarize-dmg" => parse_notarization_command(subcommand_arguments),
        "prepare-and-publish" => Ok(ReleaseCommand::PrepareAndPublish(
            parse_publication_request(subcommand_arguments, PREPARE_AND_PUBLISH_FLAGS)?,
        )),
        "prepare-signed" => Ok(ReleaseCommand::PrepareSigned(parse_publication_request(
            subcommand_arguments,
            PREPARE_SIGNED_FLAGS,
        )?)),
        "accept-sparkle-update" if subcommand_arguments.is_empty() => {
            Ok(ReleaseCommand::AcceptSparkleUpdate)
        }
        "app-store-pkg" | "app-store-upload" => {
            parse_app_store_command(subcommand, subcommand_arguments)
        }
        "install-stable" => parse_install_stable_command(subcommand_arguments),
        "build-and-install-stable" => parse_build_and_install_stable_command(subcommand_arguments),
        _ => Err(RELEASE_USAGE.to_owned()),
    }
}

const PREPARE_AND_PUBLISH_FLAGS: &[&str] = &[
    "--tag",
    "--notes-file",
    "--output-directory",
    "--signing-identity",
    "--team-id",
    "--notary-profile",
];

const PREPARE_SIGNED_FLAGS: &[&str] = &["--tag", "--notes-file", "--output-directory"];

fn parse_disk_image_creation_command(
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    let flag_values = parse_flag_values(subcommand_arguments, &["--app-bundle", "--output"])?;
    Ok(ReleaseCommand::CreateDiskImage(DiskImageRequest {
        app_bundle: required_path(&flag_values, "--app-bundle")?,
        output_disk_image: required_path(&flag_values, "--output")?,
    }))
}

fn parse_disk_image_validation_command(
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    let flag_values = parse_flag_values(subcommand_arguments, &["--dmg"])?;
    Ok(ReleaseCommand::ValidateDiskImage(required_path(
        &flag_values,
        "--dmg",
    )?))
}

fn parse_distribution_validation_command(
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    let flag_values = parse_flag_values(subcommand_arguments, &["--app-bundle", "--team-id"])?;
    Ok(ReleaseCommand::ValidateDistributionApp {
        app_bundle: required_path(&flag_values, "--app-bundle")?,
        team_identifier: required_text(&flag_values, "--team-id")?,
    })
}

/// The two Mac App Store subcommands, kept as one release command so the
/// release dispatch does not grow with each store step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AppStoreCommand {
    /// Wrap a store bundle in its signed installer package.
    Package(AppStorePackageRequest),
    /// Ask App Store Connect to validate that package, and send it on request.
    Upload(AppStoreUploadRequest),
}

fn parse_app_store_command(
    subcommand_name: &str,
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    if subcommand_name == "app-store-upload" {
        parse_app_store_upload_command(subcommand_arguments)
    } else {
        parse_app_store_package_command(subcommand_arguments)
    }
}

fn run_app_store_command(app_store_command: &AppStoreCommand) -> Result<(), String> {
    match app_store_command {
        AppStoreCommand::Package(package_request) => package_for_app_store(package_request),
        AppStoreCommand::Upload(upload_request) => {
            validate_and_upload_for_app_store(upload_request)
        }
    }
}

fn parse_app_store_package_command(
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    let flag_values = parse_flag_values(
        subcommand_arguments,
        &["--app-bundle", "--installer-identity", "--output-pkg"],
    )?;
    Ok(ReleaseCommand::AppStore(AppStoreCommand::Package(
        AppStorePackageRequest {
            app_bundle: required_path(&flag_values, "--app-bundle")?,
            installer_identity: required_text(&flag_values, "--installer-identity")?,
            output_package: flag_values.get("--output-pkg").map(PathBuf::from),
        },
    )))
}

/// Validation is the default; `--upload` must be given by name, because an
/// uploaded build cannot be taken back.
fn parse_app_store_upload_command(
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    let mut should_upload = false;
    let mut valued_arguments = Vec::new();
    for subcommand_argument in subcommand_arguments {
        if subcommand_argument == "--upload" {
            should_upload = true;
        } else {
            valued_arguments.push(subcommand_argument.clone());
        }
    }
    let flag_values = parse_flag_values(
        &valued_arguments,
        &["--package", "--api-key", "--api-issuer"],
    )?;
    Ok(ReleaseCommand::AppStore(AppStoreCommand::Upload(
        AppStoreUploadRequest {
            package: required_path(&flag_values, "--package")?,
            api_key_identifier: required_text(&flag_values, "--api-key")?,
            api_issuer_identifier: required_text(&flag_values, "--api-issuer")?,
            should_upload,
        },
    )))
}

fn parse_install_stable_command(subcommand_arguments: &[String]) -> Result<ReleaseCommand, String> {
    let (is_dry_run, valued_arguments) = take_dry_run_flag(subcommand_arguments);
    let flag_values = parse_flag_values(&valued_arguments, &["--app-bundle"])?;
    Ok(ReleaseCommand::InstallStable {
        app_bundle: flag_values.get("--app-bundle").map(PathBuf::from),
        is_dry_run,
    })
}

/// Takes no flag but `--dry-run`, so the whole argument list is matched
/// exactly rather than parsed as `--flag value` pairs.
fn parse_build_and_install_stable_command(
    subcommand_arguments: &[String],
) -> Result<ReleaseCommand, String> {
    match subcommand_arguments {
        [] => Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: false }),
        [dry_run_flag] if dry_run_flag == "--dry-run" => {
            Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: true })
        }
        _ => Err(RELEASE_USAGE.to_owned()),
    }
}

fn parse_notarization_command(subcommand_arguments: &[String]) -> Result<ReleaseCommand, String> {
    let flag_values = parse_flag_values(
        subcommand_arguments,
        &[
            "--dmg",
            "--signing-identity",
            "--notary-profile",
            "--timeout-seconds",
        ],
    )?;
    Ok(ReleaseCommand::NotarizeDiskImage(NotarizationRequest {
        disk_image: required_path(&flag_values, "--dmg")?,
        signing_identity: required_text(&flag_values, "--signing-identity")?,
        notary_profile: required_text(&flag_values, "--notary-profile")?,
        timeout_seconds: parse_timeout_seconds(flag_values.get("--timeout-seconds"))?,
    }))
}

/// `--dry-run` is the one valueless flag; it is taken out before the
/// `--flag value` pairs are parsed.
fn take_dry_run_flag(subcommand_arguments: &[String]) -> (bool, Vec<String>) {
    take_switch_flag(subcommand_arguments, "--dry-run")
}

/// Removes a valueless switch from the arguments and reports whether it was
/// present.
fn take_switch_flag(subcommand_arguments: &[String], switch_flag: &str) -> (bool, Vec<String>) {
    let mut is_present = false;
    let mut valued_arguments = Vec::new();
    for subcommand_argument in subcommand_arguments {
        if subcommand_argument == switch_flag {
            is_present = true;
        } else {
            valued_arguments.push(subcommand_argument.clone());
        }
    }
    (is_present, valued_arguments)
}

fn parse_publication_request(
    subcommand_arguments: &[String],
    accepted_flags: &[&str],
) -> Result<PublicationRequest, String> {
    let (is_publish, valued_arguments) = take_switch_flag(subcommand_arguments, "--publish");
    let flag_values = parse_flag_values(&valued_arguments, accepted_flags)?;
    Ok(PublicationRequest {
        release_tag: required_text(&flag_values, "--tag")?,
        notes_file: flag_values.get("--notes-file").map(PathBuf::from),
        output_directory: flag_values.get("--output-directory").map(PathBuf::from),
        is_publish,
        signing_identity: flag_values.get("--signing-identity").cloned(),
        team_identifier: flag_values.get("--team-id").cloned(),
        notary_profile: flag_values.get("--notary-profile").cloned(),
    })
}

fn required_text(flag_values: &BTreeMap<String, String>, flag: &str) -> Result<String, String> {
    match flag_values.get(flag) {
        Some(flag_value) => Ok(flag_value.clone()),
        None => Err(format!("{flag} is required\n{RELEASE_USAGE}")),
    }
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    match parse_release_command(arguments)? {
        ReleaseCommand::CreateDiskImage(disk_image_request) => {
            create_disk_image(repository_root, &disk_image_request)
        }
        ReleaseCommand::ValidateDiskImage(disk_image_path) => validate_disk_image(&disk_image_path),
        ReleaseCommand::ValidateDistributionApp {
            app_bundle,
            team_identifier,
        } => validate_distribution_app(&app_bundle, &team_identifier),
        ReleaseCommand::NotarizeDiskImage(notarization_request) => {
            notarize_disk_image(&notarization_request)
        }
        ReleaseCommand::InstallStable {
            app_bundle,
            is_dry_run,
        } => install_stable(&InstallRequest {
            source_bundle: match app_bundle {
                Some(app_bundle) => app_bundle,
                None => default_stable_bundle(repository_root),
            },
            is_dry_run,
        }),
        ReleaseCommand::BuildAndInstallStable { is_dry_run } => {
            build_and_install_stable(repository_root, is_dry_run)
        }
        ReleaseCommand::AppStore(app_store_command) => run_app_store_command(&app_store_command),
        ReleaseCommand::PrepareAndPublish(publication_request) => {
            run_prepare_and_publish(repository_root, &publication_request)
        }
        ReleaseCommand::PrepareSigned(publication_request) => {
            run_prepare_signed(repository_root, &publication_request)
        }
        ReleaseCommand::AcceptSparkleUpdate => accept_sparkle_update(repository_root),
    }
}

fn required_path(flag_values: &BTreeMap<String, String>, flag: &str) -> Result<PathBuf, String> {
    match flag_values.get(flag) {
        Some(flag_value) => Ok(PathBuf::from(flag_value)),
        None => Err(format!("{flag} is required\n{RELEASE_USAGE}")),
    }
}

/// `--flag value` pairs; each flag at most once, and a value can never be
/// another flag, so a forgotten value fails instead of shifting the rest.
pub(crate) fn parse_flag_values(
    flag_arguments: &[String],
    accepted_flags: &[&str],
) -> Result<BTreeMap<String, String>, String> {
    let mut flag_values = BTreeMap::new();
    let mut argument_position = 0;
    while argument_position < flag_arguments.len() {
        let flag = flag_arguments[argument_position].as_str();
        if !accepted_flags.contains(&flag) {
            return Err(format!("unrecognized argument: {flag}\n{RELEASE_USAGE}"));
        }
        let flag_value = match flag_arguments.get(argument_position + 1) {
            Some(flag_value) if !flag_value.is_empty() && !flag_value.starts_with("--") => {
                flag_value
            }
            _ => return Err(format!("{flag} requires a value")),
        };
        if flag_values
            .insert(flag.to_owned(), flag_value.clone())
            .is_some()
        {
            return Err(format!("{flag} may be supplied only once"));
        }
        argument_position += 2;
    }
    Ok(flag_values)
}

#[cfg(test)]
#[path = "command_tests.rs"]
mod command_tests;
