//! Argument parsing for `cargo xtask release`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::disk_image_creation::{DiskImageRequest, create_disk_image};
use super::disk_image_validation::validate_disk_image;
use super::distribution_signature::validate_distribution_app;
use super::notarization::{NotarizationRequest, notarize_disk_image, parse_timeout_seconds};
use super::stable_installation::{
    InstallRequest, build_and_install_stable, default_stable_bundle, install_stable,
};

pub(crate) const RELEASE_USAGE: &str = "usage: cargo xtask release create-dmg --app-bundle PATH --output PATH | release validate-dmg --dmg PATH | release validate-distribution-app --app-bundle PATH --team-id ID | release notarize-dmg --dmg PATH --signing-identity NAME --notary-profile PROFILE [--timeout-seconds N] | release install-stable [--app-bundle PATH] [--dry-run] | release build-and-install-stable [--dry-run]";

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
}

pub(crate) fn parse_release_command(arguments: &[String]) -> Result<ReleaseCommand, String> {
    let Some((subcommand, subcommand_arguments)) = arguments.split_first() else {
        return Err(RELEASE_USAGE.to_owned());
    };
    match subcommand.as_str() {
        "create-dmg" => {
            let flag_values =
                parse_flag_values(subcommand_arguments, &["--app-bundle", "--output"])?;
            Ok(ReleaseCommand::CreateDiskImage(DiskImageRequest {
                app_bundle: required_path(&flag_values, "--app-bundle")?,
                output_disk_image: required_path(&flag_values, "--output")?,
            }))
        }
        "validate-dmg" => {
            let flag_values = parse_flag_values(subcommand_arguments, &["--dmg"])?;
            Ok(ReleaseCommand::ValidateDiskImage(required_path(
                &flag_values,
                "--dmg",
            )?))
        }
        "validate-distribution-app" => {
            let flag_values =
                parse_flag_values(subcommand_arguments, &["--app-bundle", "--team-id"])?;
            Ok(ReleaseCommand::ValidateDistributionApp {
                app_bundle: required_path(&flag_values, "--app-bundle")?,
                team_identifier: required_text(&flag_values, "--team-id")?,
            })
        }
        "notarize-dmg" => parse_notarization_command(subcommand_arguments),
        "install-stable" => {
            let (is_dry_run, valued_arguments) = take_dry_run_flag(subcommand_arguments);
            let flag_values = parse_flag_values(&valued_arguments, &["--app-bundle"])?;
            Ok(ReleaseCommand::InstallStable {
                app_bundle: flag_values.get("--app-bundle").map(PathBuf::from),
                is_dry_run,
            })
        }
        "build-and-install-stable" => match subcommand_arguments {
            [] => Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: false }),
            [dry_run_flag] if dry_run_flag == "--dry-run" => {
                Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: true })
            }
            _ => Err(RELEASE_USAGE.to_owned()),
        },
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
    let mut is_dry_run = false;
    let mut valued_arguments = Vec::new();
    for subcommand_argument in subcommand_arguments {
        if subcommand_argument == "--dry-run" {
            is_dry_run = true;
        } else {
            valued_arguments.push(subcommand_argument.clone());
        }
    }
    (is_dry_run, valued_arguments)
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
