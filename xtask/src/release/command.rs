//! Argument parsing for `cargo xtask release`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::disk_image_creation::{DiskImageRequest, create_disk_image};
use super::disk_image_validation::validate_disk_image;

pub(crate) const RELEASE_USAGE: &str = "usage: cargo xtask release create-dmg --app-bundle PATH --output PATH | release validate-dmg --dmg PATH";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReleaseCommand {
    CreateDiskImage(DiskImageRequest),
    ValidateDiskImage(PathBuf),
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
        _ => Err(RELEASE_USAGE.to_owned()),
    }
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    match parse_release_command(arguments)? {
        ReleaseCommand::CreateDiskImage(disk_image_request) => {
            create_disk_image(repository_root, &disk_image_request)
        }
        ReleaseCommand::ValidateDiskImage(disk_image_path) => validate_disk_image(&disk_image_path),
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
