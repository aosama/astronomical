//! Argument parsing for `cargo xtask app`.

use std::path::{Path, PathBuf};

use super::build_identity::{BuildIdentity, github_pages_feed_url};
use super::channel_profile::{Channel, ChannelProfile};
use super::dev_pipeline::run_dev_pipeline;
use super::info_plist::render_info_plist;
use super::instance_control::{InstanceAction, run_instance_action};
use super::package_pipeline::{PackageRequest, run_package, validate_package_request};
use super::smoke_test::{SmokeTestRequest, run_smoke_test};

pub(crate) const APP_USAGE: &str = "usage: cargo xtask app dev | app instance status|start|stop|restart | app package --channel stable|app-store [--signing-identity NAME] | app smoke-test --app-bundle PATH [--real-model MODEL_ID] | app info-plist --channel development|stable|app-store";

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AppCommand {
    /// Rebuild the Development app, swap it in, and open it.
    Dev,
    /// Print the Info.plist a channel would get from this checkout, so the
    /// channel profiles can be inspected without building anything.
    InfoPlist(Channel),
    /// Control the Development instance headless from the shell.
    Instance(InstanceAction),
    /// Build a release bundle from a clean commit.
    Package(PackageRequest),
    /// Launch a built bundle in isolation and check what it serves.
    SmokeTest(SmokeTestRequest),
}

pub(crate) fn parse_app_command(arguments: &[String]) -> Result<AppCommand, String> {
    let Some((subcommand, subcommand_arguments)) = arguments.split_first() else {
        return Err(APP_USAGE.to_owned());
    };
    match (subcommand.as_str(), subcommand_arguments) {
        ("dev", []) => Ok(AppCommand::Dev),
        ("instance", [action_name]) => Ok(AppCommand::Instance(InstanceAction::from_action_name(
            action_name,
        )?)),
        ("info-plist", [channel_flag, channel_name]) if channel_flag == "--channel" => Ok(
            AppCommand::InfoPlist(Channel::from_channel_name(channel_name)?),
        ),
        ("package", package_arguments) => Ok(AppCommand::Package(parse_package_request(
            package_arguments,
        )?)),
        ("smoke-test", smoke_test_arguments) => Ok(AppCommand::SmokeTest(
            parse_smoke_test_request(smoke_test_arguments)?,
        )),
        _ => Err(APP_USAGE.to_owned()),
    }
}

pub(crate) fn parse_package_request(
    package_arguments: &[String],
) -> Result<PackageRequest, String> {
    let flag_values = parse_flag_values(package_arguments, &["--channel", "--signing-identity"])?;
    let Some(channel_name) = flag_values.get("--channel") else {
        return Err(format!("app package requires --channel\n{APP_USAGE}"));
    };
    let package_request = PackageRequest {
        channel: Channel::from_channel_name(channel_name)?,
        signing_identity: flag_values.get("--signing-identity").cloned(),
    };
    validate_package_request(&package_request)?;
    Ok(package_request)
}

pub(crate) fn parse_smoke_test_request(
    smoke_test_arguments: &[String],
) -> Result<SmokeTestRequest, String> {
    let flag_values = parse_flag_values(smoke_test_arguments, &["--app-bundle", "--real-model"])?;
    let Some(bundle_path) = flag_values.get("--app-bundle") else {
        return Err(format!("app smoke-test requires --app-bundle\n{APP_USAGE}"));
    };
    Ok(SmokeTestRequest {
        bundle_path: PathBuf::from(bundle_path),
        real_model_identifier: flag_values.get("--real-model").cloned(),
    })
}

/// `--flag value` pairs; each flag at most once, every value non-empty and
/// not itself a flag, so a forgotten value never swallows the next flag.
fn parse_flag_values(
    flag_arguments: &[String],
    accepted_flags: &[&str],
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut flag_values = std::collections::BTreeMap::new();
    let mut argument_position = 0;
    while argument_position < flag_arguments.len() {
        let flag = flag_arguments[argument_position].as_str();
        if !accepted_flags.contains(&flag) {
            return Err(format!("unrecognized argument: {flag}\n{APP_USAGE}"));
        }
        let flag_value = match flag_arguments.get(argument_position + 1) {
            Some(flag_value) if !flag_value.is_empty() && !flag_value.starts_with("--") => {
                flag_value
            }
            _ => return Err(format!("{flag} requires a value\n{APP_USAGE}")),
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

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    match parse_app_command(arguments)? {
        AppCommand::Dev => run_dev_pipeline(repository_root),
        AppCommand::Instance(instance_action) => {
            run_instance_action(repository_root, instance_action)
        }
        AppCommand::Package(package_request) => run_package(repository_root, &package_request),
        AppCommand::SmokeTest(smoke_test_request) => {
            run_smoke_test(repository_root, &smoke_test_request)
        }
        AppCommand::InfoPlist(channel) => {
            let build_identity = BuildIdentity::gather(repository_root)?;
            let update_feed_url = github_pages_feed_url(&build_identity.repository_url)?;
            print!(
                "{}",
                render_info_plist(
                    &ChannelProfile::for_channel(channel),
                    &build_identity,
                    &update_feed_url
                )
            );
            Ok(())
        }
    }
}
