//! Argument parsing for `cargo xtask app`.

use std::path::{Path, PathBuf};

use super::build_identity::{BuildIdentity, github_pages_feed_url};
use super::channel_profile::{Channel, ChannelProfile};
use super::dev_pipeline::run_dev_pipeline;
use super::info_plist::{render_agent_info_plist, render_info_plist};
use super::instance_control::{InstanceAction, run_instance_action};
use super::package_pipeline::{PackageRequest, run_package, validate_package_request};
use super::smoke_test::{SmokeTestRequest, run_smoke_test};

pub(crate) const APP_USAGE: &str = "usage: cargo xtask app dev [--background] | app draft TEXT | app export-to FOLDER | app instance status|start|stop|restart | app package --channel stable|app-store [--signing-identity NAME] | app smoke-test --app-bundle PATH [--real-model MODEL_ID] | app info-plist --channel development|stable|app-store [--agent]";

/// `info-plist --channel NAME` prints the main app's plist; a trailing
/// `--agent` prints the menu-bar agent's instead.
fn parse_info_plist_command(plist_arguments: &[String]) -> Result<AppCommand, String> {
    let (channel_name, plist_subject) = match plist_arguments {
        [channel_flag, channel_name] if channel_flag == "--channel" => {
            (channel_name, PlistSubject::MainApplication)
        }
        [channel_flag, channel_name, agent_flag]
            if channel_flag == "--channel" && agent_flag == "--agent" =>
        {
            (channel_name, PlistSubject::Agent)
        }
        _ => return Err(APP_USAGE.to_owned()),
    };
    Ok(AppCommand::InfoPlist(
        Channel::from_channel_name(channel_name)?,
        plist_subject,
    ))
}

/// Which of the bundle's two plists `info-plist` prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlistSubject {
    MainApplication,
    Agent,
}

/// Where the rebuilt Development app's window appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DevOpenPlacement {
    InFront,
    /// Behind every other window, without taking the keyboard: a rebuild in
    /// the middle of other work must not interrupt it.
    InBackground,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AppCommand {
    /// Rebuild the Development app, swap it in, and open it: in front, or
    /// behind every other window for someone still working in another app.
    Dev(DevOpenPlacement),
    /// Put a draft in the running Development chat's message field, without
    /// bringing the app forward or sending it.
    Draft(String),
    /// Have the running Development chat write its next export into a folder
    /// with no save panel, which would otherwise come to the front.
    ExportTo(String),
    /// Print the Info.plist a channel would get from this checkout, so the
    /// channel profiles can be inspected without building anything.
    InfoPlist(Channel, PlistSubject),
    /// Control the Development instance headless from the shell.
    Instance(InstanceAction),
    /// Build a release bundle from a clean commit.
    Package(PackageRequest),
    /// Launch a built bundle in isolation and check what it serves.
    SmokeTest(SmokeTestRequest),
}

/// `dev` opens the rebuilt app in front; `dev --background` behind the others.
fn parse_dev_command(dev_arguments: &[String]) -> Result<AppCommand, String> {
    match dev_arguments {
        [] => Ok(AppCommand::Dev(DevOpenPlacement::InFront)),
        [background_flag] if background_flag == "--background" => {
            Ok(AppCommand::Dev(DevOpenPlacement::InBackground))
        }
        _ => Err(APP_USAGE.to_owned()),
    }
}

/// `draft TEXT` takes exactly one argument with something in it.
fn parse_draft_command(draft_arguments: &[String]) -> Result<AppCommand, String> {
    match draft_arguments {
        [draft_text] if !draft_text.trim().is_empty() => Ok(AppCommand::Draft(draft_text.clone())),
        _ => Err(APP_USAGE.to_owned()),
    }
}

/// `export-to FOLDER` takes exactly one absolute path: the app is a separate
/// process with its own working directory, so a relative one would name a
/// different folder there.
fn parse_export_to_command(export_arguments: &[String]) -> Result<AppCommand, String> {
    match export_arguments {
        [export_folder] if export_folder.starts_with('/') && export_folder.len() > 1 => {
            Ok(AppCommand::ExportTo(export_folder.clone()))
        }
        [_relative_folder] => Err(format!(
            "export-to needs an absolute folder path\n{APP_USAGE}"
        )),
        _ => Err(APP_USAGE.to_owned()),
    }
}

/// The two commands that open a background link in the Development chat.
fn parse_background_link_command(
    subcommand: &str,
    link_arguments: &[String],
) -> Result<AppCommand, String> {
    if subcommand == "draft" {
        parse_draft_command(link_arguments)
    } else {
        parse_export_to_command(link_arguments)
    }
}

pub(crate) fn parse_app_command(arguments: &[String]) -> Result<AppCommand, String> {
    let Some((subcommand, subcommand_arguments)) = arguments.split_first() else {
        return Err(APP_USAGE.to_owned());
    };
    match (subcommand.as_str(), subcommand_arguments) {
        ("dev", dev_arguments) => parse_dev_command(dev_arguments),
        ("draft" | "export-to", link_arguments) => {
            parse_background_link_command(subcommand, link_arguments)
        }
        ("instance", [action_name]) => Ok(AppCommand::Instance(InstanceAction::from_action_name(
            action_name,
        )?)),
        ("info-plist", plist_arguments) => parse_info_plist_command(plist_arguments),
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

use super::running_instance::{open_draft_link, open_export_folder_link};

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    match parse_app_command(arguments)? {
        AppCommand::Dev(open_placement) => run_dev_pipeline(repository_root, open_placement),
        AppCommand::Draft(draft_text) => open_draft_link(&draft_text),
        AppCommand::ExportTo(export_folder) => open_export_folder_link(&export_folder),
        AppCommand::Instance(instance_action) => {
            run_instance_action(repository_root, instance_action)
        }
        AppCommand::Package(package_request) => run_package(repository_root, &package_request),
        AppCommand::SmokeTest(smoke_test_request) => {
            run_smoke_test(repository_root, &smoke_test_request)
        }
        AppCommand::InfoPlist(channel, plist_subject) => {
            print_info_plist(repository_root, channel, plist_subject)
        }
    }
}

fn print_info_plist(
    repository_root: &Path,
    channel: Channel,
    plist_subject: PlistSubject,
) -> Result<(), String> {
    let build_identity = BuildIdentity::gather(repository_root)?;
    let channel_profile = ChannelProfile::for_channel(channel);
    let plist_text = match plist_subject {
        PlistSubject::MainApplication => render_info_plist(
            &channel_profile,
            &build_identity,
            &github_pages_feed_url(&build_identity.repository_url)?,
        ),
        PlistSubject::Agent => render_agent_info_plist(&channel_profile, &build_identity),
    };
    print!("{plist_text}");
    Ok(())
}
