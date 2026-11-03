//! Argument parsing for `cargo xtask app`.

use std::path::Path;

use super::build_identity::{BuildIdentity, github_pages_feed_url};
use super::channel_profile::{Channel, ChannelProfile};
use super::dev_pipeline::run_dev_pipeline;
use super::info_plist::render_info_plist;
use super::instance_control::{InstanceAction, run_instance_action};

pub(crate) const APP_USAGE: &str = "usage: cargo xtask app dev | app instance status|start|stop|restart | app info-plist --channel development|stable|app-store";

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AppCommand {
    /// Rebuild the Development app, swap it in, and open it.
    Dev,
    /// Print the Info.plist a channel would get from this checkout, so the
    /// channel profiles can be inspected without building anything.
    InfoPlist(Channel),
    /// Control the Development instance headless from the shell.
    Instance(InstanceAction),
}

pub(crate) fn parse_app_command(arguments: &[String]) -> Result<AppCommand, String> {
    match arguments {
        [subcommand] if subcommand == "dev" => Ok(AppCommand::Dev),
        [subcommand, action_name] if subcommand == "instance" => Ok(AppCommand::Instance(
            InstanceAction::from_action_name(action_name)?,
        )),
        [subcommand, channel_flag, channel_name]
            if subcommand == "info-plist" && channel_flag == "--channel" =>
        {
            Ok(AppCommand::InfoPlist(Channel::from_channel_name(
                channel_name,
            )?))
        }
        _ => Err(APP_USAGE.to_owned()),
    }
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    match parse_app_command(arguments)? {
        AppCommand::Dev => run_dev_pipeline(repository_root),
        AppCommand::Instance(instance_action) => {
            run_instance_action(repository_root, instance_action)
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
