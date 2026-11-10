//! Contracts for finding, quitting and reopening the running applications of
//! a channel: the command-line patterns `pgrep -f` matches, and the agent a
//! restart brings back.

use std::path::Path;

use super::channel_profile::{Channel, ChannelProfile};
use super::instance_control::reopen_agent_if_it_was_running;
use super::running_instance::agent_process_pattern;

/// A command line as `pgrep -f` sees it, from a channel's bundle built in
/// `worktree_name`'s target directory.
fn bundle_command_line(worktree_name: &str, channel: Channel, executable_name: &str) -> String {
    let channel_profile = ChannelProfile::for_channel(channel);
    format!(
        "/repository/.claude/worktrees/{worktree_name}/target/{}/{}.app/{}",
        channel_profile.output_directory_name,
        channel_profile.bundle_name,
        executable_relative_path(&channel_profile, executable_name)
    )
}

/// The agent runs from its nested bundle; everything else from Contents/MacOS.
fn executable_relative_path(channel_profile: &ChannelProfile, executable_name: &str) -> String {
    if executable_name == "astronomical-menu" {
        return channel_profile.agent_executable_relative_path();
    }
    format!("Contents/MacOS/{executable_name}")
}

#[test]
fn should_match_the_development_menu_app_from_any_worktree() {
    let development_pattern =
        agent_process_pattern(&ChannelProfile::for_channel(Channel::Development));

    for worktree_name in ["first-worktree", "second-worktree"] {
        let command_line =
            bundle_command_line(worktree_name, Channel::Development, "astronomical-menu");
        assert!(
            command_line.contains(&development_pattern),
            "{command_line}"
        );
    }
}

#[test]
fn should_not_match_the_development_menu_app_with_the_stable_pattern() {
    let stable_pattern = agent_process_pattern(&ChannelProfile::for_channel(Channel::Stable));
    let development_command_line =
        bundle_command_line("any-worktree", Channel::Development, "astronomical-menu");

    assert!(!development_command_line.contains(&stable_pattern));
}

#[test]
fn should_keep_stable_and_app_store_menu_apps_apart_despite_one_bundle_name() {
    let stable_pattern = agent_process_pattern(&ChannelProfile::for_channel(Channel::Stable));
    let stable_command_line =
        bundle_command_line("any-worktree", Channel::Stable, "astronomical-menu");
    let app_store_command_line =
        bundle_command_line("any-worktree", Channel::AppStore, "astronomical-menu");

    assert!(stable_command_line.contains(&stable_pattern));
    assert!(!app_store_command_line.contains(&stable_pattern));
}

#[test]
fn should_not_match_the_daemon_or_worker_inside_the_same_bundle() {
    let development_pattern =
        agent_process_pattern(&ChannelProfile::for_channel(Channel::Development));

    for executable_name in ["astronomicald", "astronomical-inference-worker"] {
        let command_line =
            bundle_command_line("any-worktree", Channel::Development, executable_name);
        assert!(
            !command_line.contains(&development_pattern),
            "{command_line}"
        );
    }
}

fn open_agent_succeeds(_bundle_path: &Path) -> Result<(), String> {
    Ok(())
}

fn open_agent_fails(_bundle_path: &Path) -> Result<(), String> {
    Err("open refused".to_owned())
}

#[test]
fn should_reopen_a_menu_app_that_was_running() {
    let reopen_outcome = reopen_agent_if_it_was_running(
        true,
        Path::new("/fictional/Fictional.app"),
        open_agent_succeeds,
    );

    assert_eq!(reopen_outcome, Ok(()));
}

#[test]
fn should_not_open_a_menu_app_that_was_not_running() {
    // The opener always fails, so an Ok here proves it was never called.
    let reopen_outcome = reopen_agent_if_it_was_running(
        false,
        Path::new("/fictional/Fictional.app"),
        open_agent_fails,
    );

    assert_eq!(reopen_outcome, Ok(()));
}

#[test]
fn should_say_the_daemon_is_running_when_the_menu_app_cannot_reopen() {
    let reopen_outcome = reopen_agent_if_it_was_running(
        true,
        Path::new("/fictional/Fictional.app"),
        open_agent_fails,
    );

    assert_eq!(
        reopen_outcome,
        Err(
            "the daemon is running, but the menu-bar agent did not reopen: open refused".to_owned()
        )
    );
}
