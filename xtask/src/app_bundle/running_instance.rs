//! Stops the running instance of a channel before its bundle is replaced,
//! and opens the new bundle afterwards.
//!
//! The main app and its menu-bar agent are asked to quit through Apple
//! Events and the daemon through its shutdown endpoint, the same doors a
//! person uses. Nothing is
//! ever sent a signal: an instance that ignores both requests is reported,
//! because killing it could lose a model write in progress.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use super::channel_profile::{APPLICATION_EXECUTABLE_NAME, ChannelProfile};
use super::command::DevOpenPlacement;
use super::tool_command::run_quiet_tool;

const APPLICATION_QUIT_WAIT: Duration = Duration::from_secs(15);
/// Quit rounds per application before giving up: one per concurrently
/// running worktree bundle of the channel, with headroom for parallel sessions.
const MAXIMUM_QUIT_ROUNDS: usize = 4;
/// The daemon may still be releasing a model's wired GPU memory when it is
/// asked to stop, so it gets longer than the two applications do.
const DAEMON_EXIT_WAIT: Duration = Duration::from_secs(30);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(250);
const WAIT_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// One of the bundle's two applications, as the quit needs to know it.
struct BundledApplication {
    description: &'static str,
    bundle_identifier: String,
    command_line_pattern: String,
}

/// Quits the channel's main app and menu-bar agent in every worktree's
/// bundle; returns whether an agent was running. The agent's quit handler
/// tears down the daemon it owns, so this is usually all it takes.
///
/// The main app goes first: for a few seconds after its launch it starts the
/// agent again when it finds none, which would undo an agent quit sent first.
pub(crate) fn quit_bundled_applications(channel_profile: &ChannelProfile) -> Result<bool, String> {
    quit_bundled_application(&BundledApplication {
        description: "main app",
        bundle_identifier: channel_profile.bundle_identifier.to_owned(),
        command_line_pattern: application_process_pattern(channel_profile),
    })?;
    quit_bundled_application(&BundledApplication {
        description: "menu-bar agent",
        bundle_identifier: channel_profile.agent_bundle_identifier(),
        command_line_pattern: login_or_path_command_line_pattern(
            &agent_process_pattern(channel_profile),
            &channel_profile.agent_bundle_identifier(),
        ),
    })
}

/// Quits every running copy of one application of the channel; returns
/// whether one was running.
///
/// The match is per channel, not per bundle path: each worktree builds its
/// own bundle, and an agent opened from another worktree would otherwise
/// stay in the menu bar with no daemon behind it once this channel's daemon
/// stops. One quit Apple Event reaches a single copy when several share the
/// bundle identifier, so the quit repeats until none is left.
fn quit_bundled_application(bundled_application: &BundledApplication) -> Result<bool, String> {
    let process_pattern = &bundled_application.command_line_pattern;
    let was_running = process_is_running(process_pattern);
    for _quit_round in 0..MAXIMUM_QUIT_ROUNDS {
        let running_copy_count = running_process_count(process_pattern);
        if running_copy_count == 0 {
            return Ok(was_running);
        }
        run_quiet_tool(Command::new("osascript").args([
            "-e",
            &format!(
                "tell application id \"{}\" to quit",
                bundled_application.bundle_identifier
            ),
        ]))?;
        wait_for_process_count_at_most(
            process_pattern,
            bundled_application.description,
            APPLICATION_QUIT_WAIT,
            running_copy_count - 1,
        )?;
    }
    if process_is_running(process_pattern) {
        return Err(format!(
            "{MAXIMUM_QUIT_ROUNDS} quit requests left a {} of this channel running (process state: {}); quit it yourself and rerun",
            bundled_application.description,
            process_states(process_pattern)
        ));
    }
    Ok(was_running)
}

/// The command-line fragment shared by the channel's menu-bar agent in every
/// worktree's bundle. The output directory is part of it because Stable and
/// App Store share the bundle name "Astronomical": without it, stopping one
/// channel would wait on the other's agent, which the quit sent to this
/// channel's bundle identifier never reaches.
pub(crate) fn agent_process_pattern(channel_profile: &ChannelProfile) -> String {
    format!(
        "/{}/{}.app/{}",
        channel_profile.output_directory_name,
        channel_profile.bundle_name,
        channel_profile.agent_executable_relative_path()
    )
}

/// The `pgrep -f` pattern for an application that macOS may also start as a
/// login item: a command line holding `executable_path_fragment`, or one that
/// is nothing but `service_label`.
///
/// launchd starts a login item with its service label, which is its bundle
/// identifier, as the whole command line. Matched by path alone, an agent
/// started that way was never seen: a stop reported the instance as stopped
/// and left the agent holding the channel's single-agent lock. `ps` offers no
/// way round the command line, because on macOS its `comm` column also
/// reports the first launch argument and not the file the kernel loaded.
/// Both parts are escaped, so the dots of a bundle identifier match only
/// dots and a longer identifier of another channel is not taken for this one.
pub(crate) fn login_or_path_command_line_pattern(
    executable_path_fragment: &str,
    service_label: &str,
) -> String {
    format!(
        "{}|^{}$",
        escape_extended_regular_expression(executable_path_fragment),
        escape_extended_regular_expression(service_label)
    )
}

/// `literal_text` as an extended regular expression that matches only itself.
pub(crate) fn escape_extended_regular_expression(literal_text: &str) -> String {
    let mut escaped_expression = String::with_capacity(literal_text.len());
    for literal_character in literal_text.chars() {
        if r".[]()*+?{}|^$\".contains(literal_character) {
            escaped_expression.push('\\');
        }
        escaped_expression.push(literal_character);
    }
    escaped_expression
}

/// The same fragment for the channel's main app.
pub(crate) fn application_process_pattern(channel_profile: &ChannelProfile) -> String {
    format!(
        "/{}/{}.app/Contents/MacOS/{APPLICATION_EXECUTABLE_NAME}",
        channel_profile.output_directory_name, channel_profile.bundle_name
    )
}

/// Waits for the channel's daemon to release its port before the new
/// bundle opens. Runs after the swap: macOS lets a bundle be renamed while
/// its processes run, so only the launch, not the swap, waits on the daemon.
pub(crate) fn wait_for_daemon_exit(channel_profile: &ChannelProfile) -> Result<(), String> {
    let daemon_pattern = daemon_process_pattern(channel_profile);
    if !process_is_running(&daemon_pattern) {
        return Ok(());
    }
    // A daemon the agent did not own survives the agent's quit; ask it
    // through the same endpoint the agent uses. A refused request is fine as
    // long as the process exits.
    let _ = Command::new("curl")
        .args(["--silent", "--max-time", "3", "--request", "POST"])
        .arg(format!(
            "http://127.0.0.1:{}/v1/control/shutdown",
            channel_profile.daemon_port
        ))
        .output();
    wait_for_exit(&daemon_pattern, "daemon", DAEMON_EXIT_WAIT)
}

/// The launch argument that tells the app to show its window behind every
/// other window. `open -g` alone is not enough: it launches the app inactive,
/// but the app brings itself forward when it shows its window unless told
/// not to. The app reads the same text in `ChatLaunchPlacement.swift`.
pub(crate) const LAUNCH_IN_BACKGROUND_ARGUMENT: &str = "--launch-in-background";

/// The `open` arguments for the main app, which shows the chat window and
/// starts its agent.
pub(crate) fn open_bundle_arguments(
    bundle_path: &Path,
    open_placement: DevOpenPlacement,
) -> Vec<std::ffi::OsString> {
    match open_placement {
        DevOpenPlacement::InFront => vec![bundle_path.into()],
        DevOpenPlacement::InBackground => vec![
            "-g".into(),
            bundle_path.into(),
            "--args".into(),
            LAUNCH_IN_BACKGROUND_ARGUMENT.into(),
        ],
    }
}

pub(crate) fn open_bundle(
    bundle_path: &Path,
    open_placement: DevOpenPlacement,
) -> Result<(), String> {
    run_quiet_tool(Command::new("open").args(open_bundle_arguments(bundle_path, open_placement)))
        .map(drop)
}

/// Opens only the menu-bar agent, in the background: the status item comes
/// back without a chat window taking the screen.
pub(crate) fn open_agent(agent_bundle_path: &Path) -> Result<(), String> {
    run_quiet_tool(Command::new("open").arg("-g").arg(agent_bundle_path)).map(drop)
}

pub(crate) fn daemon_process_pattern(channel_profile: &ChannelProfile) -> String {
    format!(
        "astronomicald --instance {}",
        channel_profile.channel.as_channel_name()
    )
}

fn process_is_running(command_line_pattern: &str) -> bool {
    running_process_count(command_line_pattern) > 0
}

pub(crate) fn running_process_count(command_line_pattern: &str) -> usize {
    match Command::new("pgrep")
        .args(["-f", command_line_pattern])
        .output()
    {
        Ok(pgrep_output) => String::from_utf8_lossy(&pgrep_output.stdout)
            .lines()
            .count(),
        Err(_) => 0,
    }
}

fn wait_for_exit(
    command_line_pattern: &str,
    process_description: &str,
    exit_wait: Duration,
) -> Result<(), String> {
    wait_for_process_count_at_most(command_line_pattern, process_description, exit_wait, 0)
}

/// Waits until at most `remaining_process_count` processes match, so a quit
/// sent to one of several copies can wait for just that copy.
fn wait_for_process_count_at_most(
    command_line_pattern: &str,
    process_description: &str,
    exit_wait: Duration,
    remaining_process_count: usize,
) -> Result<(), String> {
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    while running_process_count(command_line_pattern) > remaining_process_count {
        if wait_started_at.elapsed() >= exit_wait {
            return Err(format!(
                "the running {process_description} did not exit within {}s (process state: {}); quit it yourself and rerun",
                exit_wait.as_secs(),
                process_states(command_line_pattern)
            ));
        }
        if last_progress_at.elapsed() >= WAIT_PROGRESS_INTERVAL {
            println!(
                "[app] waiting for the {process_description} to exit: {:.0}s of {}s",
                wait_started_at.elapsed().as_secs_f64(),
                exit_wait.as_secs()
            );
            last_progress_at = Instant::now();
        }
        std::thread::sleep(STOP_POLL_INTERVAL);
    }
    println!(
        "[app] stopped the running {process_description} in {:.1}s",
        wait_started_at.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `pid stat` of every matching process, so a timeout says whether the
/// process was a zombie, stuck exiting, or still serving.
fn process_states(command_line_pattern: &str) -> String {
    let Ok(pgrep_output) = Command::new("pgrep")
        .args(["-f", command_line_pattern])
        .output()
    else {
        return "unknown".to_owned();
    };
    let matching_process_ids = String::from_utf8_lossy(&pgrep_output.stdout)
        .trim()
        .replace('\n', ",");
    if matching_process_ids.is_empty() {
        return "exited".to_owned();
    }
    match Command::new("ps")
        .args(["-o", "pid=,stat=", "-p", &matching_process_ids])
        .output()
    {
        Ok(ps_output) => String::from_utf8_lossy(&ps_output.stdout)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
        Err(_) => matching_process_ids,
    }
}
