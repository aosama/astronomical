//! Stops the running instance of a channel before its bundle is replaced,
//! and opens the new bundle afterwards.
//!
//! The menu app is asked to quit through an Apple Event and the daemon
//! through its shutdown endpoint, the same doors a person uses. Nothing is
//! ever sent a signal: an instance that ignores both requests is reported,
//! because killing it could lose a model write in progress.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use super::channel_profile::ChannelProfile;
use super::tool_command::run_quiet_tool;

const MENU_QUIT_WAIT: Duration = Duration::from_secs(15);
/// Quit rounds before giving up: one per concurrently running worktree
/// bundle of the channel, with headroom for parallel sessions.
const MAXIMUM_MENU_QUIT_ROUNDS: usize = 4;
/// The daemon may still be releasing a model's wired GPU memory when it is
/// asked to stop, so it gets longer than the menu app does.
const DAEMON_EXIT_WAIT: Duration = Duration::from_secs(30);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(250);
const WAIT_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Quits every running menu app of the channel; returns whether one was
/// running. Its quit handler tears down the daemon it owns, so this is
/// usually all it takes.
///
/// The match is per channel, not per bundle path: each worktree builds its
/// own bundle, and a menu app opened from another worktree would otherwise
/// stay in the menu bar with no daemon behind it once this channel's daemon
/// stops. One quit Apple Event reaches a single copy when several share the
/// bundle identifier, so the quit repeats until none is left.
pub(crate) fn quit_menu_app(channel_profile: &ChannelProfile) -> Result<bool, String> {
    let menu_pattern = menu_process_pattern(channel_profile);
    let was_running = process_is_running(&menu_pattern);
    for _quit_round in 0..MAXIMUM_MENU_QUIT_ROUNDS {
        let running_menu_count = running_process_count(&menu_pattern);
        if running_menu_count == 0 {
            return Ok(was_running);
        }
        run_quiet_tool(Command::new("osascript").args([
            "-e",
            &format!(
                "tell application id \"{}\" to quit",
                channel_profile.bundle_identifier
            ),
        ]))?;
        wait_for_process_count_at_most(
            &menu_pattern,
            "menu app",
            MENU_QUIT_WAIT,
            running_menu_count - 1,
        )?;
    }
    if process_is_running(&menu_pattern) {
        return Err(format!(
            "{MAXIMUM_MENU_QUIT_ROUNDS} quit requests left a menu app of this channel running (process state: {}); quit it yourself and rerun",
            process_states(&menu_pattern)
        ));
    }
    Ok(was_running)
}

/// The command-line fragment shared by the channel's menu app in every
/// worktree's bundle. The output directory is part of it because Stable and
/// App Store share the bundle name "Astronomical": without it, stopping one
/// channel would wait on the other's menu app, which the quit sent to this
/// channel's bundle identifier never reaches.
pub(crate) fn menu_process_pattern(channel_profile: &ChannelProfile) -> String {
    format!(
        "/{}/{}.app/Contents/MacOS/astronomical-menu",
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
    // A daemon the menu app did not own survives the menu's quit; ask it
    // through the same endpoint the menu uses. A refused request is fine as
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

pub(crate) fn open_bundle(bundle_path: &Path) -> Result<(), String> {
    run_quiet_tool(Command::new("open").arg(bundle_path)).map(drop)
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

fn running_process_count(command_line_pattern: &str) -> usize {
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
