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
/// The daemon may still be releasing a model's wired GPU memory when it is
/// asked to stop, so it gets longer than the menu app does.
const DAEMON_EXIT_WAIT: Duration = Duration::from_secs(30);
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(250);
const WAIT_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Quits the running menu app; returns whether one was running. Its quit
/// handler tears down the daemon it owns, so this is usually all it takes.
pub(crate) fn quit_menu_app(
    bundle_path: &Path,
    channel_profile: &ChannelProfile,
) -> Result<bool, String> {
    let menu_pattern = bundle_path
        .join("Contents/MacOS/astronomical-menu")
        .display()
        .to_string();
    if !process_is_running(&menu_pattern) {
        return Ok(false);
    }
    run_quiet_tool(Command::new("osascript").args([
        "-e",
        &format!(
            "tell application id \"{}\" to quit",
            channel_profile.bundle_identifier
        ),
    ]))?;
    wait_for_exit(&menu_pattern, "menu app", MENU_QUIT_WAIT)?;
    Ok(true)
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
    match Command::new("pgrep")
        .args(["-f", command_line_pattern])
        .output()
    {
        Ok(pgrep_output) => pgrep_output.status.success(),
        Err(_) => false,
    }
}

fn wait_for_exit(
    command_line_pattern: &str,
    process_description: &str,
    exit_wait: Duration,
) -> Result<(), String> {
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    while process_is_running(command_line_pattern) {
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
