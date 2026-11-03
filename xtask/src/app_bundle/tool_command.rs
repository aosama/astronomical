//! Runs one external tool with its own output streaming through, printing
//! the step heartbeat every five seconds until it exits.
//!
//! Polling `try_wait` on the calling thread keeps the heartbeat beside the
//! child without a reporter thread, so a failing tool stops the pipeline at
//! the exact step that printed it.

use std::process::Command;
use std::time::{Duration, Instant};

use super::step_reporter::StepReporter;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) fn run_tool(
    tool_command: &mut Command,
    step_reporter: &StepReporter,
) -> Result<(), String> {
    let tool_description = describe_command(tool_command);
    let mut tool_process = match tool_command.spawn() {
        Ok(tool_process) => tool_process,
        Err(spawn_error) => {
            return Err(format!("could not start {tool_description}: {spawn_error}"));
        }
    };
    let mut last_heartbeat_at = Instant::now();
    loop {
        match tool_process.try_wait() {
            Ok(Some(exit_status)) if exit_status.success() => return Ok(()),
            Ok(Some(exit_status)) => {
                return Err(format!("{tool_description} failed with {exit_status}"));
            }
            Ok(None) => {}
            Err(wait_error) => {
                return Err(format!(
                    "could not wait for {tool_description}: {wait_error}"
                ));
            }
        }
        if last_heartbeat_at.elapsed() >= HEARTBEAT_INTERVAL {
            step_reporter.print_heartbeat();
            last_heartbeat_at = Instant::now();
        }
        std::thread::sleep(EXIT_POLL_INTERVAL);
    }
}

/// Runs a short tool whose output is not interesting unless it fails.
pub(crate) fn run_quiet_tool(tool_command: &mut Command) -> Result<String, String> {
    let tool_description = describe_command(tool_command);
    let tool_output = match tool_command.output() {
        Ok(tool_output) => tool_output,
        Err(spawn_error) => {
            return Err(format!("could not start {tool_description}: {spawn_error}"));
        }
    };
    if !tool_output.status.success() {
        return Err(format!(
            "{tool_description} failed with {}: {}",
            tool_output.status,
            String::from_utf8_lossy(&tool_output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&tool_output.stdout)
        .trim()
        .to_owned())
}

fn describe_command(tool_command: &Command) -> String {
    let mut command_words = vec![tool_command.get_program().to_string_lossy().into_owned()];
    for command_argument in tool_command.get_args() {
        command_words.push(command_argument.to_string_lossy().into_owned());
    }
    command_words.join(" ")
}
