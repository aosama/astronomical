//! Runs one named release step as an external tool, with its output
//! streaming through and the shared step lines around it, so release logs
//! read the same as when these steps were shell scripts.

use std::process::Command;

use crate::step_log::start_step;

pub(crate) fn run_system_step(
    step_name: &'static str,
    step_command: &mut Command,
) -> Result<(), String> {
    let step_timer = start_step(step_name);
    match step_command.status() {
        Ok(exit_status) if exit_status.success() => {
            step_timer.finish("success");
            Ok(())
        }
        Ok(exit_status) => {
            step_timer.finish("failed");
            Err(format!("step {step_name} failed with {exit_status}"))
        }
        Err(spawn_error) => {
            step_timer.finish("failed");
            Err(format!("step {step_name} could not start: {spawn_error}"))
        }
    }
}

/// Fails fast with the shell scripts' wording when a required tool is
/// missing, before any scratch directory or mount exists.
pub(crate) fn require_commands(command_names: &[&str]) -> Result<(), String> {
    let search_path = std::env::var("PATH").unwrap_or_default();
    for command_name in command_names {
        let mut is_available = false;
        for search_directory in search_path.split(':') {
            if !search_directory.is_empty()
                && std::path::Path::new(search_directory)
                    .join(command_name)
                    .is_file()
            {
                is_available = true;
                break;
            }
        }
        if !is_available {
            return Err(format!("required command is unavailable: {command_name}"));
        }
    }
    Ok(())
}
