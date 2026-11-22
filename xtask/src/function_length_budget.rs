//! `cargo xtask function-length-budget` — the pre-commit hook's workspace
//! clippy pass, which enforces the 100-line function budget (`clippy.toml`,
//! issue #1637), made aware of the host operating system.
//!
//! On macOS the pass is the same `cargo clippy --workspace --all-targets` the
//! hook always ran. On another host it leaves out the crates that host cannot
//! build, as their manifests declare (see `host_platforms`).

use std::path::Path;
use std::process::Command;

use crate::host_platforms::packages_unbuildable_on_this_host;
use crate::step_log::{log_line, start_step};

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("function-length-budget does not accept arguments".to_owned());
    }
    let packages_left_out = packages_unbuildable_on_this_host(repository_root)?;
    if !packages_left_out.is_empty() {
        log_line(&format!(
            "function-length-budget host={} left_out={}",
            std::env::consts::OS,
            packages_left_out.join(",")
        ));
    }
    let budget_step = start_step("function-length-budget");
    let clippy_status =
        match function_length_clippy_command(repository_root, &packages_left_out).status() {
            Ok(clippy_status) => clippy_status,
            Err(spawn_error) => return Err(format!("cargo could not start: {spawn_error}")),
        };
    if !clippy_status.success() {
        budget_step.finish("failed");
        return Err(format!(
            "the workspace clippy pass failed ({clippy_status})"
        ));
    }
    budget_step.finish("success");
    Ok(())
}

/// The workspace clippy pass, without the crates this host cannot build.
pub(crate) fn function_length_clippy_command(
    repository_root: &Path,
    packages_left_out: &[String],
) -> Command {
    let mut clippy_command = Command::new("cargo");
    clippy_command
        .args(["clippy", "--workspace", "--all-targets"])
        .current_dir(repository_root);
    for package_name in packages_left_out {
        clippy_command.args(["--exclude", package_name]);
    }
    clippy_command
}
