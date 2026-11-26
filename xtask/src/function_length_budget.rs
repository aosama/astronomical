//! `cargo xtask function-length-budget` — the pre-commit hook's workspace
//! clippy pass, which enforces the 80-line function budget (`clippy.toml`,
//! issue #1637; the limit AGENTS.md sets), made aware of the host operating
//! system.
//!
//! Before compiling anything it refuses any `#[allow]` or `#[expect]` of
//! `clippy::too_many_lines` in production or test code (see
//! `function_length_exemptions`), since such an attribute would let a long
//! function through the clippy pass.
//!
//! On macOS the pass is the same `cargo clippy --workspace --all-targets` the
//! hook always ran. On another host it leaves out the crates that host cannot
//! build, as their manifests declare (see `host_platforms`).

use std::path::Path;
use std::process::Command;

use crate::function_length_exemptions::{
    FUNCTION_LENGTH_BUDGET_CODE_LINES, exemption_report_lines, find_function_length_exemptions,
};
use crate::host_platforms::packages_unbuildable_on_this_host;
use crate::step_log::{log_line, start_step};

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("function-length-budget does not accept arguments".to_owned());
    }
    refuse_function_length_exemptions(repository_root)?;
    run_workspace_clippy_pass(repository_root)
}

/// The clippy pass itself, after the exemption scan has passed.
fn run_workspace_clippy_pass(repository_root: &Path) -> Result<(), String> {
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

/// Fails, naming each location, when code exempts itself from the budget;
/// the clippy pass would otherwise accept the long function.
fn refuse_function_length_exemptions(repository_root: &Path) -> Result<(), String> {
    let exemption_step = start_step("function-length-exemptions");
    let (exemptions, scanned_file_count) = find_function_length_exemptions(repository_root)?;
    log_line(&format!(
        "function-length-budget scanned_files={scanned_file_count} exemptions={}",
        exemptions.len()
    ));
    if exemptions.is_empty() {
        exemption_step.finish("success");
        return Ok(());
    }
    for report_line in exemption_report_lines(&exemptions) {
        eprintln!("{report_line}");
    }
    exemption_step.finish("failed");
    Err(format!(
        "{} exemption(s) from the {FUNCTION_LENGTH_BUDGET_CODE_LINES}-line function budget",
        exemptions.len()
    ))
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
