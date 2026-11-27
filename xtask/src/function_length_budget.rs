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
//! The pass lints the workspace packages that own the commit's staged Rust
//! files (`staged_package_scope` says why that is enough and when the whole
//! workspace is still linted); `--workspace` lints every package. On another
//! host than macOS it leaves out the crates that host cannot build, as their
//! manifests declare (see `host_platforms`).

use std::path::Path;
use std::process::Command;

use crate::function_length_exemptions::{
    FUNCTION_LENGTH_BUDGET_CODE_LINES, exemption_report_lines, find_function_length_exemptions,
};
use crate::host_platforms::packages_unbuildable_on_this_host;
use crate::staged_package_scope::{ClippyScope, clippy_scope, staged_paths, workspace_members};
use crate::step_log::{log_line, start_step};

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let scope = requested_scope(repository_root, arguments)?;
    refuse_function_length_exemptions(repository_root)?;
    run_workspace_clippy_pass(repository_root, &scope)
}

/// The packages to lint: the staged packages, or every package with
/// `--workspace`.
fn requested_scope(repository_root: &Path, arguments: &[String]) -> Result<ClippyScope, String> {
    if arguments.is_empty() {
        return staged_scope(repository_root);
    }
    if arguments == ["--workspace"] {
        Ok(ClippyScope::Workspace)
    } else {
        Err("function-length-budget accepts only --workspace".to_owned())
    }
}

fn staged_scope(repository_root: &Path) -> Result<ClippyScope, String> {
    let staged_paths = staged_paths(repository_root)?;
    Ok(clippy_scope(
        &staged_paths,
        &workspace_members(repository_root)?,
    ))
}

/// The clippy pass itself, after the exemption scan has passed.
fn run_workspace_clippy_pass(repository_root: &Path, scope: &ClippyScope) -> Result<(), String> {
    let packages_left_out = packages_unbuildable_on_this_host(repository_root)?;
    if leaves_nothing_to_lint(scope, &packages_left_out) {
        return Ok(());
    }
    log_packages_left_out(&packages_left_out);
    let budget_step = start_step("function-length-budget");
    let clippy_status =
        match function_length_clippy_command(repository_root, scope, &packages_left_out).status() {
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

/// Names the crates this host cannot build, when there are any.
fn log_packages_left_out(packages_left_out: &[String]) {
    if !packages_left_out.is_empty() {
        log_line(&format!(
            "function-length-budget host={} left_out={}",
            std::env::consts::OS,
            packages_left_out.join(",")
        ));
    }
}

/// Logs the scope, and whether every staged package is one this host cannot
/// build, which leaves the pass nothing to lint.
fn leaves_nothing_to_lint(scope: &ClippyScope, packages_left_out: &[String]) -> bool {
    log_line(&format!(
        "function-length-budget scope={}",
        scope_text(scope)
    ));
    let ClippyScope::Packages(package_names) = scope else {
        return false;
    };
    let nothing_builds = package_names
        .iter()
        .all(|package_name| packages_left_out.contains(package_name));
    if nothing_builds {
        log_line(
            "function-length-budget status=skipped reason=no-staged-package-builds-on-this-host",
        );
    }
    nothing_builds
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
    scope: &ClippyScope,
    packages_left_out: &[String],
) -> Command {
    let mut clippy_command = Command::new("cargo");
    clippy_command
        .arg("clippy")
        .args(clippy_arguments(scope, packages_left_out))
        .current_dir(repository_root);
    clippy_command
}

fn clippy_arguments(scope: &ClippyScope, packages_left_out: &[String]) -> Vec<String> {
    match scope {
        ClippyScope::Workspace => workspace_arguments(packages_left_out),
        ClippyScope::Packages(package_names) => package_arguments(package_names, packages_left_out),
    }
}

fn workspace_arguments(packages_left_out: &[String]) -> Vec<String> {
    let mut arguments = vec!["--workspace".to_owned(), "--all-targets".to_owned()];
    for package_name in packages_left_out {
        arguments.extend(["--exclude".to_owned(), package_name.clone()]);
    }
    arguments
}

fn package_arguments(package_names: &[String], packages_left_out: &[String]) -> Vec<String> {
    let mut arguments = vec!["--all-targets".to_owned()];
    for package_name in package_names {
        if !packages_left_out.contains(package_name) {
            arguments.extend(["-p".to_owned(), package_name.clone()]);
        }
    }
    arguments
}

fn scope_text(scope: &ClippyScope) -> String {
    match scope {
        ClippyScope::Workspace => "workspace".to_owned(),
        ClippyScope::Packages(package_names) => package_names.join(","),
    }
}
