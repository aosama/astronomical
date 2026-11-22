//! `cargo xtask model-serving-default-features` — the pre-commit hook's lint
//! of model-serving and the crates split out of it (issue #2340) on their own
//! default features (issue #1719).
//!
//! The workspace pass unifies features, which switches direct-mlx on for
//! these crates and hides dead code that only the direct-MLX engines use;
//! that drift once broke the plain `cargo test -p model-serving` lane. The
//! quantization probe is MLX-only: it turns on direct-mlx for the Qwen crate
//! and everything under it, so it is linted in its own call rather than with
//! the others, where it would switch this check's default world off.
//!
//! On macOS both calls are exactly the ones the hook ran before. On another
//! host the crates it cannot build are left out (see `host_platforms`), and a
//! call with nothing left to lint is skipped.

use std::path::Path;
use std::process::Command;

use crate::host_platforms::packages_unbuildable_on_this_host;
use crate::step_log::log_line;

/// The default-world crates, in the order the hook has always named them.
pub(crate) const DEFAULT_WORLD_PACKAGES: [&str; 8] = [
    "model-primitives",
    "speech",
    "artifact-validation",
    "image-generation",
    "engine-core",
    "qwen35",
    "decision-models",
    "model-serving",
];

pub(crate) const QUANTIZATION_PROBE_PACKAGE: &str = "qwen35-quantization-probe";

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("model-serving-default-features does not accept arguments".to_owned());
    }
    let packages_left_out = packages_unbuildable_on_this_host(repository_root)?;
    let default_world_packages =
        packages_this_host_builds(&DEFAULT_WORLD_PACKAGES, &packages_left_out);
    let probe_packages =
        packages_this_host_builds(&[QUANTIZATION_PROBE_PACKAGE], &packages_left_out);
    for package_lint in [default_world_packages, probe_packages] {
        run_package_lint(repository_root, &package_lint)?;
    }
    Ok(())
}

fn run_package_lint(repository_root: &Path, package_names: &[&str]) -> Result<(), String> {
    if package_names.is_empty() {
        log_line(&format!(
            "model-serving-default-features host={} skipped=no-buildable-package",
            std::env::consts::OS
        ));
        return Ok(());
    }
    let lint_status = match package_clippy_command(repository_root, package_names).status() {
        Ok(lint_status) => lint_status,
        Err(spawn_error) => return Err(format!("cargo could not start: {spawn_error}")),
    };
    if !lint_status.success() {
        return Err(format!(
            "clippy failed on {} ({lint_status})",
            package_names.join(", ")
        ));
    }
    Ok(())
}

/// The packages of `wanted_packages` the host can build, in their order.
pub(crate) fn packages_this_host_builds<'package>(
    wanted_packages: &[&'package str],
    packages_left_out: &[String],
) -> Vec<&'package str> {
    let mut buildable_packages = Vec::new();
    for wanted_package in wanted_packages {
        let is_left_out = packages_left_out
            .iter()
            .any(|left_out| left_out == wanted_package);
        if !is_left_out {
            buildable_packages.push(*wanted_package);
        }
    }
    buildable_packages
}

pub(crate) fn package_clippy_command(repository_root: &Path, package_names: &[&str]) -> Command {
    let mut clippy_command = Command::new("cargo");
    clippy_command.arg("clippy").current_dir(repository_root);
    for package_name in package_names {
        clippy_command.args(["-p", package_name]);
    }
    clippy_command.arg("--all-targets");
    clippy_command
}
