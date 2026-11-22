//! `cargo xtask lint-acceptance-features` — lints the acceptance tests with
//! every feature of their crate switched on. The pre-commit hook runs it.
//!
//! The acceptance journeys sit behind features (serving-acceptance,
//! direct-mlx, performance-measurement and others), so a default-feature
//! pass never type-checks them, and no continuous-integration lane compiles
//! the crate: its journeys need an Apple-silicon graphics processor and
//! installed models. A change in a crate they only consume, such as a new
//! worker event, therefore stopped them compiling unseen twice (issues #2052
//! and #2076).
//!
//! Clippy is used rather than `cargo check` so the pass shares the dependency
//! artifacts the hook's workspace clippy pass has just produced; a plain
//! check keeps its own and would recheck every dependency. Measured with the
//! `target/` build outputs warm: half a second with nothing to recheck, three seconds after an edit
//! to a crate the tests depend on.

use std::path::Path;
use std::process::Command;

use crate::host_platforms::packages_unbuildable_on_this_host;
use crate::step_log::{log_line, start_step};

const ACCEPTANCE_PACKAGE_NAME: &str = "astronomical-acceptance";

/// The hook's entry point: the lint, unless this host cannot build the
/// acceptance crate.
pub fn run_where_buildable(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() || host_builds_acceptance_tests(repository_root)? {
        run(repository_root, arguments)
    } else {
        Ok(())
    }
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("lint-acceptance-features does not accept arguments".to_owned());
    }
    let lint_step = start_step("lint-acceptance-features");
    let lint_status = match acceptance_feature_lint_command(repository_root).status() {
        Ok(lint_status) => lint_status,
        Err(spawn_error) => return Err(format!("cargo could not start: {spawn_error}")),
    };
    if !lint_status.success() {
        lint_step.finish("failed");
        return Err(format!(
            "a feature-gated acceptance test no longer compiles or lints ({lint_status}); the change that broke it may be in a crate the test uses rather than in the test itself"
        ));
    }
    lint_step.finish("success");
    Ok(())
}

/// The acceptance crate builds only where its native engines do; on a host
/// that cannot build it (see `host_platforms`) the lint is skipped and says so.
fn host_builds_acceptance_tests(repository_root: &Path) -> Result<bool, String> {
    let packages_left_out = packages_unbuildable_on_this_host(repository_root)?;
    let is_left_out = packages_left_out
        .iter()
        .any(|left_out| left_out == ACCEPTANCE_PACKAGE_NAME);
    if is_left_out {
        log_line(&format!(
            "lint-acceptance-features host={} skipped={ACCEPTANCE_PACKAGE_NAME}",
            std::env::consts::OS
        ));
    }
    Ok(!is_left_out)
}

/// `--all-features` applies to the selected package only, so this is every
/// acceptance feature at once without naming them: a feature added to the
/// crate later is covered with no edit here.
pub(crate) fn acceptance_feature_lint_command(repository_root: &Path) -> Command {
    let mut lint_command = Command::new("cargo");
    lint_command
        .args(["clippy", "--package", ACCEPTANCE_PACKAGE_NAME])
        .args(["--tests", "--all-features"])
        .current_dir(repository_root);
    lint_command
}
