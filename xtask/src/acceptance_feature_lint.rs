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
//! check keeps its own and would recheck every dependency. Measured on a warm
//! cache: half a second with nothing to recheck, three seconds after an edit
//! to a crate the tests depend on.

use std::path::Path;
use std::process::Command;

use crate::step_log::start_step;

const ACCEPTANCE_PACKAGE_NAME: &str = "astronomical-acceptance";

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
