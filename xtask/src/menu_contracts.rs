//! `cargo xtask menu-contracts` — runs the macOS menu app's Swift contract
//! tests the way CI does, for local verification. Typed port of the retired
//! `scripts/ci/test-macos-menu-contracts.sh`; CI runs the same `swift test`
//! inline because its Swift job carries no Rust toolchain.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::step_log::start_step;

/// The menu contracts finish well inside this on a warm build; a hang is a
/// defect, not a slow test.
const MENU_CONTRACT_BUDGET: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("menu-contracts does not accept arguments".to_owned());
    }
    let contract_step = start_step("macos-menu-contract-tests");
    let mut swift_command = menu_contract_command(repository_root);
    let mut swift_process = match swift_command.spawn() {
        Ok(swift_process) => swift_process,
        Err(spawn_error) => return Err(format!("swift could not start: {spawn_error}")),
    };
    let started_at = Instant::now();
    loop {
        match swift_process.try_wait() {
            Ok(Some(exit_status)) if exit_status.success() => break,
            Ok(Some(exit_status)) => {
                return Err(format!("the menu contracts failed with {exit_status}"));
            }
            Ok(None) if started_at.elapsed() < MENU_CONTRACT_BUDGET => {
                std::thread::sleep(POLL_INTERVAL)
            }
            Ok(None) => {
                let _ = swift_process.kill();
                return Err(format!(
                    "the menu contracts exceeded {} seconds",
                    MENU_CONTRACT_BUDGET.as_secs()
                ));
            }
            Err(wait_error) => return Err(format!("could not wait for swift: {wait_error}")),
        }
    }
    contract_step.finish("success");
    Ok(())
}

/// A verification run must never touch stored credentials: the menu
/// package's Sparkle dependency is public and resolves anonymously, so the
/// Git credential helper is cleared and a fresh checkout never prompts for
/// the developer keychain. Warnings fail the build, as Rust warnings do.
pub(crate) fn menu_contract_command(repository_root: &Path) -> Command {
    let mut swift_command = Command::new("swift");
    swift_command
        .args(["test", "--package-path"])
        .arg(repository_root.join("apps/astronomical-menu"))
        .args(["-Xswiftc", "-warnings-as-errors"])
        .env("GIT_CONFIG_COUNT", "1")
        .env("GIT_CONFIG_KEY_0", "credential.helper")
        .env("GIT_CONFIG_VALUE_0", "");
    swift_command
}
