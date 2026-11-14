//! The login-style launch check of `cargo xtask app smoke-test`: starts the
//! bundle's menu-bar agent the way macOS starts it at login and requires the
//! agent to bring the engine up by itself.
//!
//! launchd starts a login item with its service label as the first launch
//! argument and the root directory as its working directory, where every
//! other launch hands the agent its real path. An agent that reads its own
//! location from that argument cannot find the engine, and no other check
//! notices: the rest of the smoke test starts the engine itself. This is an
//! imitation of launchd, not a logout and login, so it proves how the agent
//! behaves when started this way and not that macOS registers the login item.

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::smoke_test::{
    BundleInstanceIdentity, HEALTH_WAIT, POLL_INTERVAL, PROGRESS_INTERVAL, agent_executable_path,
    http_get, http_get_json, http_post_json, read_plist_document, spawn_with_logs,
    status_identity_problem, wait_for_child_exit,
};

const LOGIN_LAUNCH_WORKING_DIRECTORY: &str = "/";
const ENGINE_STOP_WAIT: Duration = Duration::from_secs(30);
const AGENT_LOG_FILE_NAME: &str = "agent-login-launch.log";

/// How launchd starts the agent at login.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoginLaunchPlan {
    pub(crate) agent_executable: PathBuf,
    pub(crate) service_label: String,
}

/// The plan for the agent at `agent_executable`, whose bundle plist is
/// `agent_plist_document`. The label of a login item is its bundle
/// identifier. A label holding a path separator is refused: handed to the
/// agent it could pass for a real path, and the check would prove nothing.
/// A relative executable path is refused too: the launch changes the working
/// directory to the root, where that path would name a different file.
pub(crate) fn login_launch_plan(
    agent_executable: &Path,
    agent_plist_document: &serde_json::Value,
) -> Result<LoginLaunchPlan, String> {
    if !agent_executable.is_absolute() {
        return Err(format!(
            "a login-style launch needs the agent's absolute path, not {}",
            agent_executable.display()
        ));
    }
    let Some(service_label) = agent_plist_document["CFBundleIdentifier"].as_str() else {
        return Err(
            "the agent's plist has no bundle identifier to use as its login label".to_owned(),
        );
    };
    if service_label.is_empty() || service_label.contains('/') {
        return Err(format!(
            "the agent's bundle identifier is not a usable login label: {service_label:?}"
        ));
    }
    Ok(LoginLaunchPlan {
        agent_executable: agent_executable.to_path_buf(),
        service_label: service_label.to_owned(),
    })
}

/// The command that starts the agent as launchd would: the label in place of
/// the executable path as the first launch argument, from the root directory.
pub(crate) fn login_launch_command(login_launch_plan: &LoginLaunchPlan) -> Command {
    let mut launch_command = Command::new(&login_launch_plan.agent_executable);
    launch_command
        .arg0(&login_launch_plan.service_label)
        .current_dir(LOGIN_LAUNCH_WORKING_DIRECTORY);
    launch_command
}

pub(super) fn check_login_launch_starts_engine(
    bundle_path: &Path,
    instance_identity: &BundleInstanceIdentity,
    daemon_base_url: &str,
    log_directory: &Path,
) -> Result<(), String> {
    // The bundle path may be given relative to where the smoke test runs.
    let agent_executable = match std::path::absolute(agent_executable_path(bundle_path)?) {
        Ok(agent_executable) => agent_executable,
        Err(resolve_error) => {
            return Err(format!(
                "could not resolve the agent's absolute path: {resolve_error}"
            ));
        }
    };
    let Some(agent_contents_directory) = agent_executable.parent().and_then(Path::parent) else {
        return Err(format!(
            "the agent executable is not inside a bundle: {}",
            agent_executable.display()
        ));
    };
    let login_launch_plan = login_launch_plan(
        &agent_executable,
        &read_plist_document(&agent_contents_directory.join("Info.plist"))?,
    )?;
    // The teardown below asks whatever answers on this endpoint to shut down.
    // Refusing to start while something already answers keeps that request
    // from ever reaching an engine this check did not start.
    if http_get(&format!("{daemon_base_url}/health")).is_ok() {
        return Err(format!(
            "the {} endpoint is occupied again; the login-style launch will not stop an instance it did not start",
            instance_identity.channel_name
        ));
    }
    let agent_log_path = log_directory.join(AGENT_LOG_FILE_NAME);
    let mut agent_process = spawn_with_logs(
        &mut login_launch_command(&login_launch_plan),
        &agent_log_path,
    )?;
    let engine_result = wait_for_agent_started_engine(
        &mut agent_process,
        instance_identity,
        daemon_base_url,
        &agent_log_path,
    );
    let teardown_result = stop_agent_and_its_engine(&mut agent_process, daemon_base_url);
    engine_result?;
    teardown_result?;
    println!(
        "[app-smoke-test] step=login-launch label={} status=success",
        login_launch_plan.service_label
    );
    Ok(())
}

/// Waits for the engine the agent was expected to start, and requires it to
/// be this bundle's. An agent that exits ends the wait at once: nothing else
/// will start an engine, so waiting out the budget would only hide the cause.
pub(crate) fn wait_for_agent_started_engine(
    agent_process: &mut Child,
    instance_identity: &BundleInstanceIdentity,
    daemon_base_url: &str,
    agent_log_path: &Path,
) -> Result<(), String> {
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    while http_get(&format!("{daemon_base_url}/health")).is_err() {
        if !matches!(agent_process.try_wait(), Ok(None)) {
            return Err(format!(
                "the menu-bar agent exited after a login-style launch without starting the engine; read {}",
                agent_log_path.display()
            ));
        }
        if wait_started_at.elapsed() >= HEALTH_WAIT {
            return Err(format!(
                "the menu-bar agent did not start the engine within {}s of a login-style launch; read {}{}",
                HEALTH_WAIT.as_secs(),
                agent_log_path.display(),
                sandboxed_location_hint(instance_identity)
            ));
        }
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            println!(
                "[app-smoke-test] waiting for the agent to start the engine: {:.0}s of {}s",
                wait_started_at.elapsed().as_secs_f64(),
                HEALTH_WAIT.as_secs()
            );
            last_progress_at = Instant::now();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    let status_document = http_get_json(&format!("{daemon_base_url}/v1/status"))?;
    match status_identity_problem(&status_document, instance_identity) {
        Some(identity_problem) => Err(identity_problem),
        None => Ok(()),
    }
}

/// A sandboxed bundle run from a build folder did not start its engine in
/// the first store run, while the same bundle did from the Applications
/// folder (issue #2182), so a timeout on such a bundle says where to try it.
pub(crate) fn sandboxed_location_hint(instance_identity: &BundleInstanceIdentity) -> &'static str {
    if instance_identity.is_sandboxed {
        "; a sandboxed bundle may be unable to reach its engine outside the Applications folder, so copy it there and run the smoke test on that copy"
    } else {
        ""
    }
}

/// Stops what the check started. The agent goes first so that nothing is
/// left to start another engine. A terminated agent does not stop its
/// engine, which is this check's grandchild and so cannot be waited on as a
/// child: it is asked to shut down and its endpoint is watched instead.
fn stop_agent_and_its_engine(
    agent_process: &mut Child,
    daemon_base_url: &str,
) -> Result<(), String> {
    let _ = Command::new("kill")
        .args(["-TERM", &agent_process.id().to_string()])
        .status();
    let agent_exit_result = wait_for_child_exit(agent_process, "menu-bar agent");
    let _ = http_post_json(&format!("{daemon_base_url}/v1/control/shutdown"), "{}");
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    while http_get(&format!("{daemon_base_url}/health")).is_ok() {
        if wait_started_at.elapsed() >= ENGINE_STOP_WAIT {
            return Err(format!(
                "the engine the agent started still answers on {daemon_base_url} after {}s; stop it before the next run",
                ENGINE_STOP_WAIT.as_secs()
            ));
        }
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            println!(
                "[app-smoke-test] waiting for the agent's engine to stop: {:.0}s of {}s",
                wait_started_at.elapsed().as_secs_f64(),
                ENGINE_STOP_WAIT.as_secs()
            );
            last_progress_at = Instant::now();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    agent_exit_result
}
