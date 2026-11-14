//! Contracts of the smoke test's login-style launch: which label the agent
//! is started under, what launch shape that produces, and how the wait for
//! the engine ends when the agent gives up.

use std::path::Path;
use std::process::{Command, Stdio};

use super::login_launch_check::{
    LoginLaunchPlan, login_launch_command, login_launch_plan, sandboxed_location_hint,
    wait_for_agent_started_engine,
};
use super::smoke_test::BundleInstanceIdentity;

const AGENT_EXECUTABLE: &str = "/Applications/Example.app/Contents/Library/LoginItems/Example Agent.app/Contents/MacOS/astronomical-menu";
const AGENT_LABEL: &str = "dev.example.app.agent";
/// Nothing listens on the discard port of the loopback interface, so every
/// health request to it is refused at once.
const UNOCCUPIED_ENDPOINT: &str = "http://127.0.0.1:9";

fn sample_instance_identity() -> BundleInstanceIdentity {
    BundleInstanceIdentity {
        channel_name: "stable".to_owned(),
        runtime_instance_name: "stable".to_owned(),
        is_sandboxed: false,
        daemon_port: 9,
        application_version: "0.2.88".to_owned(),
        build_commit: "49af41bd9bba".to_owned(),
    }
}

#[test]
fn should_plan_a_login_launch_under_the_agent_bundle_identifier() {
    let agent_plist_document = serde_json::json!({"CFBundleIdentifier": AGENT_LABEL});

    assert_eq!(
        login_launch_plan(Path::new(AGENT_EXECUTABLE), &agent_plist_document),
        Ok(LoginLaunchPlan {
            agent_executable: AGENT_EXECUTABLE.into(),
            service_label: AGENT_LABEL.to_owned(),
        })
    );
}

#[test]
fn should_refuse_a_plist_that_names_no_bundle_identifier() {
    for agent_plist_document in [
        serde_json::json!({}),
        serde_json::json!({"CFBundleIdentifier": 7}),
        serde_json::json!({"CFBundleIdentifier": null}),
    ] {
        let plan_error = login_launch_plan(Path::new(AGENT_EXECUTABLE), &agent_plist_document)
            .expect_err("a plist without a text identifier gives no label");

        assert!(plan_error.contains("no bundle identifier"));
    }
}

/// A label with a path separator could be taken for the agent's real path,
/// and the launch would then no longer be the one macOS performs at login.
#[test]
fn should_refuse_a_label_that_is_empty_or_could_pass_for_a_path() {
    for unusable_label in ["", "/", AGENT_EXECUTABLE, "dev.example/agent"] {
        let agent_plist_document = serde_json::json!({"CFBundleIdentifier": unusable_label});

        let plan_error = login_launch_plan(Path::new(AGENT_EXECUTABLE), &agent_plist_document)
            .expect_err("an empty or path-like identifier is not a login label");

        assert!(plan_error.contains("not a usable login label"));
    }
}

/// The launch moves to the root directory, so a path relative to the place
/// the smoke test was run from would point somewhere else.
#[test]
fn should_refuse_an_agent_path_that_is_relative() {
    let agent_plist_document = serde_json::json!({"CFBundleIdentifier": AGENT_LABEL});

    let plan_error = login_launch_plan(
        Path::new("target/Example.app/Contents/Library/LoginItems/Agent.app/Contents/MacOS/astronomical-menu"),
        &agent_plist_document,
    )
    .expect_err("a relative agent path cannot be launched from the root directory");

    assert!(plan_error.contains("needs the agent's absolute path"));
}

/// Runs a shell under the plan and has it report what a login-launched agent
/// sees: its first launch argument and its working directory.
#[test]
fn should_start_the_process_under_its_label_from_the_root_directory() {
    let shell_launch_plan = LoginLaunchPlan {
        agent_executable: "/bin/bash".into(),
        service_label: AGENT_LABEL.to_owned(),
    };

    let shell_output = login_launch_command(&shell_launch_plan)
        .args(["-c", r#"printf '%s\n' "$0"; pwd"#])
        .output()
        .expect("the shell should run");

    assert_eq!(
        String::from_utf8_lossy(&shell_output.stdout),
        format!("{AGENT_LABEL}\n/\n")
    );
}

/// The executable is still the real file: only the first launch argument is
/// replaced, so the kernel loads the agent and not a file named by the label.
#[test]
fn should_keep_the_agent_executable_as_the_program_to_run() {
    let launch_command = login_launch_command(&LoginLaunchPlan {
        agent_executable: AGENT_EXECUTABLE.into(),
        service_label: AGENT_LABEL.to_owned(),
    });

    assert_eq!(launch_command.get_program(), AGENT_EXECUTABLE);
    assert_eq!(launch_command.get_current_dir(), Some(Path::new("/")));
    assert_eq!(launch_command.get_args().count(), 0);
}

/// An agent that exits can no longer start an engine, so the wait must end
/// with that reason instead of running out its thirty-second budget.
#[test]
fn should_stop_waiting_for_the_engine_once_the_agent_has_exited() {
    let mut exited_agent_process = Command::new("/usr/bin/true")
        .stdin(Stdio::null())
        .spawn()
        .expect("the stand-in agent should start");
    exited_agent_process
        .wait()
        .expect("the stand-in agent should exit");
    let wait_started_at = std::time::Instant::now();

    let wait_error = wait_for_agent_started_engine(
        &mut exited_agent_process,
        &sample_instance_identity(),
        UNOCCUPIED_ENDPOINT,
        Path::new("/nonexistent/agent-login-launch.log"),
    )
    .expect_err("an exited agent started no engine");

    assert!(wait_error.contains("exited after a login-style launch without starting the engine"));
    assert!(wait_started_at.elapsed().as_secs() < 10);
}

/// Only a sandboxed bundle's timeout points at the Applications folder; a
/// direct bundle starts its engine from anywhere, so the hint would mislead.
#[test]
fn should_hint_at_the_applications_folder_only_for_a_sandboxed_bundle() {
    let mut sandboxed_identity = sample_instance_identity();
    sandboxed_identity.is_sandboxed = true;

    assert!(sandboxed_location_hint(&sandboxed_identity).contains("Applications folder"));
    assert_eq!(sandboxed_location_hint(&sample_instance_identity()), "");
}
