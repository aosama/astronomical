//! Unit tests for the doctor report: the binding line and both channels'
//! liveness verdicts must be present no matter how dead the daemons are,
//! because a diagnostic that fails alongside its subject helps nobody
//! (issue #1391).

use std::{io::Write, path::Path, path::PathBuf, time::Duration};

use astronomical_config::AstronomicalRuntimeInstance;

use crate::verbs::doctor::{DoctorDependencies, run_doctor};

fn fresh_test_directory(test_name: &str) -> PathBuf {
    let test_directory = std::env::temp_dir().join(format!(
        "astronomical-doctor-{test_name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&test_directory);
    std::fs::create_dir_all(&test_directory).unwrap();
    test_directory
}

fn doctor_dependencies<'a>(
    test_directory: &Path,
    stdout: &'a mut Vec<u8>,
) -> DoctorDependencies<'a> {
    DoctorDependencies {
        executable_path: test_directory.join("astronomical"),
        channel_socket_paths: vec![
            (
                AstronomicalRuntimeInstance::Stable,
                test_directory.join("stable-daemon.sock"),
            ),
            (
                AstronomicalRuntimeInstance::Development,
                test_directory.join("development-daemon.sock"),
            ),
        ],
        stdout: stdout as &mut dyn Write,
        request_timeout: Duration::from_millis(250),
    }
}

#[tokio::test]
async fn should_report_the_channel_binding_and_absent_sockets_as_not_running() {
    let test_directory = fresh_test_directory("absent");
    let mut stdout = Vec::new();
    let mut doctor_dependencies = doctor_dependencies(&test_directory, &mut stdout);

    let doctor_outcome = run_doctor(&mut doctor_dependencies).await;
    doctor_outcome.expect("the doctor report should succeed with both daemons down");

    let rendered_stdout = String::from_utf8_lossy(&stdout);
    // A nonexistent executable path classifies as Development through the
    // instance.rs fallback, so the binding line must say so explicitly.
    assert!(
        rendered_stdout.contains("channel: development (own channel of this binary)"),
        "the report must name the resolved channel: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("stable:") && rendered_stdout.contains("development:"),
        "both channels must get a section: {rendered_stdout:?}"
    );
    assert_eq!(
        rendered_stdout.matches("socket exists: no").count(),
        2,
        "both absent sockets must be reported: {rendered_stdout:?}"
    );
    assert_eq!(
        rendered_stdout
            .matches("status probe: not running — socket file absent")
            .count(),
        2,
        "an absent socket must skip the probe and answer not running: {rendered_stdout:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_a_present_but_dead_socket_through_the_status_probe() {
    let test_directory = fresh_test_directory("dead-socket");
    // A regular file where a unix socket should be: the doctor must report
    // it as existing yet not running, not crash and not call it live.
    std::fs::write(test_directory.join("stable-daemon.sock"), b"not a socket").unwrap();
    let mut stdout = Vec::new();
    let mut doctor_dependencies = doctor_dependencies(&test_directory, &mut stdout);

    let doctor_outcome = run_doctor(&mut doctor_dependencies).await;
    doctor_outcome.expect("the doctor report should succeed against a dead socket file");

    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("status probe: not running — no daemon accepted a connection"),
        "a present but connectionless socket must answer not running: {rendered_stdout:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}
