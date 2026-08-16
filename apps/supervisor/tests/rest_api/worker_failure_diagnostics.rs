use std::{path::Path, process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    time::{Instant, sleep, timeout},
};

use super::daemon_process::{terminate_daemon, write_instance_config};

const DAEMON_STARTUP_PREFIX: &str = "astronomicald listening on http://";
const WORKER_STDERR_DIAGNOSTIC: &str = "stderr-probe worker observed visible stderr";

#[tokio::test]
async fn should_persist_the_exact_worker_stderr_when_the_worker_becomes_unavailable() {
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("Cargo should provide the astronomicald executable path");
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-stderr-probe-worker")
            .expect("Cargo should provide the stderr-probe worker fixture path");
    let development_state_directory = tempfile::tempdir().expect("state should be created");
    write_instance_config(development_state_directory.path(), "127.0.0.1:0");
    let synthetic_bundle_executable_directory = development_state_directory.path().join("bin");
    std::fs::create_dir(&synthetic_bundle_executable_directory)
        .expect("the synthetic bundle executable directory should be created");
    let bundled_daemon_path = synthetic_bundle_executable_directory.join("astronomicald");
    let bundled_worker_path =
        synthetic_bundle_executable_directory.join("astronomical-inference-worker");
    std::fs::copy(daemon_executable_path, &bundled_daemon_path)
        .expect("the actual daemon should be copied into the synthetic bundle");
    std::fs::copy(worker_executable_path, bundled_worker_path)
        .expect("the stderr-probe worker should be copied into the synthetic bundle");
    let mut daemon_process = Command::new(bundled_daemon_path)
        .args(["--instance", "development", "--state-directory"])
        .arg(development_state_directory.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("the diagnostic daemon should start");
    let daemon_stdout = daemon_process
        .stdout
        .take()
        .expect("the daemon should expose startup output");
    let startup_line = timeout(
        Duration::from_secs(3),
        BufReader::new(daemon_stdout).lines().next_line(),
    )
    .await
    .expect("the daemon should report startup promptly")
    .expect("daemon startup output should remain readable")
    .expect("daemon startup output should exist");
    assert!(startup_line.starts_with(DAEMON_STARTUP_PREFIX));

    let supervisor_log_text = wait_for_supervisor_log_diagnostic(
        development_state_directory.path(),
        WORKER_STDERR_DIAGNOSTIC,
    )
    .await;

    assert!(supervisor_log_text.contains("worker process exited after closing its event stream"));
    assert!(supervisor_log_text.contains("exit code 0"));
    assert!(supervisor_log_text.contains(WORKER_STDERR_DIAGNOSTIC));
    terminate_daemon(&daemon_process);
    let _daemon_exit_status = daemon_process
        .wait()
        .await
        .expect("the diagnostic daemon should be reaped");
}

async fn wait_for_supervisor_log_diagnostic(
    state_directory: &Path,
    expected_diagnostic: &str,
) -> String {
    let diagnostic_deadline = Instant::now() + Duration::from_secs(3);
    let log_directory = state_directory.join("logs");
    while Instant::now() < diagnostic_deadline {
        let supervisor_log_text = read_supervisor_logs(&log_directory);
        if supervisor_log_text.contains(expected_diagnostic) {
            return supervisor_log_text;
        }
        sleep(Duration::from_millis(25)).await;
    }
    panic!("supervisor log did not retain the expected worker diagnostic");
}

fn read_supervisor_logs(log_directory: &Path) -> String {
    let Ok(log_entries) = std::fs::read_dir(log_directory) else {
        return String::new();
    };
    let mut supervisor_log_text = String::new();
    for log_entry in log_entries.flatten() {
        let file_name = log_entry.file_name();
        if file_name.to_string_lossy().starts_with("supervisor.")
            && let Ok(log_text) = std::fs::read_to_string(log_entry.path())
        {
            supervisor_log_text.push_str(&log_text);
        }
    }
    supervisor_log_text
}
