use std::{path::Path, process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    time::{Instant, sleep, timeout},
};

use crate::common::daemon;

const DAEMON_STARTUP_PREFIX: &str = "astronomicald listening on http://";
const WORKER_STDERR_DIAGNOSTIC: &str = "stderr-probe worker observed visible stderr";

#[tokio::test]
async fn should_persist_the_exact_worker_stderr_when_the_worker_becomes_unavailable() {
    let daemon_executable_path = astronomical_test_harness::workspace_binary("astronomicald")
        .to_string_lossy()
        .into_owned();
    let worker_executable_path =
        astronomical_test_harness::workspace_binary("astronomical-daemon-stderr-probe-worker")
            .to_string_lossy()
            .into_owned();
    let development_state_directory = tempfile::tempdir().expect("state should be created");
    daemon::write_instance_config(development_state_directory.path());
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
    let mut daemon_command = Command::new(bundled_daemon_path);
    let mut daemon_process = daemon_command
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

    let daemon_log_text = wait_for_daemon_log_diagnostic(
        development_state_directory.path(),
        WORKER_STDERR_DIAGNOSTIC,
    )
    .await;

    assert!(daemon_log_text.contains("worker process exited after closing its event stream"));
    assert!(daemon_log_text.contains("exit code 0"));
    assert!(daemon_log_text.contains(WORKER_STDERR_DIAGNOSTIC));
    daemon::terminate_daemon(&daemon_process);
    let _daemon_exit_status = daemon_process
        .wait()
        .await
        .expect("the diagnostic daemon should be reaped");
}

/// How long the daemon may take to launch the probe worker, see it exit and
/// log its stderr. This journey proves the diagnostic is retained, not how
/// fast: the worker is a freshly copied, unsigned binary, and macOS holds the
/// first launch of such a binary for a security scan whose latency on a
/// contended hosted runner reached past a 3-second deadline (one failure in
/// 60 CI runs, at 3.14 s). The wait returns as soon as the line appears, so a
/// wide deadline costs nothing when the journey passes.
const WORKER_DIAGNOSTIC_DEADLINE: Duration = Duration::from_secs(20);

async fn wait_for_daemon_log_diagnostic(
    state_directory: &Path,
    expected_diagnostic: &str,
) -> String {
    let diagnostic_deadline = Instant::now() + WORKER_DIAGNOSTIC_DEADLINE;
    let log_directory = state_directory.join("logs");
    let mut daemon_log_text = String::new();
    while Instant::now() < diagnostic_deadline {
        daemon_log_text = read_daemon_logs(&log_directory);
        if daemon_log_text.contains(expected_diagnostic) {
            return daemon_log_text;
        }
        sleep(Duration::from_millis(25)).await;
    }
    // The log read so far tells a slow worker launch apart from a lost
    // diagnostic, which the bare message could not.
    panic!(
        "daemon log did not retain the expected worker diagnostic within {WORKER_DIAGNOSTIC_DEADLINE:?}; daemon log so far:\n{daemon_log_text}"
    );
}

fn read_daemon_logs(log_directory: &Path) -> String {
    let Ok(log_entries) = std::fs::read_dir(log_directory) else {
        return String::new();
    };
    let mut daemon_log_text = String::new();
    for log_entry in log_entries.flatten() {
        let file_name = log_entry.file_name();
        if file_name.to_string_lossy().starts_with("daemon.")
            && let Ok(log_text) = std::fs::read_to_string(log_entry.path())
        {
            daemon_log_text.push_str(&log_text);
        }
    }
    daemon_log_text
}
