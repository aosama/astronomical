use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use astronomical_daemon::AstronomicalInstanceLock;
use astronomical_daemon::{StartupFailureStage, read_last_startup_failure};

/// Spawns the real `astronomicald` against a throwaway state directory and
/// returns the child plus the port parsed from its "listening on" line.
/// The `--state-directory` override gives every spawn an ephemeral port, so
/// these tests never touch the operator's live stable or development ports.
struct SpawnedDaemon {
    child_process: Child,
    bound_port: u16,
}

impl Drop for SpawnedDaemon {
    fn drop(&mut self) {
        let _ = self.child_process.kill();
        let _ = self.child_process.wait();
    }
}

fn spawn_real_daemon(state_directory: &Path) -> SpawnedDaemon {
    let daemon_executable = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("the astronomicald binary should be addressable from this test");
    let mut child_process = Command::new(daemon_executable)
        .arg("--instance")
        .arg("development")
        .arg("--state-directory")
        .arg(state_directory)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the real daemon binary should spawn");

    let child_stdout = child_process.stdout.take().expect("piped stdout");
    let bound_port = read_bound_port(child_stdout);
    SpawnedDaemon {
        child_process,
        bound_port,
    }
}

fn read_bound_port(mut child_stdout: ChildStdout) -> u16 {
    let mut reader = BufReader::new(&mut child_stdout);
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        let mut output_line = String::new();
        if reader.read_line(&mut output_line).unwrap_or(0) == 0 {
            panic!("daemon exited before announcing its listener");
        }
        if let Some(port_text) =
            output_line.strip_prefix("astronomicald listening on http://127.0.0.1:")
        {
            let bound_port: u16 = port_text
                .trim()
                .parse()
                .expect("the daemon listener port should parse");
            // Reattach stdout so the child's pipe is not dropped early.
            drop(reader);
            return bound_port;
        }
    }
    panic!("daemon never announced its listener within 20 seconds");
}

fn http_request(bound_port: u16, method: &str, path: &str) -> Option<u16> {
    let mut connection = TcpStream::connect(("127.0.0.1", bound_port)).ok()?;
    connection
        .set_read_timeout(Some(Duration::from_secs(3)))
        .ok()?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
    );
    connection.write_all(request.as_bytes()).ok()?;
    let mut response_bytes = Vec::new();
    connection.read_to_end(&mut response_bytes).ok()?;
    let response_head = String::from_utf8_lossy(&response_bytes);
    response_head
        .split_whitespace()
        .nth(1)
        .and_then(|status_text| status_text.parse::<u16>().ok())
}

fn wait_for_daemon_health(bound_port: u16) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if http_request(bound_port, "GET", "/health") == Some(200) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("daemon never became healthy within 20 seconds");
}

fn shutdown_daemon(bound_port: u16) {
    let _ = http_request(bound_port, "POST", "/v1/control/shutdown");
}

fn wait_for_child_exit(child_process: &mut Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(_exit_status) = child_process.try_wait().expect("child polling") {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("daemon did not exit within the timeout");
}

#[test]
fn should_record_a_startup_failure_when_another_daemon_holds_the_instance_lock() {
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    let mut first_daemon = spawn_real_daemon(state_directory.path());
    wait_for_daemon_health(first_daemon.bound_port);

    // The losing spawn must not wait for a listener line: it is expected to
    // die before ever binding, which is exactly the historically silent path.
    let daemon_executable = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("the astronomicald binary should be addressable from this test");
    let second_daemon = Command::new(daemon_executable)
        .arg("--instance")
        .arg("development")
        .arg("--state-directory")
        .arg(state_directory.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the second daemon spawn should succeed");
    let second_daemon_output = second_daemon.wait_with_output().expect("loser exit");
    assert!(
        !second_daemon_output.status.success(),
        "the lock-losing daemon must exit with a failure code"
    );
    let loser_stderr = String::from_utf8_lossy(&second_daemon_output.stderr);
    assert!(
        loser_stderr.contains("already running"),
        "the loser must still explain itself on stderr, got: {loser_stderr}"
    );

    let failure_record = read_last_startup_failure(state_directory.path())
        .expect("the lock conflict must leave a machine-readable record");
    assert_eq!(failure_record.stage, StartupFailureStage::InstanceLock);
    assert!(failure_record.message.contains("already running"));
    let failure_context = failure_record.context.expect("holder context");
    assert!(
        failure_context.contains(&format!("pid={}", first_daemon.child_process.id())),
        "the record must name the lock holder, got: {failure_context}"
    );
    let failure_log = std::fs::read_to_string(
        state_directory
            .path()
            .join("logs")
            .join("startup-failures.log"),
    )
    .expect("the human-readable failure log");
    assert!(failure_log.contains("[instance-lock]"));

    shutdown_daemon(first_daemon.bound_port);
    wait_for_child_exit(&mut first_daemon.child_process, Duration::from_secs(20));
}

#[test]
fn should_release_the_instance_lock_promptly_after_health_stops_responding() {
    // Regression guard for the restart race: the menu respawns the daemon the
    // moment /health stops answering, so the old process must not keep the
    // instance flock long after its health endpoint is gone. Idle teardown
    // measured ~6 ms on development hardware; the budget allows two seconds
    // so slow CI machines stay green while real regressions still fail.
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    let mut daemon = spawn_real_daemon(state_directory.path());
    wait_for_daemon_health(daemon.bound_port);

    shutdown_daemon(daemon.bound_port);
    let health_lost_at = Instant::now();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if http_request(daemon.bound_port, "GET", "/health") != Some(200) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let lock_path = state_directory.path().join("instance.lock");
    while Instant::now() - health_lost_at < Duration::from_secs(2) {
        if AstronomicalInstanceLock::acquire(&lock_path).is_ok() {
            wait_for_child_exit(&mut daemon.child_process, Duration::from_secs(20));
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the instance flock stayed held more than 2 seconds after /health stopped responding");
}
