//! Process and measurement utilities for the disposable-target journeys:
//! kill(2) probes for descendant observation and signal delivery, sccache
//! availability, `du -sk` accounting, and shell exit-status decoding.

#![allow(unsafe_code)]
// The workspace denies unsafe code everywhere else; the kill(2) probes
// below are the sanctioned edges for observing and interrupting journey
// processes from the test process.

use std::path::Path;
use std::process::{Command, Output, Stdio};

pub(crate) const RUNNER_DEADLINE_SECONDS: u64 = 30;
pub(crate) const LIFECYCLE_MARKER_NAME: &str = ".astronomical-disposable-cargo-target";

/// Arms a background kill for a journey process that outlives the runner
/// deadline: a broken runner must fail the test's status assertions instead
/// of hanging the whole test binary (the retired shell contract bounded
/// itself the same way through a `timeout 10` re-exec).
pub(crate) fn arm_termination_watchdog(process_id: u32) {
    let watchdog_process_id = process_id as i32;
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(RUNNER_DEADLINE_SECONDS));
        unsafe {
            libc::kill(watchdog_process_id, libc::SIGKILL);
        }
    });
}

pub(crate) fn process_is_alive(process_id: i32) -> bool {
    unsafe { libc::kill(process_id, 0) == 0 }
}

pub(crate) fn send_terminate(process_id: u32) {
    let termination_result = unsafe { libc::kill(process_id as i32, libc::SIGTERM) };
    assert_eq!(
        termination_result, 0,
        "SIGTERM should reach the journey process {process_id}"
    );
}

pub(crate) fn sccache_is_on_path() -> bool {
    Command::new("sh")
        .arg("-c")
        .arg("command -v sccache")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|availability_status| availability_status.success())
        .unwrap_or(false)
}

pub(crate) fn du_kibibytes(path: &Path) -> u64 {
    let du_output = Command::new("du")
        .arg("-sk")
        .arg(path)
        .output()
        .expect("du should run in the test process");
    assert!(
        du_output.status.success(),
        "du -sk {} failed: {}",
        path.display(),
        String::from_utf8_lossy(&du_output.stderr)
    );
    String::from_utf8_lossy(&du_output.stdout)
        .split_whitespace()
        .next()
        .and_then(|kibibytes_text| kibibytes_text.parse().ok())
        .unwrap_or_else(|| panic!("du -sk {} produced no kibibyte count", path.display()))
}

pub(crate) fn exit_status_code(journey_output: &Output) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    journey_output
        .status
        .code()
        .unwrap_or_else(|| 128 + journey_output.status.signal().unwrap_or(0))
}
