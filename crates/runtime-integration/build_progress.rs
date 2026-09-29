//! Live progress reporting for the pinned native runtime build.
//!
//! Cargo captures build-script stderr, so the multi-minute CMake phases are
//! invisible in hosted CI logs until they finish. When
//! `ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE` names a file, this module appends
//! one progress line per lifecycle event there, letting a `tail -f` in the CI
//! step stream the silent stretch into the job log as it happens.
//!
//! Progress reporting is best-effort telemetry: write failures are ignored so
//! reporting can never fail a build, while a misconfigured relative path is
//! rejected eagerly because a silently disabled stream would hide exactly the
//! slow phases this module exists to expose.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const NATIVE_BUILD_PROGRESS_FILE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE";

// Long enough that a healthy CMake phase is not flooded, short enough that a
// silent stretch never exceeds one heartbeat gap in a hosted log.
const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

const PROGRESS_LINE_PREFIX: &str = "[native-build-progress]";

pub struct NativeBuildProgress {
    progress_file_path: Option<PathBuf>,
    heartbeat_interval: Duration,
}

impl NativeBuildProgress {
    // Called by build.rs, which is not part of the hermetic test target that
    // includes this module through #[path]; the warning would otherwise fire there.
    #[allow(dead_code)]
    pub fn from_environment() -> Result<Self, String> {
        match std::env::var_os(NATIVE_BUILD_PROGRESS_FILE_VARIABLE) {
            None => Ok(Self::disabled()),
            Some(raw_progress_file_path) => {
                Self::from_file_path(PathBuf::from(raw_progress_file_path))
            }
        }
    }

    pub fn from_file_path(progress_file_path: PathBuf) -> Result<Self, String> {
        if !progress_file_path.is_absolute() {
            return Err(format!(
                "{NATIVE_BUILD_PROGRESS_FILE_VARIABLE} must be an absolute path: {}",
                progress_file_path.display()
            ));
        }
        Ok(Self {
            progress_file_path: Some(progress_file_path),
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        })
    }

    pub fn disabled() -> Self {
        Self {
            progress_file_path: None,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        }
    }

    // Test seam: hermetic tests shrink the heartbeat interval so heartbeat
    // contracts run in milliseconds instead of waiting thirty seconds.
    #[allow(dead_code)]
    pub fn with_heartbeat_interval(mut self, heartbeat_interval: Duration) -> Self {
        self.heartbeat_interval = heartbeat_interval;
        self
    }

    pub fn record_build_start(&self, parallel_job_count: &str) {
        self.append_line(&format!("status=start parallel_jobs={parallel_job_count}"));
    }

    pub fn begin_operation(&self, operation_name: &str) -> NativeBuildOperationProgress {
        let Some(progress_file_path) = &self.progress_file_path else {
            return NativeBuildOperationProgress::inactive();
        };
        self.append_line(&format!("operation={operation_name} status=start"));
        NativeBuildOperationProgress::active(
            progress_file_path.clone(),
            operation_name.to_owned(),
            self.heartbeat_interval,
        )
    }

    pub fn record_build_completion(&self, was_built: bool, elapsed: Duration) {
        let outcome = if was_built { "built" } else { "reused" };
        self.append_line(&format!(
            "status=complete outcome={outcome} elapsed_seconds={:.3}",
            elapsed.as_secs_f64()
        ));
    }

    fn append_line(&self, line_body: &str) {
        if let Some(progress_file_path) = &self.progress_file_path {
            append_progress_line(progress_file_path, line_body);
        }
    }
}

pub struct NativeBuildOperationProgress {
    progress_file_path: Option<PathBuf>,
    operation_name: String,
    started_at: Instant,
    heartbeat: Option<HeartbeatThread>,
}

impl NativeBuildOperationProgress {
    fn inactive() -> Self {
        Self {
            progress_file_path: None,
            operation_name: String::new(),
            started_at: Instant::now(),
            heartbeat: None,
        }
    }

    fn active(
        progress_file_path: PathBuf,
        operation_name: String,
        heartbeat_interval: Duration,
    ) -> Self {
        let heartbeat = Some(HeartbeatThread::spawn(
            progress_file_path.clone(),
            operation_name.clone(),
            heartbeat_interval,
        ));
        Self {
            progress_file_path: Some(progress_file_path),
            operation_name,
            started_at: Instant::now(),
            heartbeat,
        }
    }

    // Returns the elapsed duration so callers can keep it in failure
    // diagnostics without timing the operation a second time.
    pub fn complete(mut self, outcome: &str) -> Duration {
        let elapsed = self.started_at.elapsed();
        // Stop the heartbeat before writing the final line so no late
        // "running" heartbeat can land after the operation's outcome.
        self.stop_heartbeat();
        if let Some(progress_file_path) = &self.progress_file_path {
            append_progress_line(
                progress_file_path,
                &format!(
                    "operation={} status={outcome} elapsed_seconds={:.3}",
                    self.operation_name,
                    elapsed.as_secs_f64()
                ),
            );
        }
        elapsed
    }

    fn stop_heartbeat(&mut self) {
        if let Some(heartbeat) = self.heartbeat.take() {
            heartbeat.stop();
        }
    }
}

impl Drop for NativeBuildOperationProgress {
    fn drop(&mut self) {
        self.stop_heartbeat();
    }
}

struct HeartbeatThread {
    stop_sender: Option<mpsc::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl HeartbeatThread {
    fn spawn(
        progress_file_path: PathBuf,
        operation_name: String,
        heartbeat_interval: Duration,
    ) -> Self {
        let (stop_sender, stop_receiver) = mpsc::channel::<()>();
        let heartbeat_started_at = Instant::now();
        let spawn_result = std::thread::Builder::new()
            .name("native-build-heartbeat".to_owned())
            .spawn(move || {
                loop {
                    match stop_receiver.recv_timeout(heartbeat_interval) {
                        Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                        Err(RecvTimeoutError::Timeout) => append_progress_line(
                            &progress_file_path,
                            &format!(
                                "operation={operation_name} status=running elapsed_seconds={:.3}",
                                heartbeat_started_at.elapsed().as_secs_f64()
                            ),
                        ),
                    }
                }
            });
        match spawn_result {
            Ok(join_handle) => Self {
                stop_sender: Some(stop_sender),
                join_handle: Some(join_handle),
            },
            // A failed heartbeat spawn must not fail the build; the start and
            // completion lines still land on the main build thread.
            Err(_) => Self {
                stop_sender: None,
                join_handle: None,
            },
        }
    }

    fn stop(mut self) {
        // Dropping the sender wakes recv_timeout immediately through the
        // disconnected-channel error, so the join below never waits out the
        // full heartbeat interval.
        self.stop_sender.take();
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

fn append_progress_line(progress_file_path: &Path, line_body: &str) {
    // A per-write append-mode open keeps the heartbeat thread and the build
    // thread lock-free, and ignoring write errors keeps telemetry best-effort.
    if let Ok(mut progress_file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(progress_file_path)
    {
        let _ = writeln!(progress_file, "{PROGRESS_LINE_PREFIX} {line_body}");
    }
}
