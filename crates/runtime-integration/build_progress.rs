//! Streams native build lifecycle events to a progress file that CI tails into
//! the live job log, so a multi-minute silent CMake build is observable while
//! it runs instead of only after it finishes.

use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread::JoinHandle,
    time::{Duration, Instant},
};

pub(crate) const NATIVE_BUILD_PROGRESS_FILE_VARIABLE: &str =
    "ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE";
const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const EVENT_PREFIX: &str = "[native-build]";

pub(crate) struct NativeBuildProgress {
    progress_file_path: Option<PathBuf>,
    heartbeat_interval: Duration,
}

impl NativeBuildProgress {
    pub(crate) fn from_environment() -> Self {
        let progress_file_path = env::var_os(NATIVE_BUILD_PROGRESS_FILE_VARIABLE)
            .filter(|progress_file_text| !progress_file_text.is_empty())
            .map(PathBuf::from);
        Self::new(progress_file_path, DEFAULT_HEARTBEAT_INTERVAL)
    }

    pub(crate) fn new(progress_file_path: Option<PathBuf>, heartbeat_interval: Duration) -> Self {
        if let Some(parent_directory) = progress_file_path
            .as_ref()
            .and_then(|progress_file_path| progress_file_path.parent())
        {
            let _ = fs::create_dir_all(parent_directory);
        }
        Self {
            progress_file_path,
            heartbeat_interval,
        }
    }

    pub(crate) fn record_event(&self, event_text: &str) {
        eprintln!("{EVENT_PREFIX} {event_text}");
        self.append_to_progress_file(event_text);
    }

    /// Runs one native build operation while streaming its lifecycle: a start
    /// line, periodic heartbeat lines while the body runs, then exactly one
    /// success or failed line. Heartbeats stop before the outcome line is
    /// written, so the file never shows a heartbeat after the operation ended.
    pub(crate) fn run_operation<T, E, F>(
        &self,
        operation: &str,
        native_operation: F,
    ) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        let operation_started_at = Instant::now();
        self.record_event(&format!("operation={operation} status=start"));
        let (heartbeat_stop_sender, heartbeat_stop_receiver) = mpsc::channel::<()>();
        let heartbeat_thread =
            self.spawn_heartbeat_thread(operation, operation_started_at, heartbeat_stop_receiver);
        let operation_result = native_operation();
        // Dropping the last sender disconnects the channel, which wakes a
        // blocked heartbeat wait immediately so the join below is prompt.
        drop(heartbeat_stop_sender);
        let _ = heartbeat_thread.join();
        let elapsed_seconds = operation_started_at.elapsed().as_secs_f64();
        match &operation_result {
            Ok(_) => self.record_event(&format!(
                "operation={operation} status=success elapsed_seconds={elapsed_seconds:.3}"
            )),
            Err(operation_error) => self.record_event(&format!(
                "operation={operation} status=failed elapsed_seconds={elapsed_seconds:.3} error={operation_error}"
            )),
        }
        operation_result
    }

    fn spawn_heartbeat_thread(
        &self,
        operation: &str,
        operation_started_at: Instant,
        heartbeat_stop_receiver: Receiver<()>,
    ) -> JoinHandle<()> {
        let heartbeat_interval = self.heartbeat_interval;
        let progress_file_path = self.progress_file_path.clone();
        let operation_text = operation.to_owned();
        std::thread::spawn(move || {
            let mut heartbeat_count: u32 = 0;
            loop {
                match heartbeat_stop_receiver.recv_timeout(heartbeat_interval) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                    Err(RecvTimeoutError::Timeout) => {
                        heartbeat_count += 1;
                        append_line_to_file(
                            progress_file_path.as_deref(),
                            &format!(
                                "operation={operation_text} status=heartbeat \
                                 count={heartbeat_count} elapsed_seconds={:.3}",
                                operation_started_at.elapsed().as_secs_f64()
                            ),
                        );
                    }
                }
            }
        })
    }

    fn append_to_progress_file(&self, event_text: &str) {
        append_line_to_file(self.progress_file_path.as_deref(), event_text);
    }
}

fn append_line_to_file(progress_file_path: Option<&Path>, event_text: &str) {
    let Some(progress_file_path) = progress_file_path else {
        return;
    };
    // Progress streaming is best-effort: a missing or unwritable progress file
    // must never fail the native build it observes. The whole line is written
    // with one append so concurrent lifecycle and heartbeat writes cannot
    // interleave mid-line.
    let progress_line = format!("{EVENT_PREFIX} {event_text}\n");
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open(progress_file_path)
        .and_then(|mut progress_file| progress_file.write_all(progress_line.as_bytes()));
}
