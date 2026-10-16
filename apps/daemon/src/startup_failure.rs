//! Startup-failure persistence for the pre-logging phase of `astronomicald`.
//!
//! The daemon acquires its instance lock, loads configuration, and only then
//! initializes tracing. Every failure before that point historically reached
//! the operator only through `eprintln!`, and the menu app spawns the daemon
//! with stderr discarded — so an early exit was completely invisible. This
//! module gives those failures two durable homes inside the instance state
//! directory: a machine-readable `last-startup-failure.json` the menu can read
//! and render, and an append-only `logs/startup-failures.log` for humans.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// File the menu app reads to explain why the previous daemon start died.
pub const LAST_STARTUP_FAILURE_FILE_NAME: &str = "last-startup-failure.json";

/// Where the daemon writes its own instance-pid note while holding the lock.
pub const INSTANCE_LOCK_FILE_NAME: &str = "instance.lock";

#[derive(Debug, Serialize, Deserialize)]
pub struct StartupFailureRecord {
    /// Unix epoch milliseconds of the failed start attempt. The workspace
    /// deliberately has no calendar dependency; the menu renders this to a
    /// local date string.
    pub timestamp_unix_millis: u64,
    /// Which early startup stage failed, e.g. `instance-lock`.
    pub stage: StartupFailureStage,
    /// Human-readable top-level error message.
    pub message: String,
    /// Extra operational context, e.g. the pid note of whichever daemon held
    /// the instance lock during a lock conflict.
    pub context: Option<String>,
    /// Error chain, outermost error first, sources following. This is the
    /// fullest failure story Rust can produce for `Result`-based startup
    /// paths; real backtraces only exist for panics.
    pub cause_chain: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StartupFailureStage {
    InstancePaths,
    InstanceLock,
    Configuration,
    LoggingInitialization,
}

impl fmt::Display for StartupFailureStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stage_label = match self {
            StartupFailureStage::InstancePaths => "instance-paths",
            StartupFailureStage::InstanceLock => "instance-lock",
            StartupFailureStage::Configuration => "configuration",
            StartupFailureStage::LoggingInitialization => "logging-initialization",
        };
        formatter.write_str(stage_label)
    }
}

/// Records a pre-logging startup failure durably.
///
/// Best-effort by design: a failure to record a failure must never mask or
/// change the original startup outcome, so every I/O error here is swallowed
/// after being reported to stderr.
pub fn record_startup_failure(
    state_directory: &Path,
    stage: StartupFailureStage,
    startup_error: &(dyn Error + 'static),
    context: Option<String>,
) {
    let mut cause_chain = vec![startup_error.to_string()];
    let mut current_source = startup_error.source();
    while let Some(source_error) = current_source {
        cause_chain.push(source_error.to_string());
        current_source = source_error.source();
    }
    let record = StartupFailureRecord {
        timestamp_unix_millis: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since_epoch| since_epoch.as_millis() as u64)
            .unwrap_or(0),
        stage,
        message: startup_error.to_string(),
        context,
        cause_chain,
    };
    if let Err(persist_error) = persist_last_startup_failure(state_directory, &record) {
        eprintln!("astronomicald could not persist the startup-failure record: {persist_error}");
    }
    if let Err(append_error) = append_startup_failure_log_line(state_directory, &record) {
        eprintln!("astronomicald could not append the startup-failure log: {append_error}");
    }
}

/// Reads the persisted record of the most recent startup failure, if any.
pub fn read_last_startup_failure(state_directory: &Path) -> Option<StartupFailureRecord> {
    let record_path = last_startup_failure_path(state_directory);
    let record_bytes = fs::read(record_path).ok()?;
    serde_json::from_slice(&record_bytes).ok()
}

fn persist_last_startup_failure(
    state_directory: &Path,
    record: &StartupFailureRecord,
) -> std::io::Result<()> {
    let record_path = last_startup_failure_path(state_directory);
    let temporary_path = record_path.with_extension("json.tmp");
    let record_bytes = serde_json::to_vec_pretty(record).map_err(std::io::Error::other)?;
    fs::write(&temporary_path, &record_bytes)?;
    fs::rename(&temporary_path, &record_path)
}

fn append_startup_failure_log_line(
    state_directory: &Path,
    record: &StartupFailureRecord,
) -> std::io::Result<()> {
    let logs_directory = state_directory.join("logs");
    fs::create_dir_all(&logs_directory)?;
    let mut log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs_directory.join("startup-failures.log"))?;
    let cause_story = if record.cause_chain.len() > 1 {
        format!(
            " (caused by {})",
            record.cause_chain[1..].join("; caused by ")
        )
    } else {
        String::new()
    };
    writeln!(
        log_file,
        "unix_ms={} [{}] {}{}",
        record.timestamp_unix_millis, record.stage, record.message, cause_story
    )
}

/// Reads the instance note written by whichever daemon currently holds the
/// flock, so a lock conflict can name its holder instead of guessing.
pub fn read_instance_lock_holder(state_directory: &Path) -> Option<String> {
    let lock_contents = fs::read_to_string(state_directory.join(INSTANCE_LOCK_FILE_NAME)).ok()?;
    let trimmed_contents = lock_contents.trim();
    if trimmed_contents.is_empty() {
        None
    } else {
        Some(trimmed_contents.to_owned())
    }
}

fn last_startup_failure_path(state_directory: &Path) -> PathBuf {
    state_directory.join(LAST_STARTUP_FAILURE_FILE_NAME)
}

#[cfg(test)]
mod startup_failure_tests;
