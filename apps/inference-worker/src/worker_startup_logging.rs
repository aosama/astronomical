use std::error::Error;

use astronomical_ipc_protocol::{WorkerLogLevel, WorkerStartupConfiguration};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt};

/// Returns the rolling policy shared by the worker process log.
#[must_use]
pub fn astronomical_log_rotation() -> tracing_appender::rolling::Rotation {
    tracing_appender::rolling::Rotation::HOURLY
}

/// Initializes bounded worker diagnostics in the configured rolling log file.
pub fn initialize_tracing(
    worker_startup_configuration: &WorkerStartupConfiguration,
) -> Result<WorkerGuard, Box<dyn Error + Send + Sync>> {
    std::fs::create_dir_all(&worker_startup_configuration.logging_directory)?;
    let file_appender = tracing_appender::rolling::Builder::new()
        .rotation(astronomical_log_rotation())
        .filename_prefix("worker")
        .filename_suffix("log")
        .max_log_files(worker_startup_configuration.retained_log_file_count)
        .build(&worker_startup_configuration.logging_directory)?;
    // Lossy delivery is intentional: inference must never block or accumulate
    // an enormous queue merely because diagnostic storage is slow.
    let (file_writer, guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
        .buffered_lines_limit(1_024)
        .lossy(true)
        .thread_name("astronomical-worker-log-writer")
        .finish(file_appender);
    fmt()
        .with_env_filter(EnvFilter::new(worker_log_level_name(
            worker_startup_configuration.logging_level,
        )))
        .with_ansi(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_writer(file_writer)
        .try_init()?;
    Ok(guard)
}

fn worker_log_level_name(worker_log_level: WorkerLogLevel) -> &'static str {
    match worker_log_level {
        WorkerLogLevel::Error => "error",
        WorkerLogLevel::Warn => "warn",
        WorkerLogLevel::Info => "info",
        WorkerLogLevel::Debug => "debug",
        WorkerLogLevel::Trace => "trace",
    }
}
