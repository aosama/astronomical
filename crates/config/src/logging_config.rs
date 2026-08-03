use std::path::{Path, PathBuf};

use serde::Deserialize;

// tracing-appender otherwise retains as many as 128,000 heap-allocated lines.
// A small lossy queue keeps diagnostics off the inference path without letting
// a slow or full disk turn logging into a material memory consumer.
const LOG_BUFFERED_LINE_LIMIT: usize = 1_024;

/// Supported diagnostic verbosity, ordered from least to most verbose.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    #[default]
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    /// Returns the tracing filter directive for this level.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

/// Resolved bounded file-logging settings shared by both processes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoggingConfig {
    buffered_line_limit: usize,
    directory: PathBuf,
    level: LogLevel,
    retained_files: usize,
}

impl LoggingConfig {
    /// Creates one resolved logging policy with a fixed memory-safe queue.
    #[must_use]
    pub const fn new(directory: PathBuf, level: LogLevel, retained_files: usize) -> Self {
        Self {
            buffered_line_limit: LOG_BUFFERED_LINE_LIMIT,
            directory,
            level,
            retained_files,
        }
    }

    /// Returns the fixed in-memory queue capacity for non-blocking log lines.
    #[must_use]
    pub const fn buffered_line_limit(&self) -> usize {
        self.buffered_line_limit
    }

    /// Returns the absolute directory that owns rolling log files.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the configured diagnostic verbosity.
    #[must_use]
    pub const fn level(&self) -> LogLevel {
        self.level
    }

    /// Returns the positive number of hourly files retained per process.
    #[must_use]
    pub const fn retained_files(&self) -> usize {
        self.retained_files
    }
}
