//! The record of every report that has left this Mac, kept so the person can
//! check exactly what was sent and when. One JSON object per line, appended,
//! so a crash mid-write can damage at most the last line, and a damaged line
//! is skipped on reading instead of hiding the rest.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The log's file name inside the instance's state folder.
pub(crate) const SENT_REPORT_LOG_FILE_NAME: &str = "sent-problem-reports.jsonl";

/// One sent report as the log keeps it: when it left, and the exact lines the
/// person read before it was sent.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct SentProblemReport {
    pub(crate) sent_at_unix_seconds: u64,
    pub(crate) readable_lines: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct SentReportLog {
    log_file_path: PathBuf,
}

impl SentReportLog {
    pub(crate) fn in_state_directory(state_directory: &Path) -> Self {
        Self {
            log_file_path: state_directory.join(SENT_REPORT_LOG_FILE_NAME),
        }
    }

    pub(crate) fn record(&self, sent_report: &SentProblemReport) -> std::io::Result<()> {
        if let Some(log_directory) = self.log_file_path.parent() {
            fs::create_dir_all(log_directory)?;
        }
        let mut log_line = serde_json::to_string(sent_report).map_err(std::io::Error::other)?;
        log_line.push('\n');
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file_path)?
            .write_all(log_line.as_bytes())
    }

    /// Every sent report, oldest first. A missing log means nothing was sent.
    pub(crate) fn sent_reports(&self) -> Vec<SentProblemReport> {
        let Ok(log_text) = fs::read_to_string(&self.log_file_path) else {
            return Vec::new();
        };
        let mut sent_reports = Vec::new();
        for log_line in log_text.lines() {
            if let Ok(sent_report) = serde_json::from_str::<SentProblemReport>(log_line) {
                sent_reports.push(sent_report);
            }
        }
        sent_reports
    }
}

#[cfg(test)]
#[path = "sent_report_log_tests.rs"]
mod sent_report_log_tests;
