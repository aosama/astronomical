//! The progress lines both decision verbs print while they work: one line
//! at the start, then at most one per interval, then one at the end, each
//! with the share done and the time left.

use std::io::Write;
use std::time::{Duration, Instant};

use astronomical_decision_training::calibrated_model_folder::FolderCopyProgress;
use astronomical_decision_training::progress_line::progress_line;

pub(super) const BYTES_PER_MEGABYTE: u64 = 1024 * 1024;

/// Prints a progress line at most once per interval, and always at the end.
pub(super) struct TimedProgress {
    stage_name: &'static str,
    unit_name: &'static str,
    started_at: Instant,
    last_report_at: Option<Instant>,
    report_interval: Duration,
}

impl TimedProgress {
    pub(super) fn new(
        stage_name: &'static str,
        unit_name: &'static str,
        report_interval: Duration,
    ) -> Self {
        Self {
            stage_name,
            unit_name,
            started_at: Instant::now(),
            last_report_at: None,
            report_interval,
        }
    }

    pub(super) fn report(
        &mut self,
        stderr: &mut dyn Write,
        completed_count: u64,
        total_count: u64,
    ) {
        let is_finished = completed_count >= total_count;
        let is_due = match self.last_report_at {
            Some(last_report_at) => last_report_at.elapsed() >= self.report_interval,
            None => true,
        };
        if !is_finished && !is_due {
            return;
        }
        self.last_report_at = Some(Instant::now());
        let line = progress_line(
            self.stage_name,
            completed_count,
            total_count,
            self.unit_name,
            self.started_at.elapsed(),
        );
        let _ = writeln!(stderr, "{line}");
    }
}

/// The copy-and-check pass reports bytes; people read megabytes.
pub(super) struct CopyProgressLine<'w> {
    pub(super) stderr: &'w mut dyn Write,
    pub(super) timed_progress: TimedProgress,
}

impl FolderCopyProgress for CopyProgressLine<'_> {
    fn report(&mut self, processed_bytes: u64, total_bytes: u64) {
        let processed_megabytes = processed_bytes / BYTES_PER_MEGABYTE;
        let total_megabytes = total_bytes.div_ceil(BYTES_PER_MEGABYTE);
        let completed_megabytes = if processed_bytes >= total_bytes {
            total_megabytes
        } else {
            processed_megabytes
        };
        self.timed_progress
            .report(self.stderr, completed_megabytes, total_megabytes);
    }
}

#[cfg(test)]
#[path = "progress_reporting_tests.rs"]
mod tests;
