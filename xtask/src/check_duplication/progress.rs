//! Progress and ETA for the file-reading pass, which is the part of a run
//! that grows with the workspace. A line is printed at most every
//! [`PROGRESS_INTERVAL`], so a run that finishes sooner prints none.

use std::time::{Duration, Instant};

pub(crate) const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// The progress line for `completed_files` of `total_files` after `elapsed`.
/// The ETA assumes the remaining files take as long on average as the ones
/// read so far; it is zero before anything is read, when no rate is known.
pub(crate) fn progress_line(
    completed_files: usize,
    total_files: usize,
    elapsed: Duration,
) -> String {
    let elapsed_seconds = elapsed.as_secs_f64();
    let percentage = if total_files == 0 {
        100.0
    } else {
        completed_files as f64 * 100.0 / total_files as f64
    };
    let eta_seconds = if completed_files == 0 {
        0.0
    } else {
        let seconds_per_file = elapsed_seconds / completed_files as f64;
        seconds_per_file * total_files.saturating_sub(completed_files) as f64
    };
    format!(
        "[check-duplication] status=progress step=read-sources files={completed_files}/{total_files} percentage={percentage:.1} elapsed_seconds={elapsed_seconds:.1} ETA_seconds={eta_seconds:.1}"
    )
}

/// Whether enough time has passed since the last progress line to print
/// another.
pub(crate) fn is_progress_report_due(since_last_report: Duration) -> bool {
    since_last_report >= PROGRESS_INTERVAL
}

/// The clock behind the progress lines of one pass.
pub(crate) struct ReadProgress {
    started_at: Instant,
    last_reported_at: Instant,
    total_files: usize,
}

impl ReadProgress {
    pub(crate) fn start(total_files: usize) -> Self {
        let started_at = Instant::now();
        Self {
            started_at,
            last_reported_at: started_at,
            total_files,
        }
    }

    /// Prints a progress line when one is due.
    pub(crate) fn observe(&mut self, completed_files: usize) {
        if !is_progress_report_due(self.last_reported_at.elapsed()) {
            return;
        }
        self.last_reported_at = Instant::now();
        println!(
            "{}",
            progress_line(completed_files, self.total_files, self.started_at.elapsed())
        );
    }
}
