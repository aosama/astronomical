//! Percentage and estimated-time-to-completion reporting for every batch stage.
//!
//! The repository rule for data-processing jobs is that progress and an ETA
//! are reported on a fixed interval (five seconds by default). The sweep and
//! the requantizer both run for hours on large models, often with output
//! redirected to a file, and OptiQ's own history shows what happens without
//! it: an 11-hour sweep that printed nothing until it ended with nothing to
//! allocate. The reporter is injected so tests can observe it and the CLI can
//! route it to standard error.

use std::time::{Duration, Instant};

#[cfg(test)]
#[path = "progress_tests.rs"]
mod progress_tests;

/// One snapshot of a stage's progress.
#[derive(Clone, Debug, PartialEq)]
pub struct StageProgress {
    pub stage_name: &'static str,
    pub completed_work_units: u64,
    pub total_work_units: u64,
    pub elapsed: Duration,
    pub estimate: EtaEstimate,
    /// A short stage-specific detail, for example the module being probed.
    pub detail: String,
}

impl StageProgress {
    #[must_use]
    pub fn percent_complete(&self) -> f64 {
        if self.total_work_units == 0 {
            return 100.0;
        }
        100.0 * self.completed_work_units as f64 / self.total_work_units as f64
    }
}

/// The projected remaining duration, or why none can be given yet.
#[derive(Clone, Debug, PartialEq)]
pub enum EtaEstimate {
    /// Not enough completed work units to extrapolate.
    NotYetAvailable,
    Remaining(Duration),
}

/// Where progress lines go. Implemented by a standard-error writer in the CLI
/// and by a recording fake in tests.
pub trait ProgressReporter {
    fn report(&mut self, progress: &StageProgress);
}

/// A reporter that discards everything, for callers that already report at a
/// higher level.
#[derive(Debug, Default)]
pub struct SilentProgressReporter;

impl ProgressReporter for SilentProgressReporter {
    fn report(&mut self, progress: &StageProgress) {
        let _ = progress;
    }
}

/// Tracks one stage and forwards a snapshot to the reporter at most once per
/// interval, plus once at completion.
///
/// Work to port: the ETA is `(elapsed / completed) * remaining`, the same
/// per-unit extrapolation OptiQ prints as `(Ns/layer, ~Hh left)`. A resumed
/// sweep must extrapolate from the units completed in this run only, not from
/// checkpointed ones, or the first report after a resume claims the run is
/// nearly done. Time comes from an injected clock so tests are deterministic.
pub struct StageProgressTracker<'reporter> {
    stage_name: &'static str,
    total_work_units: u64,
    completed_work_units: u64,
    /// Units already complete when the tracker started (a resumed checkpoint);
    /// excluded from the rate estimate.
    units_completed_before_start: u64,
    started_at: Instant,
    last_report_at: Option<Instant>,
    report_interval: Duration,
    reporter: &'reporter mut dyn ProgressReporter,
}

impl<'reporter> StageProgressTracker<'reporter> {
    /// The repository's default reporting cadence.
    pub const DEFAULT_REPORT_INTERVAL: Duration = Duration::from_secs(5);

    pub fn new(
        stage_name: &'static str,
        total_work_units: u64,
        units_completed_before_start: u64,
        report_interval: Duration,
        reporter: &'reporter mut dyn ProgressReporter,
    ) -> Self {
        Self {
            stage_name,
            total_work_units,
            completed_work_units: units_completed_before_start,
            units_completed_before_start,
            started_at: Instant::now(),
            last_report_at: None,
            report_interval,
            reporter,
        }
    }

    /// Records one more completed unit and reports if the interval elapsed.
    pub fn advance(&mut self, detail: &str) {
        let _ = detail;
        todo!(
            "increment completed units; call report_now when last_report_at is None or the interval elapsed"
        )
    }

    /// Reports unconditionally; called at stage end so the last line always
    /// shows 100 percent.
    pub fn report_now(&mut self, detail: &str) {
        let _ = (detail, &mut self.reporter);
        todo!(
            "build StageProgress with the ETA from the units completed since start and forward it to the reporter"
        )
    }

    /// Computes the ETA from the elapsed time and the units completed since
    /// the tracker started.
    #[must_use]
    pub fn estimate(&self, now: Instant) -> EtaEstimate {
        let _ = now;
        todo!(
            "NotYetAvailable until at least one unit completed in this run; otherwise elapsed / completed_since_start * remaining"
        )
    }

    #[must_use]
    pub const fn total_work_units(&self) -> u64 {
        self.total_work_units
    }

    #[must_use]
    pub const fn completed_work_units(&self) -> u64 {
        self.completed_work_units
    }

    #[must_use]
    pub const fn units_completed_before_start(&self) -> u64 {
        self.units_completed_before_start
    }

    #[must_use]
    pub const fn started_at(&self) -> Instant {
        self.started_at
    }

    #[must_use]
    pub const fn last_report_at(&self) -> Option<Instant> {
        self.last_report_at
    }

    #[must_use]
    pub const fn report_interval(&self) -> Duration {
        self.report_interval
    }

    #[must_use]
    pub const fn stage_name(&self) -> &'static str {
        self.stage_name
    }
}
