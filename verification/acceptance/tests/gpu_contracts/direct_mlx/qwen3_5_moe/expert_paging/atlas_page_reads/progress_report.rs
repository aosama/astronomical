//! Five-second progress, percentage and ETA for the probe's long phases.

use std::time::{Duration, Instant};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

pub(super) struct ProgressReporter {
    phase_name: &'static str,
    total_unit_count: usize,
    last_report_at: Instant,
}

impl ProgressReporter {
    pub(super) fn new(phase_name: &'static str, total_unit_count: usize) -> Self {
        eprintln!("[atlas-page-reads] {phase_name}: starting {total_unit_count} units");
        Self {
            phase_name,
            total_unit_count,
            last_report_at: Instant::now(),
        }
    }

    /// Reports when five seconds have passed since the last report, and
    /// always at the final unit.
    pub(super) fn record(&mut self, completed_unit_count: usize, phase_started_at: Instant) {
        let is_final_unit = completed_unit_count == self.total_unit_count;
        if !is_final_unit && self.last_report_at.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        self.last_report_at = Instant::now();
        let elapsed_seconds = phase_started_at.elapsed().as_secs_f64();
        let completed_fraction = completed_unit_count as f64 / self.total_unit_count.max(1) as f64;
        let eta_seconds = elapsed_seconds / completed_fraction.max(f64::EPSILON) - elapsed_seconds;
        eprintln!(
            "[atlas-page-reads] {}: {completed_unit_count}/{} ({:.0}%) elapsed {elapsed_seconds:.1}s ETA {eta_seconds:.1}s",
            self.phase_name,
            self.total_unit_count,
            completed_fraction * 100.0
        );
    }
}
