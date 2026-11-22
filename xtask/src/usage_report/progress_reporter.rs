//! Progress and time left for the report's slowest step, on standard error
//! so the report on standard output stays clean enough to paste or save.

use std::time::{Duration, Instant};

/// How often progress is printed, matching the repository rule that
/// long-running tools report progress and time left every 5 seconds.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Prints `step: done/total (percent)` with elapsed time and time left.
pub struct ProgressReporter {
    step_name: &'static str,
    total_count: usize,
    started_at: Instant,
    last_reported_at: Option<Instant>,
}

impl ProgressReporter {
    pub fn start(step_name: &'static str, total_count: usize) -> Self {
        eprintln!("[usage-report] {step_name}: 0/{total_count} (0%)");
        Self {
            step_name,
            total_count,
            started_at: Instant::now(),
            last_reported_at: None,
        }
    }

    /// Prints progress when 5 seconds have passed since the last line, and
    /// always on the final item.
    pub fn record_completed(&mut self, completed_count: usize) {
        let is_final_item = completed_count >= self.total_count;
        let last_reported_at = self.last_reported_at.unwrap_or(self.started_at);
        if !is_final_item && last_reported_at.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        self.last_reported_at = Some(Instant::now());
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64();
        let percent_complete = completed_count * 100 / self.total_count.max(1);
        let remaining_seconds =
            estimated_seconds_left(elapsed_seconds, completed_count, self.total_count);
        eprintln!(
            "[usage-report] {}: {completed_count}/{} ({percent_complete}%) elapsed {elapsed_seconds:.0}s, about {remaining_seconds:.0}s left",
            self.step_name, self.total_count
        );
    }
}

/// Time left at the average pace so far; zero once everything is done.
pub fn estimated_seconds_left(
    elapsed_seconds: f64,
    completed_count: usize,
    total_count: usize,
) -> f64 {
    if completed_count == 0 || completed_count >= total_count {
        return 0.0;
    }
    elapsed_seconds / completed_count as f64 * (total_count - completed_count) as f64
}
