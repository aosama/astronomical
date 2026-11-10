//! Progress lines for a conversion: how far it is and how long is left.

use std::time::{Duration, Instant};

/// A line at most this often, so a long conversion shows it is alive
/// without flooding the terminal.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) struct ConversionProgress {
    /// The command the lines belong to, such as `convert-laya`.
    command_label: &'static str,
    total_source_bytes: u64,
    converted_source_bytes: u64,
    started_at: Instant,
    last_reported_at: Instant,
}

impl ConversionProgress {
    pub(crate) fn start(command_label: &'static str, total_source_bytes: u64) -> Self {
        let started_at = Instant::now();
        Self {
            command_label,
            total_source_bytes,
            converted_source_bytes: 0,
            started_at,
            last_reported_at: started_at,
        }
    }

    /// Records `source_bytes` more as converted and prints a line when the
    /// interval has passed.
    pub(crate) fn advance(&mut self, source_bytes: u64) {
        self.converted_source_bytes += source_bytes;
        if self.last_reported_at.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        self.last_reported_at = Instant::now();
        println!(
            "{}",
            progress_line(
                self.command_label,
                self.converted_source_bytes,
                self.total_source_bytes,
                self.started_at.elapsed()
            )
        );
    }

    pub(crate) fn finish(&self) {
        println!(
            "[{}] status=converted percentage=100 ETA_seconds=0 elapsed_seconds={}",
            self.command_label,
            self.started_at.elapsed().as_secs()
        );
    }
}

/// One progress line. The time left assumes the rest converts at the rate
/// seen so far, which holds because every byte costs about the same.
pub(crate) fn progress_line(
    command_label: &str,
    converted_source_bytes: u64,
    total_source_bytes: u64,
    elapsed: Duration,
) -> String {
    // An empty checkpoint has nothing left to convert, so it is complete.
    let percentage = (converted_source_bytes * 100)
        .checked_div(total_source_bytes)
        .unwrap_or(100);
    let remaining_source_bytes = total_source_bytes.saturating_sub(converted_source_bytes);
    let eta_seconds = if converted_source_bytes == 0 {
        0
    } else {
        (elapsed.as_secs_f64() * remaining_source_bytes as f64 / converted_source_bytes as f64)
            .round() as u64
    };
    format!(
        "[{command_label}] status=progress percentage={percentage} ETA_seconds={eta_seconds} \
         elapsed_seconds={}",
        elapsed.as_secs()
    )
}

#[cfg(test)]
#[path = "conversion_progress_tests.rs"]
mod tests;
