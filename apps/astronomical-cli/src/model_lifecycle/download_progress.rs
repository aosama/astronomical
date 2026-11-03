//! What a person watches while a model downloads: a live line with percent,
//! sizes, transfer rate and time remaining, and the judgement of when a
//! download has stopped and the wait should end.
//!
//! The daemon's download job reports only bytes completed and bytes total, so
//! rate and time remaining are worked out here from successive polls. Both
//! types take the elapsed time as an argument instead of reading a clock, so
//! every rule is testable without sleeping.
//!
//! The wait ends on lack of progress, not on a fixed clock: a 15 GB model on
//! an ordinary connection takes many minutes, and cutting it off at a fixed
//! two minutes (the earlier behavior) left a first-time user unable to finish
//! their first command. A download that keeps moving is allowed to finish.

use std::collections::VecDeque;
use std::time::Duration;

use astronomical_ipc_protocol::DaemonDownloadJob;

use super::render_download_progress;

/// Bytes moved within this much recent time make the transfer rate. Long
/// enough to smooth the burstiness of ranged segments, short enough that the
/// rate follows a connection that speeds up or slows down.
const RATE_WINDOW: Duration = Duration::from_secs(20);

/// Verifying and publishing a model move no bytes yet can take minutes (a
/// checksum over many gigabytes), so they get this many times the stall bound
/// before the wait gives up on them.
const FINALIZE_STALL_MULTIPLIER: u32 = 10;

const BYTES_PER_MEGABYTE: f64 = 1_048_576.0;
const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3_600;
const SECONDS_PER_DAY: u64 = 86_400;

/// Turns successive polls into a live progress line.
pub(crate) struct DownloadProgressTracker {
    /// (elapsed since the wait began, bytes completed), oldest first.
    samples: VecDeque<(Duration, u64)>,
    /// The longest line printed so far, in characters.
    longest_line_length: usize,
}

impl DownloadProgressTracker {
    pub(crate) fn new() -> Self {
        DownloadProgressTracker {
            samples: VecDeque::new(),
            longest_line_length: 0,
        }
    }

    /// Records one poll. A byte count that moved backwards (the daemon
    /// reconciled against the files on disk after an interruption) starts the
    /// window over: a rate measured across that jump would be meaningless.
    pub(crate) fn record(&mut self, elapsed: Duration, bytes_completed: u64) {
        if let Some((_, previous_bytes)) = self.samples.back()
            && bytes_completed < *previous_bytes
        {
            self.samples.clear();
        }
        self.samples.push_back((elapsed, bytes_completed));
        while let Some((oldest_elapsed, _)) = self.samples.front() {
            if elapsed.saturating_sub(*oldest_elapsed) <= RATE_WINDOW {
                break;
            }
            self.samples.pop_front();
        }
    }

    /// Bytes per second over the recent window, or `None` until two polls
    /// have been seen and some bytes moved between them.
    pub(crate) fn bytes_per_second(&self) -> Option<f64> {
        let (oldest_elapsed, oldest_bytes) = self.samples.front()?;
        let (newest_elapsed, newest_bytes) = self.samples.back()?;
        let seconds = newest_elapsed.saturating_sub(*oldest_elapsed).as_secs_f64();
        let bytes_moved = newest_bytes.saturating_sub(*oldest_bytes);
        if seconds <= 0.0 || bytes_moved == 0 {
            return None;
        }
        Some(bytes_moved as f64 / seconds)
    }

    /// The line for this poll: the base progress text plus, while bytes are
    /// flowing, the rate and the time remaining. Padded to the longest line
    /// so far, because the terminal rewrites it in place and a shorter line
    /// would otherwise leave the end of the previous one behind.
    pub(crate) fn render_line(&mut self, download_job: &DaemonDownloadJob) -> String {
        let mut line = render_download_progress(download_job);
        if download_job.state == "downloading"
            && let Some(rate) = self.bytes_per_second()
        {
            line.push_str(&format!(" · {}", format_rate(rate)));
            let remaining_bytes = download_job
                .bytes_total
                .saturating_sub(download_job.bytes_completed);
            line.push_str(&format!(
                " · {}",
                format_time_remaining(remaining_bytes as f64 / rate)
            ));
        }
        self.pad_to_longest(line)
    }

    fn pad_to_longest(&mut self, mut line: String) -> String {
        let line_length = line.chars().count();
        if line_length > self.longest_line_length {
            self.longest_line_length = line_length;
        }
        for _padding in line_length..self.longest_line_length {
            line.push(' ');
        }
        line
    }
}

/// `12.3 MB/s`, or `1.2 GB/s` from a thousand megabytes a second up. The
/// megabyte is 2²⁰ bytes, matching the gigabyte the rest of the CLI prints.
pub(crate) fn format_rate(bytes_per_second: f64) -> String {
    let megabytes_per_second = bytes_per_second / BYTES_PER_MEGABYTE;
    if megabytes_per_second >= 1_000.0 {
        return format!("{:.1} GB/s", megabytes_per_second / 1_024.0);
    }
    format!("{megabytes_per_second:.1} MB/s")
}

/// `about 45s left`, `about 3m 40s left`, `about 2h 05m left`, and a plain
/// "over a day" past that, where a precise figure would only mislead.
pub(crate) fn format_time_remaining(remaining_seconds: f64) -> String {
    if !remaining_seconds.is_finite() || remaining_seconds < 0.0 {
        return "time left unknown".to_owned();
    }
    let total_seconds = remaining_seconds.ceil() as u64;
    if total_seconds >= SECONDS_PER_DAY {
        return "over a day left".to_owned();
    }
    if total_seconds >= SECONDS_PER_HOUR {
        let hours = total_seconds / SECONDS_PER_HOUR;
        let minutes = (total_seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
        return format!("about {hours}h {minutes:02}m left");
    }
    if total_seconds >= SECONDS_PER_MINUTE {
        let minutes = total_seconds / SECONDS_PER_MINUTE;
        let seconds = total_seconds % SECONDS_PER_MINUTE;
        return format!("about {minutes}m {seconds:02}s left");
    }
    format!("about {total_seconds}s left")
}

/// Whether the wait should keep going.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum StallVerdict {
    Progressing,
    Stalled {
        /// How long nothing changed.
        stalled_for: Duration,
        /// The job state it stalled in.
        state: String,
    },
}

/// Watches for the moment progress stops. Any change to the job's state or
/// its byte count is progress; the clock for "nothing changed" restarts then.
pub(crate) struct StallWatch {
    stall_bound: Duration,
    last_signature: Option<(String, u64)>,
    last_change_at: Duration,
}

impl StallWatch {
    pub(crate) fn new(stall_bound: Duration) -> Self {
        StallWatch {
            stall_bound,
            last_signature: None,
            last_change_at: Duration::ZERO,
        }
    }

    pub(crate) fn observe(
        &mut self,
        elapsed: Duration,
        state: &str,
        bytes_completed: u64,
    ) -> StallVerdict {
        let changed = match &self.last_signature {
            Some((last_state, last_bytes)) => last_state != state || *last_bytes != bytes_completed,
            None => true,
        };
        if changed {
            self.last_signature = Some((state.to_owned(), bytes_completed));
            self.last_change_at = elapsed;
            return StallVerdict::Progressing;
        }
        let stalled_for = elapsed.saturating_sub(self.last_change_at);
        if stalled_for >= self.bound_for_state(state) {
            return StallVerdict::Stalled {
                stalled_for,
                state: state.to_owned(),
            };
        }
        StallVerdict::Progressing
    }

    fn bound_for_state(&self, state: &str) -> Duration {
        match state {
            "verifying" | "publishing" => self.stall_bound * FINALIZE_STALL_MULTIPLIER,
            _ => self.stall_bound,
        }
    }
}

#[cfg(test)]
#[path = "download_progress_tests.rs"]
mod download_progress_tests;
