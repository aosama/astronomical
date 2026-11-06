//! Waits for the daemon's next frame without mistaking a slow model load for a
//! daemon that has gone away.
//!
//! The daemon sends nothing on a generation's connection until the model has
//! loaded, and a load may use most of the daemon's 60-second budget before the
//! request's own silent work, queueing and prefill, has even begun.
//! Lengthening a flat timeout would only make a truly stuck daemon take longer to
//! report, so the bound stays on silence, and a load report counts as hearing from
//! the daemon. The daemon already enforces its own budget on the load and clears the
//! report when it gives up.

use std::{future::Future, io::Write, time::Duration};

use astronomical_ipc_protocol::ModelLoadInFlight;

use super::RespondError;
use crate::{DaemonProbe, formatting::MODEL_LOAD_LINE};

/// How long the wait may go unanswered, and how often to ask about a load.
#[derive(Clone, Copy, Debug)]
pub(super) struct DaemonWaitBounds {
    /// How long the daemon may say nothing at all.
    pub(super) silence_bound: Duration,
    /// How long one load may keep being reported, for a daemon that has wedged but
    /// still answers status.
    pub(super) load_stall_bound: Duration,
    pub(super) load_poll_interval: Duration,
}

/// Remembers which load was last reported, so only a new one restarts the stall
/// bound and redraws the line.
#[derive(Debug, Default)]
pub(super) struct ModelLoadWatch {
    last_report: Option<ModelLoadInFlight>,
    live_line_width: usize,
}

impl ModelLoadWatch {
    /// Whether this report is about a load the last one was not.
    fn reports_a_new_load(&mut self, model_load_in_flight: &ModelLoadInFlight) -> bool {
        let is_new_load = self.last_report.as_ref() != Some(model_load_in_flight);
        self.last_report = Some(model_load_in_flight.clone());
        is_new_load
    }

    /// Rewrites the live line in place, padded so a shorter line leaves nothing
    /// of a longer one behind. A closed stderr must not abort the answer.
    fn draw_live_line(&mut self, progress_line: &str, stderr: &mut dyn Write) {
        let padded_width = self.live_line_width.max(progress_line.len());
        let _ = write!(stderr, "\r{progress_line:<padded_width$}");
        let _ = stderr.flush();
        self.live_line_width = padded_width;
    }

    /// Ends the live line, so the answer's own stderr output starts on a new one.
    fn finish_live_line(&mut self, stderr: &mut dyn Write) {
        if self.live_line_width > 0 {
            let _ = writeln!(stderr);
            let _ = stderr.flush();
            self.live_line_width = 0;
        }
    }
}

/// Waits for `frame_wait`, polling the load's progress meanwhile.
///
/// The same pinned future stays alive across polls because a frame read that is
/// dropped partway through would lose the bytes it already consumed.
pub(super) async fn await_frame_watching_model_load<FrameWait>(
    frame_wait: FrameWait,
    daemon_probe: &DaemonProbe,
    wait_bounds: DaemonWaitBounds,
    model_load_watch: &mut ModelLoadWatch,
    stderr: &mut dyn Write,
) -> Result<FrameWait::Output, RespondError>
where
    FrameWait: Future,
{
    tokio::pin!(frame_wait);
    let wait_started_at = tokio::time::Instant::now();
    let mut silence_deadline = wait_started_at + wait_bounds.silence_bound;
    let mut current_load_reported_since = wait_started_at;
    loop {
        tokio::select! {
            daemon_frame = &mut frame_wait => {
                model_load_watch.finish_live_line(stderr);
                return Ok(daemon_frame);
            }
            () = tokio::time::sleep_until(silence_deadline) => {
                model_load_watch.finish_live_line(stderr);
                return Err(RespondError::DaemonStoppedResponding);
            }
            latest_report = next_load_report(daemon_probe, wait_bounds.load_poll_interval) => {
                let Some(model_load_in_flight) = latest_report else {
                    continue;
                };
                let heard_at = tokio::time::Instant::now();
                if model_load_watch.reports_a_new_load(&model_load_in_flight) {
                    current_load_reported_since = heard_at;
                    model_load_watch.draw_live_line(MODEL_LOAD_LINE, stderr);
                }
                if heard_at.duration_since(current_load_reported_since) < wait_bounds.load_stall_bound {
                    silence_deadline = heard_at + wait_bounds.silence_bound;
                }
            }
        }
    }
}

/// Waits one interval, then asks the daemon. A failed probe reads as no report:
/// the frame wait is what notices a daemon that has really gone.
async fn next_load_report(
    daemon_probe: &DaemonProbe,
    load_poll_interval: Duration,
) -> Option<ModelLoadInFlight> {
    tokio::time::sleep(load_poll_interval).await;
    match daemon_probe.status_snapshot().await {
        Ok(status_snapshot) => status_snapshot.model_load_in_flight,
        Err(_probe_error) => None,
    }
}

#[cfg(test)]
#[path = "frame_wait_tests.rs"]
mod tests;
