//! Waits for the daemon's next frame without mistaking slow work for a daemon that
//! has gone away.
//!
//! The daemon sends nothing on a request's connection until it has something to answer
//! with: not while it loads the model, not while the request waits behind another, and
//! not during prefill, which a long prompt can stretch past any fixed limit.
//! Lengthening a flat timeout would only make a truly stuck daemon take longer to
//! report, so the bound stays on silence, and either of two things counts as hearing
//! from the daemon: a load report, or a request making progress. A stuck worker does
//! neither. The daemon enforces its own budget on a load and clears the report when
//! it gives up.
//!
//! The progress count covers the whole worker, not this request, because a request in
//! the queue has no progress of its own to report. So another request's progress keeps
//! this wait open with no upper limit, even if something has stalled only this
//! request's connection; a cap would bring the false failure back for a long queue.
//! Embeddings report no progress, so waiting behind a long embeddings request can
//! still reach the bound, and so can an embeddings request's own long batch.

use std::{future::Future, io::Write, time::Duration};

use astronomical_ipc_protocol::ModelLoadInFlight;
use tokio::time::Instant;

use crate::{DaemonProbe, DaemonStatusSnapshot, formatting::MODEL_LOAD_LINE};

/// How many poll intervals one status request may take. Answering status is a brief
/// read in the daemon, so a request this slow is lost; giving up on it lets the next
/// poll see the count before the silence bound runs out on a healthy wait.
const STATUS_ANSWER_POLL_INTERVALS: u32 = 5;

/// The wait ended with nothing heard from the daemon inside its bound. Each verb turns
/// this into its own "stopped responding" failure.
#[derive(Debug)]
pub(crate) struct DaemonWentSilent;

/// How long the wait may go unanswered, and how often to ask what the daemon is doing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DaemonWaitBounds {
    /// How long the daemon may say nothing at all.
    pub(crate) silence_bound: Duration,
    /// How long one load may keep being reported, for a daemon that has wedged but
    /// still answers status.
    pub(crate) load_stall_bound: Duration,
    pub(crate) work_poll_interval: Duration,
}

/// A load the daemon is reporting, and when this wait first heard of it.
#[derive(Debug)]
struct ReportedLoad {
    model_load_in_flight: ModelLoadInFlight,
    reported_since: Instant,
}

/// Remembers what the daemon last reported: only a new load restarts the stall bound
/// and redraws the line, and only a changed count is progress.
#[derive(Debug, Default)]
pub(crate) struct DaemonWorkWatch {
    reported_load: Option<ReportedLoad>,
    last_request_progress_count: Option<u64>,
    live_line_width: usize,
}

impl DaemonWorkWatch {
    /// Whether this status counts as hearing from the daemon. Both checks always run,
    /// because each one records what it saw for the next status to compare against.
    fn has_heard_from_daemon(
        &mut self,
        status_snapshot: &DaemonStatusSnapshot,
        heard_at: Instant,
        load_stall_bound: Duration,
        stderr: &mut dyn Write,
    ) -> bool {
        let does_load_count = self.note_load_report(
            status_snapshot.model_load_in_flight.as_ref(),
            heard_at,
            load_stall_bound,
            stderr,
        );
        let has_request_progressed =
            self.reports_request_progress(status_snapshot.request_progress_count);
        does_load_count || has_request_progressed
    }

    /// Records this report and says whether it counts as hearing from the daemon: a
    /// load is reported and has not stayed put past its bound. An ended load is forgotten
    /// and its line finished: one frame wait can span minutes of queue or prefill, so a
    /// later load of the same model is new, and the line must not go on saying the model
    /// is loading.
    fn note_load_report(
        &mut self,
        model_load_in_flight: Option<&ModelLoadInFlight>,
        heard_at: Instant,
        load_stall_bound: Duration,
        stderr: &mut dyn Write,
    ) -> bool {
        let Some(model_load_in_flight) = model_load_in_flight else {
            self.reported_load = None;
            self.finish_live_line(stderr);
            return false;
        };
        match &self.reported_load {
            Some(reported_load) if reported_load.model_load_in_flight == *model_load_in_flight => {
                heard_at.duration_since(reported_load.reported_since) < load_stall_bound
            }
            _ => {
                self.reported_load = Some(ReportedLoad {
                    model_load_in_flight: model_load_in_flight.clone(),
                    reported_since: heard_at,
                });
                self.draw_live_line(MODEL_LOAD_LINE, stderr);
                true
            }
        }
    }

    /// Whether a request has moved since the last report. The first report only sets
    /// the starting point, because a count that never changes again is a stuck worker
    /// and must not earn it another full silence bound.
    fn reports_request_progress(&mut self, request_progress_count: u64) -> bool {
        let has_progressed = matches!(
            self.last_request_progress_count,
            Some(last_count) if last_count != request_progress_count
        );
        self.last_request_progress_count = Some(request_progress_count);
        has_progressed
    }

    /// A frame proves the daemon is alive and whatever was loading has finished. Polls
    /// only run after a silence, so one may never have seen the report clear.
    fn forget_load_after_frame(&mut self, stderr: &mut dyn Write) {
        self.reported_load = None;
        self.finish_live_line(stderr);
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

/// Waits for `frame_wait`, asking what the daemon is doing meanwhile.
///
/// The same pinned future stays alive across polls because a frame read that is
/// dropped partway through would lose the bytes it already consumed.
pub(crate) async fn await_frame_watching_daemon_work<FrameWait>(
    frame_wait: FrameWait,
    daemon_probe: &DaemonProbe,
    wait_bounds: DaemonWaitBounds,
    daemon_work_watch: &mut DaemonWorkWatch,
    stderr: &mut dyn Write,
) -> Result<FrameWait::Output, DaemonWentSilent>
where
    FrameWait: Future,
{
    tokio::pin!(frame_wait);
    let mut silence_deadline = Instant::now() + wait_bounds.silence_bound;
    loop {
        tokio::select! {
            daemon_frame = &mut frame_wait => {
                daemon_work_watch.forget_load_after_frame(stderr);
                return Ok(daemon_frame);
            }
            () = tokio::time::sleep_until(silence_deadline) => {
                daemon_work_watch.finish_live_line(stderr);
                return Err(DaemonWentSilent);
            }
            latest_status = next_daemon_status(daemon_probe, wait_bounds.work_poll_interval) => {
                let Some(status_snapshot) = latest_status else {
                    continue;
                };
                let heard_at = Instant::now();
                if daemon_work_watch.has_heard_from_daemon(
                    &status_snapshot,
                    heard_at,
                    wait_bounds.load_stall_bound,
                    stderr,
                ) {
                    silence_deadline = heard_at + wait_bounds.silence_bound;
                }
            }
        }
    }
}

/// Waits one interval, then asks the daemon. A failed probe reads as no answer:
/// the frame wait is what notices a daemon that has really gone.
async fn next_daemon_status(
    daemon_probe: &DaemonProbe,
    work_poll_interval: Duration,
) -> Option<DaemonStatusSnapshot> {
    tokio::time::sleep(work_poll_interval).await;
    tokio::time::timeout(
        work_poll_interval * STATUS_ANSWER_POLL_INTERVALS,
        daemon_probe.status_snapshot(),
    )
    .await
    .ok()?
    .ok()
}

#[cfg(test)]
#[path = "daemon_frame_wait_tests.rs"]
mod tests;
