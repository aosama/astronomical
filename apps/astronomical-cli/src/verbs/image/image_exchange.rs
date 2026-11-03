//! The one request/reply exchange of the `image` journey, with the waiting
//! made visible: an image takes minutes, and a silent terminal reads as a
//! hang. While the daemon works, a progress line goes to standard error on a
//! fixed interval, and the wait ends at a hard bound so a wedged worker
//! cannot hold the calling script forever.

use std::io::Write;
use std::time::Duration;

use astronomical_ipc_protocol::{DaemonIpcClient, DaemonRequest, DaemonResponse};
use tokio::time::{Instant, interval_at, sleep, timeout};

use super::ImageError;

/// The bounds one exchange runs under.
pub(super) struct ImageExchangeBounds {
    /// For sending the request.
    pub(super) request_timeout: Duration,
    /// For the whole wait on the image.
    pub(super) generation_bound: Duration,
    /// Between progress lines.
    pub(super) progress_interval: Duration,
}

/// Sends the request and waits for the daemon's single terminal frame.
pub(super) async fn exchange_image_request(
    daemon_client: &mut DaemonIpcClient,
    image_request: &DaemonRequest,
    exchange_bounds: &ImageExchangeBounds,
    progress_writer: &mut dyn Write,
) -> Result<DaemonResponse, ImageError> {
    match timeout(
        exchange_bounds.request_timeout,
        daemon_client.send_request(image_request),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => {
            return Err(ImageError::DaemonStoppedResponding);
        }
    }
    wait_for_reply(daemon_client, exchange_bounds, progress_writer).await
}

/// Waits for the reply, writing an elapsed-time line on every interval.
///
/// The elapsed time is the honest figure: the daemon reports no partial
/// progress for a diffusion run, and a guessed percentage would be wrong
/// more often than right. The line names the bound so the wait has an end.
async fn wait_for_reply(
    daemon_client: &mut DaemonIpcClient,
    exchange_bounds: &ImageExchangeBounds,
    progress_writer: &mut dyn Write,
) -> Result<DaemonResponse, ImageError> {
    let started_at = Instant::now();
    let reply_future = daemon_client.next_response();
    tokio::pin!(reply_future);
    let deadline_future = sleep(exchange_bounds.generation_bound);
    tokio::pin!(deadline_future);
    let mut progress_ticker = interval_at(
        started_at + exchange_bounds.progress_interval,
        exchange_bounds.progress_interval,
    );
    loop {
        tokio::select! {
            reply = &mut reply_future => {
                return match reply {
                    Ok(Some(daemon_response)) => Ok(daemon_response),
                    Ok(None) | Err(_) => Err(ImageError::DaemonStoppedResponding),
                };
            }
            _tick = progress_ticker.tick() => {
                write_progress_line(progress_writer, started_at.elapsed(), exchange_bounds);
            }
            () = &mut deadline_future => {
                return Err(ImageError::GenerationTimedOut {
                    seconds: exchange_bounds.generation_bound.as_secs(),
                });
            }
        }
    }
}

fn write_progress_line(
    progress_writer: &mut dyn Write,
    elapsed: Duration,
    exchange_bounds: &ImageExchangeBounds,
) {
    // Best effort: a closed standard error must not abort an image that is
    // already being drawn.
    let _write_result = writeln!(
        progress_writer,
        "generating: {}s elapsed (gives up after {}s)",
        elapsed.as_secs(),
        exchange_bounds.generation_bound.as_secs()
    );
    let _flush_result = progress_writer.flush();
}
