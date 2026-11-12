//! The request/reply exchange of the `speak` journey. The audio comes back
//! in bounded frames, then one terminal frame says how it was spoken. While
//! the daemon works, an elapsed-time line goes to standard error on a fixed
//! interval, and the wait ends at a hard bound.

use std::io::Write;
use std::time::Duration;

use astronomical_ipc_protocol::{
    DaemonIpcClient, DaemonRequest, DaemonResponse, SpeechResultMetadata,
};
use tokio::time::{Instant, interval_at, sleep, timeout};

use super::SpeakError;

/// The bounds one exchange runs under.
pub(super) struct SpeakExchangeBounds {
    pub(super) request_timeout: Duration,
    pub(super) generation_bound: Duration,
    pub(super) progress_interval: Duration,
}

/// The whole audio and how it was spoken.
#[derive(Debug)]
pub(super) struct SpokenReply {
    pub(super) model: String,
    pub(super) pcm16_bytes: Vec<u8>,
    pub(super) metadata: SpeechResultMetadata,
}

/// Sends the request, then collects audio frames until the terminal frame.
pub(super) async fn exchange_speak_request(
    daemon_client: &mut DaemonIpcClient,
    speak_request: &DaemonRequest,
    exchange_bounds: &SpeakExchangeBounds,
    progress_writer: &mut dyn Write,
) -> Result<SpokenReply, SpeakError> {
    match timeout(
        exchange_bounds.request_timeout,
        daemon_client.send_request(speak_request),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => return Err(SpeakError::DaemonStoppedResponding),
    }
    let started_at = Instant::now();
    let deadline_future = sleep(exchange_bounds.generation_bound);
    tokio::pin!(deadline_future);
    let mut progress_ticker = interval_at(
        started_at + exchange_bounds.progress_interval,
        exchange_bounds.progress_interval,
    );
    let mut pcm16_bytes = Vec::new();
    loop {
        tokio::select! {
            reply = daemon_client.next_response() => {
                let Ok(Some(daemon_response)) = reply else {
                    return Err(SpeakError::DaemonStoppedResponding);
                };
                if let Some(spoken_reply) = take_frame(daemon_response, &mut pcm16_bytes)? {
                    return Ok(spoken_reply);
                }
            }
            _tick = progress_ticker.tick() => {
                let _ = writeln!(
                    progress_writer,
                    "speaking: {}s elapsed (gives up after {}s)",
                    started_at.elapsed().as_secs(),
                    exchange_bounds.generation_bound.as_secs()
                );
            }
            () = &mut deadline_future => {
                return Err(SpeakError::SpeechTimedOut {
                    seconds: exchange_bounds.generation_bound.as_secs(),
                });
            }
        }
    }
}

/// Appends an audio frame, or ends the exchange on a terminal frame.
pub(super) fn take_frame(
    daemon_response: DaemonResponse,
    pcm16_bytes: &mut Vec<u8>,
) -> Result<Option<SpokenReply>, SpeakError> {
    match daemon_response {
        DaemonResponse::SpeechAudio {
            pcm16_bytes: audio_frame,
        } => {
            pcm16_bytes.extend_from_slice(&audio_frame);
            Ok(None)
        }
        DaemonResponse::SpeechCompleted { model, metadata } => Ok(Some(SpokenReply {
            model,
            pcm16_bytes: std::mem::take(pcm16_bytes),
            metadata,
        })),
        DaemonResponse::SpeechFailed { reason } => Err(SpeakError::SpeechFailed { reason }),
        DaemonResponse::GenerationRejected { reason } => Err(SpeakError::SpeechRejected { reason }),
        // No other frame answers a speech request; treat the daemon as gone
        // rather than guessing.
        _ => Err(SpeakError::DaemonStoppedResponding),
    }
}
