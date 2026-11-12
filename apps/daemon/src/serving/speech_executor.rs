//! Shared speech result types for the daemon HTTP and IPC boundaries.
//!
//! Live dispatch lives on `ImageGenerationExecutor`, as it does for
//! embeddings and decisions: `ApplicationState` holds one executor trait
//! object for every kind of request.

use std::time::Duration;

use astronomical_ipc_protocol::SpeechResultMetadata;
use astronomical_speech_contract::SpeechFailure;
use tokio::sync::mpsc;

/// One request's whole audio, as 16-bit little-endian mono samples, and how
/// it was spoken.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechOutput {
    pub pcm16_bytes: Vec<u8>,
    pub result_metadata: SpeechResultMetadata,
}

/// Failure delivered after a speech request was admitted to the worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpeechExecutionError {
    WorkerFailure(SpeechFailure),
    DeadlineExceeded,
    WorkerUnavailable,
}

/// Daemon-owned bounds for one speech request. A request reports progress
/// after every segment, so a stuck engine is caught by the stall bound long
/// before the whole-request bound, which only caps the longest text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeechTimeouts {
    pub execution_timeout: Duration,
    pub progress_stall_timeout: Duration,
}

impl SpeechTimeouts {
    /// The bounds are on compute time, not on audio length. The longest
    /// request (4,096 characters at the slowest pace) is about eighteen
    /// minutes of audio, which Kokoro speaks roughly ten times faster than
    /// it plays. One segment normally takes seconds; the stall bound matches
    /// the image one, leaving room for a slow pass on a Mac that is paging.
    pub const DEFAULT: Self = Self {
        execution_timeout: Duration::from_secs(10 * 60),
        progress_stall_timeout: Duration::from_secs(3 * 60),
    };
}

/// The channel one speech request's terminal outcome is delivered on.
pub(crate) type SpeechResultSender = mpsc::Sender<Result<SpeechOutput, SpeechExecutionError>>;

/// Resolves when the caller disconnects from an in-flight speech request.
pub(crate) async fn wait_for_speech_disconnect(speech_result_sender: Option<SpeechResultSender>) {
    let Some(speech_result_sender) = speech_result_sender else {
        std::future::pending::<()>().await;
        return;
    };
    speech_result_sender.closed().await;
}
