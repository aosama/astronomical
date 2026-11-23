//! Speech as a request kind: what its admission checks and dispatches beyond
//! the shared steps in [`request_kind_admission`], and the bounds its active
//! request runs under.
//!
//! A speech request is stepped like an image: the worker sends each spoken
//! segment as it is ready, so the active request is bounded by a progress
//! stall and by how much audio it may return, as well as by a whole-request
//! deadline. Before it runs, it needs only what a forward-pass request needs,
//! so its admission is theirs.
//!
//! [`request_kind_admission`]: crate::worker::request_kind_admission

use std::sync::{Arc, RwLock};

use ipc_protocol::{RequestId, SpeechCommand};
use speech_contract::SLOWEST_SPEED;

use crate::serving::speech_executor::{SpeechExecutionError, SpeechOutput};
use crate::worker::loop_types::{ActiveSpeechGeneration, ActiveWorkerRequest, WorkerLoopRuntime};
use crate::worker::request_kind_admission::{AdmittedRequest, RequestKind};
use crate::{
    SpeechTimeouts, WorkerActivity, WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
};

pub(in crate::worker) struct SpeechRequestKind;

/// The limits one admitted speech request runs under, beyond its start.
pub(in crate::worker) struct SpeechBounds {
    speech_timeouts: SpeechTimeouts,
    maximum_audio_bytes: usize,
}

impl RequestKind for SpeechRequestKind {
    type Work = SpeechCommand;
    type ResultMessage = Result<SpeechOutput, SpeechExecutionError>;
    type RequestBounds = SpeechBounds;
    const COMMAND_NAME: &'static str = "GenerateSpeech";
    const STARTED_ACTIVITY: WorkerActivity = WorkerActivity::Speaking;

    fn requested_model(speech_command: &SpeechCommand) -> &str {
        &speech_command.model
    }

    fn request_id(speech_command: &SpeechCommand) -> RequestId {
        speech_command.request_id
    }

    fn request_bounds(speech_command: &SpeechCommand, runtime: &WorkerLoopRuntime) -> SpeechBounds {
        SpeechBounds {
            speech_timeouts: runtime.speech_timeouts,
            maximum_audio_bytes: maximum_audio_bytes_for(speech_command.speed),
        }
    }

    fn resident_model_accepts(
        health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
        _speech_command: &SpeechCommand,
    ) -> bool {
        resident_model_speaks(health_snapshot)
    }

    async fn dispatch(
        worker_process: &mut WorkerProcess,
        speech_command: SpeechCommand,
    ) -> Result<(), WorkerControlError> {
        worker_process.start_speech_generation(speech_command).await
    }

    fn active_request(
        admitted_request: AdmittedRequest<Self::ResultMessage>,
        speech_bounds: SpeechBounds,
    ) -> ActiveWorkerRequest {
        ActiveWorkerRequest::Speech(active_speech_generation(admitted_request, speech_bounds))
    }
}

/// The most audio a request at natural pace may return: about twelve
/// minutes at 24 kHz, more than twice what the longest accepted text takes
/// to say. A slower pace stretches the same text, so the bound grows with it.
const MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE: usize = 32 * 1024 * 1024;

fn maximum_audio_bytes_for(speed: f32) -> usize {
    // Clamped to the contract's slowest pace (NaN included), so the bound
    // never exceeds four times the natural one whatever a caller sent.
    let bounded_speed = if speed.is_nan() {
        1.0
    } else {
        speed.clamp(SLOWEST_SPEED, 1.0)
    };
    let slowness = 1.0 / f64::from(bounded_speed);
    (MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE as f64 * slowness).ceil() as usize
}

/// Whether the resident model advertises a speech surface.
fn resident_model_speaks(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> bool {
    health_snapshot.read().ok().is_some_and(|snapshot| {
        snapshot
            .ready_model_capabilities
            .as_ref()
            .is_some_and(|capabilities| capabilities.speech.is_some())
    })
}

fn active_speech_generation(
    admitted_request: AdmittedRequest<Result<SpeechOutput, SpeechExecutionError>>,
    speech_bounds: SpeechBounds,
) -> ActiveSpeechGeneration {
    let AdmittedRequest {
        active_generation_permit,
        request_id,
        execution_started_at,
        result_sender,
    } = admitted_request;
    let SpeechBounds {
        speech_timeouts,
        maximum_audio_bytes,
    } = speech_bounds;
    ActiveSpeechGeneration {
        _active_generation_permit: active_generation_permit,
        request_id,
        execution_deadline: execution_started_at + speech_timeouts.execution_timeout,
        progress_stall_deadline: execution_started_at + speech_timeouts.progress_stall_timeout,
        progress_stall_timeout: speech_timeouts.progress_stall_timeout,
        pcm16_bytes: Vec::new(),
        maximum_audio_bytes,
        has_exceeded_audio_limit: false,
        completed_segments: 0,
        terminal_received_at: None,
        speech_result_sender: result_sender,
        terminal_outcome: None,
    }
}

#[cfg(test)]
#[path = "speech_request_tests.rs"]
mod speech_request_tests;
