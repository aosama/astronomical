//! Admits one speech command through the shared model-swap and
//! active-request owner.
//!
//! A speech request is stepped like an image: the worker sends each spoken
//! segment as it is ready, so the active request is bounded by a progress
//! stall as well as a whole-request deadline. Before it runs, it needs what a
//! forward-pass request needs (the worker's startup configuration, the
//! requested model resident, the client still there), so admission reuses
//! those shared steps.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{RequestId, SpeechCommand};
use tokio::sync::{OwnedSemaphorePermit, oneshot};
use tokio::time::Instant;

use crate::serving::speech_executor::SpeechResultSender;
use crate::worker::command_arms::reject_unavailable_worker;
use crate::worker::health::{clear_active_request_progress, publish_activity};
use crate::worker::loop_types::{
    ActiveSpeechGeneration, ActiveWorkerRequest, WorkerLoopCommand, WorkerLoopRuntime,
};
use crate::worker::memory::limit::take_generation_start_after_pending_memory_limit;
use crate::worker::process::containment::{cancel_active_generation, contain_worker_failure};
use crate::worker::request_admission::{
    RequestAdmissionContext, RequestAdmissionOutcome, classify_dispatch_result,
    make_requested_model_resident,
};
use astronomical_speech_contract::SLOWEST_SPEED;

use crate::{GenerationStartError, SpeechTimeouts, WorkerActivity, WorkerHealthSnapshot};

/// Handles a speech command: refuse while the worker is unavailable or busy,
/// let a pending memory limit settle, admit it, and contain the worker when
/// admission fails.
pub(in crate::worker) async fn admit_speech_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateSpeech {
        active_generation_permit,
        speech_command,
        start_sender,
        speech_result_sender,
    } = routed_command
    else {
        return;
    };
    if reject_unavailable_worker(runtime.is_ready, is_worker_running) {
        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
        return;
    }
    if runtime.active_generation.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!("received GenerateSpeech while another request is active");
        return;
    }
    let Some(start_sender) = take_generation_start_after_pending_memory_limit(
        &mut runtime.pending_mlx_memory_limit_update,
        &mut runtime.worker_process,
        runtime.model_load_timeout,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        start_sender,
    )
    .await
    else {
        return;
    };
    let speech_admission = SpeechAdmission {
        active_generation_permit,
        start_sender,
        speech_result_sender,
        maximum_audio_bytes: maximum_audio_bytes_for(speech_command.speed),
    };
    admit_speech_request(runtime, speech_command, speech_admission).await;
}

/// What a speech request owns while it is admitted.
struct SpeechAdmission {
    active_generation_permit: OwnedSemaphorePermit,
    start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
    speech_result_sender: SpeechResultSender,
    maximum_audio_bytes: usize,
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

async fn admit_speech_request(
    runtime: &mut WorkerLoopRuntime,
    speech_command: SpeechCommand,
    speech_admission: SpeechAdmission,
) {
    let admission_outcome = {
        let mut admission_context = RequestAdmissionContext {
            worker_process: &mut runtime.worker_process,
            health_snapshot: &runtime.health_snapshot,
            is_ready: &mut runtime.is_ready,
            model_load_deadline: &mut runtime.model_load_deadline,
            active_request: &mut runtime.active_generation,
            performance_log: &mut runtime.performance_log,
            completion_log: &mut runtime.completion_log,
            model_policy_catalog: &runtime.model_policy_catalog,
            model_load_timeout: runtime.model_load_timeout,
        };
        dispatch_speech_command(
            &mut admission_context,
            speech_command,
            &speech_admission.speech_result_sender,
        )
        .await
    };
    match admission_outcome {
        Ok(RequestAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        }) => {
            start_speech_request(runtime, speech_admission, request_id, execution_started_at).await
        }
        Ok(RequestAdmissionOutcome::ClientDisconnected) => {}
        Ok(RequestAdmissionOutcome::StartRejected(start_error)) => {
            let _send_outcome = speech_admission.start_sender.send(Err(start_error));
        }
        Ok(RequestAdmissionOutcome::WorkerFailed(control_error)) | Err(control_error) => {
            let _send_outcome = speech_admission
                .start_sender
                .send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(
                &mut runtime.worker_process,
                &runtime.health_snapshot,
                &mut runtime.active_generation,
                control_error,
            )
            .await;
            runtime.is_ready = false;
        }
    }
}

/// Makes the requested model resident, checks that it speaks, and hands the
/// command to the worker.
async fn dispatch_speech_command(
    admission_context: &mut RequestAdmissionContext<'_>,
    speech_command: SpeechCommand,
    speech_result_sender: &SpeechResultSender,
) -> Result<RequestAdmissionOutcome, crate::WorkerControlError> {
    if let Some(admission_ending_outcome) = make_requested_model_resident(
        admission_context,
        &speech_command.model,
        speech_result_sender,
    )
    .await?
    {
        return Ok(admission_ending_outcome);
    }
    if !resident_model_speaks(admission_context.health_snapshot) {
        return Ok(RequestAdmissionOutcome::StartRejected(
            GenerationStartError::WorkerUnavailable,
        ));
    }
    let request_id = speech_command.request_id;
    let execution_started_at = Instant::now();
    let dispatch_result = admission_context
        .worker_process
        .start_speech_generation(speech_command)
        .await;
    Ok(classify_dispatch_result(
        request_id,
        execution_started_at,
        dispatch_result,
    ))
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

/// Takes the active-request slot, confirms the start, and cancels the
/// request at once when the caller already hung up.
async fn start_speech_request(
    runtime: &mut WorkerLoopRuntime,
    speech_admission: SpeechAdmission,
    request_id: RequestId,
    execution_started_at: Instant,
) {
    let active_speech = active_speech_generation(
        speech_admission.active_generation_permit,
        request_id,
        SpeechBounds {
            execution_started_at,
            speech_timeouts: runtime.speech_timeouts,
            maximum_audio_bytes: speech_admission.maximum_audio_bytes,
        },
        speech_admission.speech_result_sender,
    );
    runtime.active_generation = Some(ActiveWorkerRequest::Speech(active_speech));
    clear_active_request_progress(&runtime.health_snapshot);
    publish_activity(&runtime.health_snapshot, WorkerActivity::Speaking);
    if speech_admission.start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            &mut runtime.worker_process,
            &runtime.health_snapshot,
            &mut runtime.active_generation,
            runtime.cancellation_acknowledgement_timeout,
            runtime.model_load_timeout,
            &mut runtime.is_ready,
        )
        .await;
    }
}

/// The limits one admitted speech request runs under.
struct SpeechBounds {
    execution_started_at: Instant,
    speech_timeouts: SpeechTimeouts,
    maximum_audio_bytes: usize,
}

fn active_speech_generation(
    active_generation_permit: OwnedSemaphorePermit,
    request_id: RequestId,
    speech_bounds: SpeechBounds,
    speech_result_sender: SpeechResultSender,
) -> ActiveSpeechGeneration {
    let SpeechBounds {
        execution_started_at,
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
        speech_result_sender,
        terminal_outcome: None,
    }
}

#[cfg(test)]
#[path = "speech_request_tests.rs"]
mod speech_request_tests;
