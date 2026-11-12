//! Admits one embeddings command through the shared model-swap and active-request owner.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{EmbeddingsCommand, RequestId};
use tokio::time::Instant;

use crate::worker::request_admission::{
    RequestAdmissionContext, RequestAdmissionOutcome, SINGLE_SHOT_EXECUTION_TIMEOUT,
    classify_dispatch_result, make_requested_model_resident,
};
use crate::{
    CompletionAttributionLog, EmbeddingsExecutionError, EmbeddingsOutput, GenerationPerformanceLog,
    GenerationStartError, RuntimeModelPolicy, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveEmbeddingsGeneration, ActiveWorkerRequest},
    worker::process::containment::{cancel_active_generation, contain_worker_failure},
};

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_generate_embeddings_command(
    worker_process: &mut WorkerProcess,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    embeddings_command: EmbeddingsCommand,
    start_sender: tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>,
    embeddings_result_sender: tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
    _admitted_at: Instant,
    _queue_wait_elapsed: std::time::Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    model_policy_catalog: &Arc<HashMap<String, RuntimeModelPolicy>>,
    model_load_timeout: std::time::Duration,
    cancellation_acknowledgement_timeout: std::time::Duration,
) -> Result<(), WorkerControlError> {
    if active_request.is_some() {
        let _send_outcome = start_sender.send(Err(GenerationStartError::CapacityUnavailable));
        tracing::error!("received GenerateEmbeddings while another request is active");
        return Ok(());
    }
    let mut admission_context = RequestAdmissionContext {
        worker_process: &mut *worker_process,
        health_snapshot,
        is_ready: &mut *is_ready,
        model_load_deadline: &mut *model_load_deadline,
        active_request: &mut *active_request,
        performance_log: &mut *performance_log,
        completion_log: &mut *completion_log,
        model_policy_catalog,
        model_load_timeout,
    };
    match admit_embeddings_command(
        &mut admission_context,
        embeddings_command,
        &embeddings_result_sender,
    )
    .await?
    {
        RequestAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        } => {
            admit_embeddings_request(
                active_request,
                active_generation_permit,
                request_id,
                execution_started_at,
                embeddings_result_sender,
                health_snapshot,
            );
        }
        RequestAdmissionOutcome::ClientDisconnected => return Ok(()),
        RequestAdmissionOutcome::StartRejected(start_error) => {
            let _send_outcome = start_sender.send(Err(start_error));
            return Ok(());
        }
        RequestAdmissionOutcome::WorkerFailed(start_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_worker_failure(worker_process, health_snapshot, active_request, start_error)
                .await;
            *is_ready = false;
            return Ok(());
        }
    }
    if start_sender.send(Ok(())).is_err() {
        cancel_active_generation(
            worker_process,
            health_snapshot,
            active_request,
            cancellation_acknowledgement_timeout,
            model_load_timeout,
            is_ready,
        )
        .await;
    }
    Ok(())
}

/// Runs the pre-execution admission pipeline: model residency through the
/// shared single-shot steps, then the embeddings capability gate and the
/// worker command dispatch. Failures come back as outcomes so the owning
/// handler answers the client and contains the worker.
async fn admit_embeddings_command(
    admission_context: &mut RequestAdmissionContext<'_>,
    embeddings_command: EmbeddingsCommand,
    embeddings_result_sender: &tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
) -> Result<RequestAdmissionOutcome, WorkerControlError> {
    if let Some(admission_ending_outcome) = make_requested_model_resident(
        admission_context,
        &embeddings_command.model,
        embeddings_result_sender,
    )
    .await?
    {
        return Ok(admission_ending_outcome);
    }
    if !embeddings_command_supported(
        admission_context.health_snapshot,
        embeddings_command.dimensions,
    ) {
        return Ok(RequestAdmissionOutcome::StartRejected(
            GenerationStartError::WorkerUnavailable,
        ));
    }
    let request_id = embeddings_command.request_id;
    let execution_started_at = Instant::now();
    let dispatch_result = admission_context
        .worker_process
        .start_embeddings_generation(embeddings_command)
        .await;
    Ok(classify_dispatch_result(
        request_id,
        execution_started_at,
        dispatch_result,
    ))
}

/// Checks the resident model's advertised embedding surface: the vector
/// width must be positive, and explicit dimensions may only truncate.
fn embeddings_command_supported(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    requested_dimensions: Option<u32>,
) -> bool {
    health_snapshot.read().ok().is_some_and(|snapshot| {
        snapshot
            .ready_model_capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.embeddings.as_ref())
            .is_some_and(|embedding_capabilities| {
                embedding_capabilities.vector_width > 0
                    && requested_dimensions
                        .is_none_or(|dimensions| dimensions <= embedding_capabilities.vector_width)
            })
    })
}

/// Takes ownership of the active-request slot for this command and clears
/// the stale per-request progress the previous request left behind.
fn admit_embeddings_request(
    active_request: &mut Option<ActiveWorkerRequest>,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: RequestId,
    execution_started_at: Instant,
    embeddings_result_sender: tokio::sync::mpsc::Sender<
        Result<EmbeddingsOutput, EmbeddingsExecutionError>,
    >,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    *active_request = Some(ActiveWorkerRequest::Embeddings(
        ActiveEmbeddingsGeneration {
            _active_generation_permit: active_generation_permit,
            request_id,
            terminal_received_at: None,
            execution_deadline: execution_started_at + SINGLE_SHOT_EXECUTION_TIMEOUT,
            embeddings_result_sender,
            terminal_outcome: None,
        },
    ));
    clear_active_request_progress(health_snapshot);
    publish_activity(health_snapshot, WorkerActivity::Idle);
}

#[cfg(test)]
#[path = "embeddings_request_tests.rs"]
mod embeddings_request_tests;
