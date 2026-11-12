//! Admits one decisions command through the shared model-swap and active-request owner.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{DecisionsCommand, RequestId};
use tokio::time::Instant;

use crate::serving::decisions_executor::DecisionsResultSender;
use crate::worker::request_admission::{
    RequestAdmissionContext, RequestAdmissionOutcome, SINGLE_SHOT_EXECUTION_TIMEOUT,
    classify_dispatch_result, make_requested_model_resident,
};
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, GenerationStartError, RuntimeModelPolicy,
    WorkerActivity, WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveDecisionsGeneration, ActiveWorkerRequest},
    worker::process::containment::{cancel_active_generation, contain_worker_failure},
};

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_generate_decisions_command(
    worker_process: &mut WorkerProcess,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    decisions_command: DecisionsCommand,
    start_sender: tokio::sync::oneshot::Sender<Result<(), GenerationStartError>>,
    decisions_result_sender: DecisionsResultSender,
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
        tracing::error!("received GenerateDecisions while another request is active");
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
    match admit_decisions_command(
        &mut admission_context,
        decisions_command,
        &decisions_result_sender,
    )
    .await?
    {
        RequestAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        } => {
            admit_decisions_request(
                active_request,
                active_generation_permit,
                request_id,
                execution_started_at,
                decisions_result_sender,
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
/// shared single-shot steps, then the decisions capability gate and the
/// worker command dispatch.
async fn admit_decisions_command(
    admission_context: &mut RequestAdmissionContext<'_>,
    decisions_command: DecisionsCommand,
    decisions_result_sender: &DecisionsResultSender,
) -> Result<RequestAdmissionOutcome, WorkerControlError> {
    if let Some(admission_ending_outcome) = make_requested_model_resident(
        admission_context,
        &decisions_command.model,
        decisions_result_sender,
    )
    .await?
    {
        return Ok(admission_ending_outcome);
    }
    if !resident_model_answers_decisions(admission_context.health_snapshot) {
        return Ok(RequestAdmissionOutcome::StartRejected(
            GenerationStartError::WorkerUnavailable,
        ));
    }
    let request_id = decisions_command.request_id;
    let execution_started_at = Instant::now();
    let dispatch_result = admission_context
        .worker_process
        .start_decisions_generation(decisions_command)
        .await;
    Ok(classify_dispatch_result(
        request_id,
        execution_started_at,
        dispatch_result,
    ))
}

/// Whether the resident model advertises a usable decisions surface.
fn resident_model_answers_decisions(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> bool {
    let Ok(snapshot) = health_snapshot.read() else {
        return false;
    };
    let Some(capabilities) = snapshot.ready_model_capabilities.as_ref() else {
        return false;
    };
    match capabilities.decisions {
        Some(decision_capabilities) => decision_capabilities.is_usable(),
        None => false,
    }
}

/// Takes ownership of the active-request slot for this command and clears
/// the stale per-request progress the previous request left behind.
fn admit_decisions_request(
    active_request: &mut Option<ActiveWorkerRequest>,
    active_generation_permit: tokio::sync::OwnedSemaphorePermit,
    request_id: RequestId,
    execution_started_at: Instant,
    decisions_result_sender: DecisionsResultSender,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    *active_request = Some(ActiveWorkerRequest::Decisions(ActiveDecisionsGeneration {
        _active_generation_permit: active_generation_permit,
        request_id,
        terminal_received_at: None,
        execution_deadline: execution_started_at + SINGLE_SHOT_EXECUTION_TIMEOUT,
        decisions_result_sender,
        terminal_outcome: None,
    }));
    clear_active_request_progress(health_snapshot);
    publish_activity(health_snapshot, WorkerActivity::Idle);
}

#[cfg(test)]
#[path = "decisions_request_tests.rs"]
mod decisions_request_tests;
