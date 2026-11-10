//! Admission steps shared by every request that runs as one forward pass.
//!
//! Embeddings and decisions requests have the same life before execution:
//! wait for the worker's startup configuration, make the requested model
//! resident (swapping the loaded one out when it differs), and notice a
//! client that went away meanwhile. Those steps live here once. What differs
//! per kind of request, its capability check, its worker command and its
//! result type, stays with that kind.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{ProtocolError, RequestId};
use tokio::sync::mpsc;
use tokio::time::{Instant, timeout};

use crate::worker::loop_types::{ActiveWorkerRequest, WorkerLoopState};
use crate::worker::process::model_swap::{ModelSwapWaitOutcome, wait_for_model_swap};
use crate::worker::process::startup_runtime::wait_for_startup_runtime_configuration;
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, GenerationStartError, RuntimeModelPolicy,
    WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
};

/// How long one forward-pass request may run. It reports no progress, so
/// one deadline is what protects the loop from a stuck engine.
pub(super) const SINGLE_SHOT_EXECUTION_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(60);

/// What the admission pipeline decided for one command.
///
/// Admission units return these outcomes instead of answering the client
/// themselves: the owning handler performs every sender send and containment
/// call in order (`tokio::sync::oneshot::Sender` is not cloneable, so the
/// answer must stay with the owner).
pub(super) enum SingleShotAdmissionOutcome {
    /// The worker accepted the command; admit the active request and confirm.
    Started {
        request_id: RequestId,
        execution_started_at: Instant,
    },
    /// The client result channel closed during admission; nothing to answer.
    ClientDisconnected,
    /// Admission refused before any worker execution; forward this error.
    StartRejected(GenerationStartError),
    /// The worker failed after the refusal notice; containment must follow.
    WorkerFailed(WorkerControlError),
}

/// The worker-loop state an admission needs, borrowed for one command.
pub(super) struct SingleShotAdmissionContext<'loop_state> {
    pub(super) worker_process: &'loop_state mut WorkerProcess,
    pub(super) health_snapshot: &'loop_state Arc<RwLock<WorkerHealthSnapshot>>,
    pub(super) is_ready: &'loop_state mut bool,
    pub(super) model_load_deadline: &'loop_state mut Option<Instant>,
    pub(super) active_request: &'loop_state mut Option<ActiveWorkerRequest>,
    pub(super) performance_log: &'loop_state mut GenerationPerformanceLog,
    pub(super) completion_log: &'loop_state mut CompletionAttributionLog,
    pub(super) model_policy_catalog: &'loop_state Arc<HashMap<String, RuntimeModelPolicy>>,
    pub(super) model_load_timeout: std::time::Duration,
}

/// Makes `requested_model` the resident model, swapping when it is not.
///
/// Returns `None` when the model is resident and the client is still
/// connected, so the caller can go on to dispatch; otherwise the outcome
/// that ends admission.
pub(super) async fn make_requested_model_resident<ResultMessage>(
    admission_context: &mut SingleShotAdmissionContext<'_>,
    requested_model: &str,
    result_sender: &mpsc::Sender<ResultMessage>,
) -> Result<Option<SingleShotAdmissionOutcome>, WorkerControlError> {
    wait_for_startup_runtime_configuration(
        admission_context.worker_process,
        admission_context.health_snapshot,
        admission_context.is_ready,
        admission_context.model_load_deadline,
        admission_context.active_request,
        admission_context.performance_log,
        admission_context.completion_log,
        admission_context.model_load_timeout,
    )
    .await?;
    let loaded_model_id = admission_context
        .health_snapshot
        .read()
        .ok()
        .and_then(|snapshot| snapshot.ready_model_id.clone());
    if loaded_model_id.as_deref() != Some(requested_model)
        && let Some(swap_ending_outcome) =
            swap_to_requested_model(admission_context, requested_model, result_sender).await
    {
        return Ok(Some(swap_ending_outcome));
    }
    if result_sender.is_closed() {
        return Ok(Some(SingleShotAdmissionOutcome::ClientDisconnected));
    }
    Ok(None)
}

/// Drives the model swap; `None` means the requested model is now resident.
async fn swap_to_requested_model<ResultMessage>(
    admission_context: &mut SingleShotAdmissionContext<'_>,
    requested_model: &str,
    result_sender: &mpsc::Sender<ResultMessage>,
) -> Option<SingleShotAdmissionOutcome> {
    let Some(model_policy) = admission_context.model_policy_catalog.get(requested_model) else {
        return Some(SingleShotAdmissionOutcome::StartRejected(
            GenerationStartError::WorkerUnavailable,
        ));
    };
    let expected_configuration_generation =
        admission_context
            .health_snapshot
            .read()
            .ok()
            .and_then(|snapshot| {
                snapshot
                    .worker_runtime_feature_configuration
                    .as_ref()
                    .map(|configuration| configuration.configuration_generation.clone())
            });
    let expected_model_runtime_configuration = model_policy
        .worker_model_configuration
        .runtime_configuration();
    if let Err(swap_error) = admission_context
        .worker_process
        .swap_model(
            model_policy.model_directory.to_string_lossy().into_owned(),
            model_policy.worker_model_configuration.clone(),
        )
        .await
    {
        return Some(SingleShotAdmissionOutcome::WorkerFailed(swap_error));
    }
    let model_load_timeout = admission_context.model_load_timeout;
    let mut swap_loop_state = WorkerLoopState {
        worker_process: &mut *admission_context.worker_process,
        health_snapshot: admission_context.health_snapshot,
        is_ready: &mut *admission_context.is_ready,
        model_load_deadline: &mut *admission_context.model_load_deadline,
        active_request: &mut *admission_context.active_request,
        performance_log: &mut *admission_context.performance_log,
        completion_log: &mut *admission_context.completion_log,
        model_load_timeout,
    };
    let mut swap_wait = Box::pin(timeout(
        model_load_timeout,
        wait_for_model_swap(
            &mut swap_loop_state,
            expected_configuration_generation.as_deref(),
            &expected_model_runtime_configuration,
        ),
    ));
    // A client that leaves mid-swap does not abandon the swap: the worker
    // is already loading, so the load is seen through and only then dropped.
    let (swap_outcome, client_disconnected_during_swap) = tokio::select! {
        swap_outcome = &mut swap_wait => (swap_outcome, false),
        () = result_sender.closed() => ((&mut swap_wait).await, true),
    };
    drop(swap_wait);
    let swap_outcome = match swap_outcome {
        Ok(swap_outcome) => swap_outcome,
        Err(_elapsed) => Err(WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: model_load_timeout.as_millis(),
        }),
    };
    match swap_outcome {
        Ok(ModelSwapWaitOutcome::Loaded) if client_disconnected_during_swap => {
            Some(SingleShotAdmissionOutcome::ClientDisconnected)
        }
        Ok(ModelSwapWaitOutcome::Loaded) => None,
        Ok(ModelSwapWaitOutcome::Rejected {
            model_load_failure_reason,
        }) => Some(SingleShotAdmissionOutcome::StartRejected(
            GenerationStartError::ModelLoadFailed {
                model_load_failure_reason,
            },
        )),
        Err(swap_error) => Some(SingleShotAdmissionOutcome::WorkerFailed(swap_error)),
    }
}

/// Classifies the result of handing a command to the worker process.
///
/// A command too large for one IPC frame is the request's fault and leaves
/// the worker healthy; any other failure means the worker must be contained.
pub(super) fn classify_dispatch_result(
    request_id: RequestId,
    execution_started_at: Instant,
    dispatch_result: Result<(), WorkerControlError>,
) -> SingleShotAdmissionOutcome {
    match dispatch_result {
        Ok(()) => SingleShotAdmissionOutcome::Started {
            request_id,
            execution_started_at,
        },
        Err(WorkerControlError::Protocol(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes,
            maximum_message_bytes,
        })) => SingleShotAdmissionOutcome::StartRejected(GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes: actual_message_bytes,
            maximum_ipc_message_bytes: maximum_message_bytes,
        }),
        Err(start_error) => SingleShotAdmissionOutcome::WorkerFailed(start_error),
    }
}
