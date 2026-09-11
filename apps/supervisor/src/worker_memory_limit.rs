use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use astronomical_ipc_protocol::WorkerEvent;
use tokio::time::{Instant, timeout};

use crate::worker_containment::contain_worker_failure;
use crate::worker_event_handler::handle_worker_event;
use crate::worker_loop_types::ActiveWorkerRequest;
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, GenerationStartError, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
};
use tokio::sync::oneshot;

/// Completion state returned by a live MLX memory-limit request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxMemoryLimitUpdateOutcome {
    Applied,
    Queued,
    Rejected,
}

/// A live ceiling raise that waited because a generation was already running.
pub(super) struct PendingMlxMemoryLimitUpdate {
    pub effective_mlx_memory_ceiling_bytes: u64,
    pub configuration_generation: String,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_mlx_memory_limit(
    worker_process: &mut WorkerProcess,
    effective_mlx_memory_ceiling_bytes: u64,
    configuration_generation: String,
    memory_limit_update_timeout: Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
    let update_outcome = timeout(memory_limit_update_timeout, async {
        worker_process
            .update_mlx_memory_limit(effective_mlx_memory_ceiling_bytes, configuration_generation)
            .await?;
        loop {
            let worker_event = worker_process
                .next_event()
                .await?
                .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
            let memory_limit_update_outcome = match &worker_event {
                WorkerEvent::MlxMemoryLimitChanged { .. } => {
                    Some(MlxMemoryLimitUpdateOutcome::Applied)
                }
                WorkerEvent::MlxMemoryLimitRejected { .. } => {
                    Some(MlxMemoryLimitUpdateOutcome::Rejected)
                }
                _ => None,
            };
            handle_worker_event(
                worker_event,
                health_snapshot,
                is_ready,
                model_load_deadline,
                active_request,
                performance_log,
                completion_log,
            )?;
            if let Some(memory_limit_update_outcome) = memory_limit_update_outcome {
                return Ok(memory_limit_update_outcome);
            }
        }
    })
    .await;
    match update_outcome {
        Ok(update_outcome) => update_outcome,
        Err(_) => {
            let memory_limit_update_timeout_millis = memory_limit_update_timeout.as_millis();
            contain_worker_failure(
                worker_process,
                health_snapshot,
                active_request,
                WorkerControlError::MlxMemoryLimitUpdateTimeout {
                    memory_limit_update_timeout_millis,
                },
            )
            .await;
            *is_ready = false;
            Err(WorkerControlError::MlxMemoryLimitUpdateTimeout {
                memory_limit_update_timeout_millis,
            })
        }
    }
}

pub(super) async fn contain_mlx_memory_limit_failure(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    is_ready: &mut bool,
    memory_limit_error: WorkerControlError,
) {
    if !matches!(
        &memory_limit_error,
        WorkerControlError::MlxMemoryLimitUpdateTimeout { .. }
    ) {
        contain_worker_failure(
            worker_process,
            health_snapshot,
            active_request,
            memory_limit_error,
        )
        .await;
    }
    *is_ready = false;
}

/// Applies a queued live ceiling raise now that the worker is idle.
///
/// Issue #515: the next Generate must not be admitted against the old
/// ceiling after the user has already asked for more RAM. Call this both
/// at the idle loop head and immediately before starting chat, image, or
/// embeddings work, because `select` can deliver a Generate that was
/// sitting in the channel beside the raise.
#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_pending_mlx_memory_limit_if_idle(
    pending_mlx_memory_limit_update: &mut Option<PendingMlxMemoryLimitUpdate>,
    worker_process: &mut WorkerProcess,
    model_load_timeout: Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<Option<MlxMemoryLimitUpdateOutcome>, WorkerControlError> {
    let Some(pending_memory_limit_update) = pending_mlx_memory_limit_update.take() else {
        return Ok(None);
    };
    let update_outcome = apply_mlx_memory_limit(
        worker_process,
        pending_memory_limit_update.effective_mlx_memory_ceiling_bytes,
        pending_memory_limit_update.configuration_generation,
        model_load_timeout,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_generation,
        performance_log,
        completion_log,
    )
    .await?;
    Ok(Some(update_outcome))
}

/// Applies a queued ceiling raise, then reports whether generation may start.
///
/// Returns `Some(start_sender)` when the request may proceed. Returns `None`
/// after refusing the start (rejected raise or control failure). The caller
/// must `continue` the worker loop when this returns `None`.
#[allow(clippy::too_many_arguments)]
pub(super) async fn take_generation_start_after_pending_memory_limit(
    pending_mlx_memory_limit_update: &mut Option<PendingMlxMemoryLimitUpdate>,
    worker_process: &mut WorkerProcess,
    model_load_timeout: Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    start_sender: oneshot::Sender<Result<(), GenerationStartError>>,
) -> Option<oneshot::Sender<Result<(), GenerationStartError>>> {
    match apply_pending_mlx_memory_limit_if_idle(
        pending_mlx_memory_limit_update,
        worker_process,
        model_load_timeout,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_generation,
        performance_log,
        completion_log,
    )
    .await
    {
        Ok(None)
        | Ok(Some(MlxMemoryLimitUpdateOutcome::Applied))
        | Ok(Some(MlxMemoryLimitUpdateOutcome::Queued)) => Some(start_sender),
        Ok(Some(MlxMemoryLimitUpdateOutcome::Rejected)) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            None
        }
        Err(memory_limit_error) => {
            let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
            contain_mlx_memory_limit_failure(
                worker_process,
                health_snapshot,
                active_generation,
                is_ready,
                memory_limit_error,
            )
            .await;
            None
        }
    }
}
