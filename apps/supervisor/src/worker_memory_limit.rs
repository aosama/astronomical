use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use astronomical_ipc_protocol::WorkerEvent;
use tokio::time::{Instant, timeout};

use crate::worker_containment::contain_worker_failure;
use crate::worker_event_handler::handle_worker_event;
use crate::worker_loop_types::ActiveWorkerRequest;
use crate::{GenerationPerformanceLog, WorkerControlError, WorkerHealthSnapshot, WorkerProcess};

/// Completion state returned by a live MLX memory-limit request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxMemoryLimitUpdateOutcome {
    Applied,
    Queued,
    Rejected,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_mlx_memory_limit(
    worker_process: &mut WorkerProcess,
    effective_mlx_memory_ceiling_bytes: u64,
    memory_limit_update_timeout: Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
    let update_outcome = timeout(memory_limit_update_timeout, async {
        worker_process
            .update_mlx_memory_limit(effective_mlx_memory_ceiling_bytes)
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
