use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::WorkerEvent;
use tokio::sync::{Semaphore, oneshot};
use tokio::time::{Instant, timeout};

use crate::worker_containment::contain_worker_failure;
use crate::worker_event_handler::handle_worker_event;
use crate::worker_loop_types::ActiveWorkerRequest;
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, GenerationQueueDepth,
    PendingPromptCacheClear, WorkerControlError, WorkerHealthSnapshot, WorkerProcess,
};

/// Result returned to the HTTP request that submitted one cache clear.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptCacheClearOutcome {
    Applied {
        model_id: Option<String>,
        blocks_removed: u64,
        bytes_freed: u64,
    },
    Queued,
}

pub(super) const PROMPT_CACHE_CLEAR_TIMEOUT: Duration = Duration::from_secs(60);

#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_prompt_cache_clear(
    worker_process: &mut WorkerProcess,
    model_id: Option<String>,
    cache_clear_timeout: Duration,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<PromptCacheClearOutcome, WorkerControlError> {
    let requested_model_id = model_id.clone();
    let clear_result = timeout(cache_clear_timeout, async {
        worker_process.clear_prompt_cache(model_id.clone()).await?;
        loop {
            let worker_event = worker_process
                .next_event()
                .await?
                .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
            if let WorkerEvent::PromptCacheCleared {
                model_id: cleared_model_id,
                blocks_removed,
                bytes_freed,
            } = worker_event
            {
                if cleared_model_id != requested_model_id {
                    return Err(WorkerControlError::WorkerProtocolViolation {
                        description: "prompt-cache clear acknowledgement scope mismatch",
                    });
                }
                return Ok(PromptCacheClearOutcome::Applied {
                    model_id: cleared_model_id,
                    blocks_removed,
                    bytes_freed,
                });
            }
            handle_worker_event(
                worker_event,
                health_snapshot,
                is_ready,
                model_load_deadline,
                active_generation,
                performance_log,
                completion_log,
            )?;
        }
    })
    .await;
    match clear_result {
        Ok(Ok(clear_outcome)) => Ok(clear_outcome),
        Ok(Err(clear_error)) => {
            tracing::error!(error = %clear_error, "worker prompt-cache clear failed");
            contain_worker_failure(
                worker_process,
                health_snapshot,
                active_generation,
                clear_error,
            )
            .await;
            *is_ready = false;
            Err(WorkerControlError::MissingActiveWorker)
        }
        Err(_) => {
            let cache_clear_timeout_millis = cache_clear_timeout.as_millis();
            let timeout_error = WorkerControlError::PromptCacheClearTimeout {
                cache_clear_timeout_millis,
            };
            contain_worker_failure(
                worker_process,
                health_snapshot,
                active_generation,
                WorkerControlError::PromptCacheClearTimeout {
                    cache_clear_timeout_millis,
                },
            )
            .await;
            *is_ready = false;
            Err(timeout_error)
        }
    }
}

/// Applies the newest queued clear only after active and queued generations drain.
#[allow(clippy::too_many_arguments)]
pub(super) async fn apply_pending_prompt_cache_clear_if_idle(
    pending_prompt_cache_clear: &mut Option<PendingPromptCacheClear>,
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveWorkerRequest>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    active_generation_permits: &Semaphore,
    generation_queue_permits: &Semaphore,
) {
    if !generation_control_is_idle(
        active_generation,
        active_generation_permits,
        generation_queue_permits,
    ) {
        return;
    }
    let Some(pending_clear) = pending_prompt_cache_clear.take() else {
        return;
    };
    crate::worker_health::publish_pending_prompt_cache_clear(health_snapshot, None);
    let _clear_result = apply_prompt_cache_clear(
        worker_process,
        pending_clear.model_id,
        PROMPT_CACHE_CLEAR_TIMEOUT,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_generation,
        performance_log,
        completion_log,
    )
    .await;
}

/// Queues a busy clear or applies it synchronously while the worker is idle.
#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_prompt_cache_clear_command(
    model_id: Option<String>,
    clear_sender: oneshot::Sender<Result<PromptCacheClearOutcome, WorkerControlError>>,
    pending_prompt_cache_clear: &mut Option<PendingPromptCacheClear>,
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveWorkerRequest>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    active_generation_permits: &Semaphore,
    generation_queue_permits: &Semaphore,
) {
    if !generation_control_is_idle(
        active_generation,
        active_generation_permits,
        generation_queue_permits,
    ) {
        *pending_prompt_cache_clear = Some(PendingPromptCacheClear { model_id });
        crate::worker_health::publish_pending_prompt_cache_clear(
            health_snapshot,
            pending_prompt_cache_clear.clone(),
        );
        let _send_outcome = clear_sender.send(Ok(PromptCacheClearOutcome::Queued));
        return;
    }
    let clear_outcome = apply_prompt_cache_clear(
        worker_process,
        model_id,
        PROMPT_CACHE_CLEAR_TIMEOUT,
        health_snapshot,
        is_ready,
        model_load_deadline,
        active_generation,
        performance_log,
        completion_log,
    )
    .await;
    let _send_outcome = clear_sender.send(clear_outcome);
}

pub(super) fn generation_control_is_idle(
    active_generation: &Option<ActiveWorkerRequest>,
    active_generation_permits: &Semaphore,
    generation_queue_permits: &Semaphore,
) -> bool {
    active_generation.is_none()
        && active_generation_permits.available_permits() == 1
        && generation_queue_permits.available_permits() == GenerationQueueDepth::value()
}
