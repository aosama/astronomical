use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::{RequestId, WorkerEvent};
use tokio::time::timeout;

use crate::chat_generation_executor::try_send_stream_event;
use crate::worker_health::{
    clear_active_request_progress, clear_latest_mlx_memory_snapshot, publish_activity,
    publish_expert_memory_mode, publish_health, publish_latest_mlx_memory_snapshot,
};
use crate::worker_loop_types::ActiveGeneration;
use crate::{
    ChatGenerationStreamErrorCode, ChatGenerationStreamEvent, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerHealthStatus, WorkerProcess,
};

pub(super) async fn cancel_active_generation(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveGeneration>,
    cancellation_acknowledgement_timeout: Duration,
) {
    let Some(cancelled_generation) = active_generation.take() else {
        return;
    };
    publish_activity(health_snapshot, WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    let cancellation_outcome = cancel_worker_request(
        worker_process,
        health_snapshot,
        cancelled_generation.request_id,
        cancellation_acknowledgement_timeout,
    )
    .await;
    if let Err(cancellation_error) = cancellation_outcome {
        tracing::error!(error = %cancellation_error, "worker cancellation failed; terminating worker");
        if let Err(cleanup_error) = worker_process.force_terminate().await {
            tracing::error!(error = %cleanup_error, "failed to terminate worker after cancellation failure");
        }
        publish_health(
            health_snapshot,
            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
        );
    }
}

async fn cancel_worker_request(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    request_id: RequestId,
    cancellation_acknowledgement_timeout: Duration,
) -> Result<(), WorkerControlError> {
    worker_process.cancel_generation(request_id).await?;
    timeout(cancellation_acknowledgement_timeout, async {
        loop {
            let worker_event = worker_process
                .next_event()
                .await?
                .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
            match worker_event {
                WorkerEvent::Output {
                    request_id: output_request_id,
                    ..
                } if output_request_id == request_id => {}
                WorkerEvent::PrefillProgress {
                    request_id: prefill_request_id,
                    ..
                } if prefill_request_id == request_id => {}
                WorkerEvent::GenerationProgress {
                    request_id: progress_request_id,
                    ..
                } if progress_request_id == request_id => {}
                WorkerEvent::PromptWorkReuse {
                    request_id: reuse_request_id,
                    ..
                } if reuse_request_id == request_id => {}
                WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode } => {
                    publish_expert_memory_mode(health_snapshot, expert_memory_mode);
                }
                WorkerEvent::GenerationFinalized {
                    expert_memory_mode,
                    mlx_memory_snapshot,
                    ..
                } => {
                    if let Some(expert_memory_mode) = expert_memory_mode {
                        publish_expert_memory_mode(health_snapshot, expert_memory_mode);
                    }
                    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
                    } else {
                        clear_latest_mlx_memory_snapshot(health_snapshot);
                    }
                }
                WorkerEvent::Completed {
                    request_id: completed_request_id,
                    ..
                } if completed_request_id == request_id => return Ok(()),
                WorkerEvent::Failed {
                    request_id: failed_request_id,
                    ..
                } if failed_request_id == request_id => return Ok(()),
                unexpected_worker_event => {
                    return Err(WorkerControlError::UnexpectedCancellationEvent {
                        request_id: request_id.value(),
                        unexpected_worker_event_summary: unexpected_worker_event
                            .diagnostic_summary(),
                    });
                }
            }
        }
    })
    .await
    .map_err(|_| WorkerControlError::CancellationAckTimeout {
        cancellation_timeout_millis: cancellation_acknowledgement_timeout.as_millis(),
    })?
}

pub(super) async fn contain_worker_failure(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveGeneration>,
    operation_error: WorkerControlError,
) {
    let worker_process_id = worker_process.process_id();
    tracing::error!(
        error = %operation_error,
        worker_process_id = ?worker_process_id,
        "worker failed; terminating local worker process"
    );
    fail_active_generation(
        active_generation,
        ChatGenerationStreamErrorCode::WorkerUnavailable,
    );
    publish_activity(health_snapshot, WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    match worker_process.force_terminate().await {
        Ok(worker_termination_outcome) => {
            tracing::error!(
                worker_process_id = ?worker_process_id,
                worker_termination_outcome = ?worker_termination_outcome,
                "local worker process terminated after failure"
            );
        }
        Err(cleanup_error) => {
            tracing::error!(
                error = %cleanup_error,
                worker_process_id = ?worker_process_id,
                "failed to terminate local worker process"
            );
        }
    }
    publish_health(
        health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
    );
}

pub(super) fn fail_active_generation(
    active_generation: &mut Option<ActiveGeneration>,
    error_code: ChatGenerationStreamErrorCode,
) {
    if let Some(failed_generation) = active_generation.take() {
        let _send_outcome = try_send_stream_event(
            &failed_generation.stream_event_sender,
            ChatGenerationStreamEvent::Error(error_code),
        );
    }
}

pub(super) async fn close_worker_if_running(worker_process: &mut WorkerProcess) {
    if worker_process.process_id().is_some()
        && let Err(shutdown_error) = worker_process.close().await
    {
        tracing::error!(error = %shutdown_error, "failed to close worker");
    }
}
