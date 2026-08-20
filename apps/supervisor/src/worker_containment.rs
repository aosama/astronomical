use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_ipc_protocol::{RequestId, WorkerEvent, WorkerRuntimeFeatureConfiguration};
use tokio::time::timeout;

use crate::chat_generation_executor::try_send_stream_event;
use crate::worker_health::{
    clear_active_request_progress, clear_latest_mlx_memory_snapshot, publish_activity,
    publish_expert_memory_mode, publish_health, publish_latest_mlx_memory_snapshot,
    publish_persistent_prompt_cache_stats,
};
use crate::worker_loop_types::ActiveWorkerRequest;
use crate::{
    ChatGenerationStreamErrorCode, ChatGenerationStreamEvent, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerHealthStatus, WorkerProcess,
};

pub(super) async fn cancel_active_generation(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    cancellation_acknowledgement_timeout: Duration,
    recovery_acknowledgement_timeout: Duration,
    is_ready: &mut bool,
) {
    let Some(cancelled_request) = active_request.take() else {
        return;
    };
    publish_activity(health_snapshot, WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    let cancellation_outcome = cancel_worker_request(
        worker_process,
        health_snapshot,
        cancelled_request.request_id(),
        cancellation_acknowledgement_timeout,
        matches!(cancelled_request, ActiveWorkerRequest::Image(_)),
    )
    .await;
    if let Err(cancellation_error) = cancellation_outcome {
        tracing::error!(error = %cancellation_error, "worker cancellation failed; replacing worker");
        *is_ready = false;
        publish_health(
            health_snapshot,
            WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
        );
        if let Err(cleanup_error) = worker_process.force_terminate().await {
            tracing::error!(error = %cleanup_error, "failed to terminate worker after cancellation failure");
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
            );
            return;
        }
        if let Err(recovery_error) = recover_worker_after_cancellation_failure(
            worker_process,
            health_snapshot,
            recovery_acknowledgement_timeout,
        )
        .await
        {
            tracing::error!(error = %recovery_error, "failed to replace worker after cancellation containment");
            let _cleanup_outcome = worker_process.force_terminate().await;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
            );
            return;
        }
        *is_ready = true;
    }
}

async fn recover_worker_after_cancellation_failure(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    recovery_acknowledgement_timeout: Duration,
) -> Result<(), WorkerControlError> {
    worker_process.relaunch_after_termination().await?;
    let expected_configuration_generation = worker_process
        .expected_configuration_generation()
        .map(str::to_owned);
    let (readiness_event, runtime_configuration) = timeout(
        recovery_acknowledgement_timeout,
        read_recovery_acknowledgement(worker_process, expected_configuration_generation.as_deref()),
    )
    .await
    .map_err(|_| WorkerControlError::CandidateAcknowledgementTimeout {
        acknowledgement_timeout_millis: recovery_acknowledgement_timeout.as_millis(),
    })??;
    publish_recovery_acknowledgement(health_snapshot, readiness_event, runtime_configuration)
}

async fn read_recovery_acknowledgement(
    worker_process: &mut WorkerProcess,
    expected_configuration_generation: Option<&str>,
) -> Result<(WorkerEvent, Option<WorkerRuntimeFeatureConfiguration>), WorkerControlError> {
    let mut readiness_event = None;
    let mut runtime_configuration = None;
    loop {
        let worker_event = worker_process
            .next_event()
            .await?
            .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
        match worker_event {
            event @ (WorkerEvent::Idle { .. } | WorkerEvent::Ready { .. }) => {
                if readiness_event.replace(event).is_some() {
                    return Err(WorkerControlError::WorkerProtocolViolation {
                        description: "replacement worker emitted duplicate readiness",
                    });
                }
            }
            WorkerEvent::RuntimeFeatureConfigurationApplied {
                worker_runtime_feature_configuration,
            } => {
                if expected_configuration_generation
                    != Some(
                        worker_runtime_feature_configuration
                            .configuration_generation
                            .as_str(),
                    )
                {
                    return Err(WorkerControlError::WorkerProtocolViolation {
                        description: "replacement worker runtime configuration mismatch",
                    });
                }
                if runtime_configuration
                    .replace(worker_runtime_feature_configuration)
                    .is_some()
                {
                    return Err(WorkerControlError::WorkerProtocolViolation {
                        description: "replacement worker emitted duplicate runtime configuration",
                    });
                }
            }
            WorkerEvent::MlxMemorySample { .. }
            | WorkerEvent::ExpertMemoryModeChanged { .. }
            | WorkerEvent::PersistentPromptCacheStats { .. } => {}
            _ => {
                return Err(WorkerControlError::WorkerProtocolViolation {
                    description: "replacement worker emitted an unexpected startup event",
                });
            }
        }
        if let Some(readiness_event) = readiness_event.clone()
            && (expected_configuration_generation.is_none() || runtime_configuration.is_some())
        {
            return Ok((readiness_event, runtime_configuration));
        }
    }
}

fn publish_recovery_acknowledgement(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    readiness_event: WorkerEvent,
    runtime_configuration: Option<WorkerRuntimeFeatureConfiguration>,
) -> Result<(), WorkerControlError> {
    match (&readiness_event, runtime_configuration.as_ref()) {
        (WorkerEvent::Idle { .. }, Some(configuration)) if configuration.loaded_model.is_some() => {
            return Err(WorkerControlError::WorkerProtocolViolation {
                description: "idle replacement worker acknowledged a loaded model",
            });
        }
        (WorkerEvent::Ready { model_id, .. }, Some(configuration))
            if configuration
                .loaded_model
                .as_ref()
                .map(|loaded_model| loaded_model.model_id())
                != Some(model_id.as_str()) =>
        {
            return Err(WorkerControlError::WorkerProtocolViolation {
                description: "ready replacement worker policy did not match its model",
            });
        }
        _ => {}
    }
    let mut recovered_snapshot = match readiness_event {
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
        } => WorkerHealthSnapshot::ready_without_model_with_memory_limits(
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
        ),
        WorkerEvent::Ready {
            model_id,
            capabilities,
            mtp_runtime_state,
            mtp_unavailable_reason,
            mtp_depth_status,
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        } => WorkerHealthSnapshot::ready_with_model(
            model_id,
            capabilities,
            mtp_runtime_state,
            mtp_unavailable_reason,
        )
        .with_mtp_depth_status(mtp_depth_status)
        .with_speculative_prefill_runtime(
            speculative_prefill_runtime_state,
            speculative_prefill_unavailable_reason,
            speculative_prefill_draft_model_id,
            speculative_prefill_draft_model_revision,
        ),
        _ => {
            return Err(WorkerControlError::WorkerProtocolViolation {
                description: "replacement worker acknowledgement lost readiness",
            });
        }
    };
    recovered_snapshot.worker_runtime_feature_configuration = runtime_configuration;
    publish_health(health_snapshot, recovered_snapshot);
    Ok(())
}

async fn cancel_worker_request(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    request_id: RequestId,
    cancellation_acknowledgement_timeout: Duration,
    expects_image_finalization: bool,
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
                WorkerEvent::GenerationPreparationStarted {
                    request_id: preparation_request_id,
                    ..
                } if preparation_request_id == request_id => {}
                WorkerEvent::GenerationProgress {
                    request_id: progress_request_id,
                    ..
                } if progress_request_id == request_id => {}
                WorkerEvent::FirstDecodeCompleted {
                    request_id: decode_request_id,
                    ..
                } if decode_request_id == request_id => {}
                WorkerEvent::PromptWorkReuse {
                    request_id: reuse_request_id,
                    ..
                } if reuse_request_id == request_id => {}
                WorkerEvent::ImageGenerationProgress {
                    request_id: image_request_id,
                    ..
                } if expects_image_finalization && image_request_id == request_id => {}
                WorkerEvent::ImageGenerationCompleted {
                    request_id: image_request_id,
                    ..
                } if expects_image_finalization && image_request_id == request_id => {}
                WorkerEvent::ImageGenerationFailed {
                    request_id: image_request_id,
                    ..
                } if expects_image_finalization && image_request_id == request_id => {}
                WorkerEvent::ImageGenerationFinalized {
                    request_id: image_request_id,
                    mlx_memory_snapshot,
                    ..
                } if expects_image_finalization && image_request_id == request_id => {
                    crate::worker_image_event::publish_image_finalized_memory_snapshot(
                        health_snapshot,
                        mlx_memory_snapshot,
                    )?;
                    return Ok(());
                }
                WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode } => {
                    publish_expert_memory_mode(health_snapshot, expert_memory_mode);
                }
                WorkerEvent::GenerationFinalized {
                    expert_memory_mode,
                    mlx_memory_snapshot,
                    expert_residency,
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
                    if let Some(expert_residency) = expert_residency {
                        crate::worker_health::publish_worker_expert_residency(
                            health_snapshot,
                            expert_residency,
                        );
                    }
                }
                WorkerEvent::PersistentPromptCacheStats { .. } => {
                    // Cache publication can finish after the client drops the
                    // stream. Keep the worker reusable and record the telemetry.
                    publish_persistent_prompt_cache_stats(health_snapshot, worker_event);
                }
                WorkerEvent::MlxMemorySample {
                    mlx_memory_snapshot,
                } => {
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
    active_request: &mut Option<ActiveWorkerRequest>,
    operation_error: WorkerControlError,
) {
    let worker_process_id = worker_process.process_id();
    tracing::error!(
        error = %operation_error,
        worker_process_id = ?worker_process_id,
        "worker failed; terminating local worker process"
    );
    fail_active_generation(
        active_request,
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
    active_request: &mut Option<ActiveWorkerRequest>,
    error_code: ChatGenerationStreamErrorCode,
) {
    match active_request.take() {
        Some(ActiveWorkerRequest::Chat(failed_generation)) => {
            let _send_outcome = try_send_stream_event(
                &failed_generation.stream_event_sender,
                ChatGenerationStreamEvent::Error(error_code),
            );
        }
        Some(ActiveWorkerRequest::Image(failed_image)) => {
            let _send_outcome = failed_image
                .image_result_sender
                .try_send(Err(crate::ImageGenerationExecutionError::WorkerUnavailable));
        }
        None => {}
    }
}

pub(super) async fn close_worker_if_running(worker_process: &mut WorkerProcess) {
    if worker_process.process_id().is_some()
        && let Err(shutdown_error) = worker_process.close().await
    {
        tracing::error!(error = %shutdown_error, "failed to close worker");
    }
}
