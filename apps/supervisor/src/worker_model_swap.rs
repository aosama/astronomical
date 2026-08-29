use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    WorkerEvent, WorkerLoadedModelRuntimeConfiguration, WorkerRuntimeFeatureConfiguration,
};
use tokio::time::Instant;

use crate::worker_event_handler::handle_worker_event;
use crate::worker_health::publish_health;
use crate::worker_loop_types::ActiveWorkerRequest;
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, WorkerControlError, WorkerHealthSnapshot,
    WorkerProcess,
};

/// Result of waiting for a worker model-swap acknowledgement.
pub(super) enum ModelSwapWaitOutcome {
    Loaded,
    Rejected { model_load_failure_reason: String },
}

/// Drains worker events until the model swap succeeds or is rejected.
///
/// Process-scoped telemetry and configuration events may have been queued
/// before `SwapModel`. They remain valid while the swap acknowledgement is
/// pending and must update supervisor health rather than terminate the worker.
pub(super) async fn wait_for_model_swap(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
    expected_configuration_generation: Option<&str>,
    expected_model_runtime_configuration: &WorkerLoadedModelRuntimeConfiguration,
) -> Result<ModelSwapWaitOutcome, WorkerControlError> {
    let mut staged_model_swap_event = None;
    let mut staged_runtime_configuration = None;
    loop {
        let worker_event_outcome = worker_process.next_event().await;
        match worker_event_outcome {
            Ok(Some(worker_event)) => match worker_event {
                model_swapped_event @ WorkerEvent::ModelSwapped { .. } => {
                    if staged_model_swap_event.is_some() {
                        return Err(WorkerControlError::WorkerProtocolViolation {
                            description: "duplicate model swap acknowledgement",
                        });
                    }
                    validate_model_swap_event(
                        &model_swapped_event,
                        expected_model_runtime_configuration,
                    )?;
                    staged_model_swap_event = Some(model_swapped_event);
                }
                WorkerEvent::ModelSwapFailed {
                    loaded_model_remains_ready,
                    model_load_failure_reason,
                } => {
                    // A failed first load leaves a healthy model-less worker. A
                    // failed replacement may leave the prior model ready; its
                    // existing health snapshot must remain intact in that case.
                    if !loaded_model_remains_ready {
                        let mlx_memory_ceilings = health_snapshot
                            .read()
                            .map(|worker_health_snapshot| {
                                (
                                    worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
                                    worker_health_snapshot.mlx_memory_ceiling_bytes,
                                    worker_health_snapshot.minimum_mlx_memory_ceiling_bytes,
                                )
                            })
                            .unwrap_or((0, 0, 1));
                        publish_health(
                            health_snapshot,
                            WorkerHealthSnapshot::ready_without_model_with_memory_limits(
                                mlx_memory_ceilings.0,
                                mlx_memory_ceilings.1,
                                mlx_memory_ceilings.2,
                            ),
                        );
                    }
                    tracing::warn!(
                        loaded_model_remains_ready,
                        model_load_failure_reason,
                        "worker rejected the requested model while remaining responsive"
                    );
                    return Ok(ModelSwapWaitOutcome::Rejected {
                        model_load_failure_reason,
                    });
                }
                interleaved_worker_event => {
                    // Do not enumerate supposedly harmless events here. The
                    // central handler accepts process-scoped updates and rejects
                    // generation events whose active-request contract is absent.
                    let interleaved_worker_event_summary =
                        interleaved_worker_event.diagnostic_summary();
                    let is_staged_runtime_acknowledgement =
                        if let WorkerEvent::RuntimeFeatureConfigurationApplied {
                            worker_runtime_feature_configuration,
                        } = &interleaved_worker_event
                            && let Some(expected_generation) = expected_configuration_generation
                        {
                            if worker_runtime_feature_configuration.configuration_generation
                                != expected_generation
                                || worker_runtime_feature_configuration.loaded_model.as_ref()
                                    != Some(expected_model_runtime_configuration)
                            {
                                return Err(WorkerControlError::WorkerProtocolViolation {
                                    description: "model swap runtime policy acknowledgement mismatch",
                                });
                            }
                            if staged_runtime_configuration
                                .replace(worker_runtime_feature_configuration.clone())
                                .is_some()
                            {
                                return Err(WorkerControlError::WorkerProtocolViolation {
                                    description: "duplicate model swap runtime policy acknowledgement",
                                });
                            }
                            true
                        } else {
                            false
                        };
                    if !is_staged_runtime_acknowledgement {
                        handle_worker_event(
                            interleaved_worker_event,
                            health_snapshot,
                            is_ready,
                            model_load_deadline,
                            active_request,
                            performance_log,
                            completion_log,
                        )?;
                        tracing::debug!(
                            worker_event = interleaved_worker_event_summary,
                            "processed interleaved worker event while awaiting model swap"
                        );
                    }
                }
            },
            Ok(None) => return Err(WorkerControlError::WorkerEventStreamClosed),
            Err(worker_event_error) => return Err(worker_event_error),
        }
        let policy_is_ready =
            expected_configuration_generation.is_none() || staged_runtime_configuration.is_some();
        if staged_model_swap_event.is_some() && policy_is_ready {
            publish_staged_model_swap(
                health_snapshot,
                staged_model_swap_event.take().ok_or(
                    WorkerControlError::WorkerProtocolViolation {
                        description: "model swap acknowledgement staging failed",
                    },
                )?,
                staged_runtime_configuration,
            )?;
            return Ok(ModelSwapWaitOutcome::Loaded);
        }
    }
}

fn validate_model_swap_event(
    model_swap_event: &WorkerEvent,
    expected_runtime_configuration: &WorkerLoadedModelRuntimeConfiguration,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::ModelSwapped {
        model_id,
        capabilities,
        ..
    } = model_swap_event
    else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "non-model event staged as model swap acknowledgement",
        });
    };
    if model_id != expected_runtime_configuration.model_id() {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "model swap identity acknowledgement mismatch",
        });
    }
    let capabilities_match_policy = match expected_runtime_configuration {
        WorkerLoadedModelRuntimeConfiguration::Autoregressive(configuration) => {
            capabilities.chat.as_ref().is_some_and(|chat_capabilities| {
                chat_capabilities.context_window == configuration.maximum_context_tokens
                    && chat_capabilities.max_output_tokens == configuration.maximum_output_tokens
                    && chat_capabilities.max_input_tokens
                        == configuration.maximum_context_tokens.saturating_sub(1)
            }) && capabilities.image_generation.is_none()
        }
        WorkerLoadedModelRuntimeConfiguration::Flux2Klein(_) => {
            capabilities.chat.is_none() && capabilities.image_generation.is_some()
        }
    };
    if !capabilities_match_policy {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "model swap capabilities acknowledgement mismatch",
        });
    }
    Ok(())
}

fn publish_staged_model_swap(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    model_swap_event: WorkerEvent,
    runtime_configuration: Option<WorkerRuntimeFeatureConfiguration>,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::ModelSwapped {
        model_id,
        capabilities,
        expert_memory_mode,
        minimum_mlx_memory_ceiling_bytes,
        mtp_runtime_state,
        mtp_unavailable_reason,
        mtp_depth_status,
        speculative_prefill_runtime_state,
        speculative_prefill_unavailable_reason,
        speculative_prefill_draft_model_id,
        speculative_prefill_draft_model_revision,
    } = model_swap_event
    else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "non-model event staged as model swap acknowledgement",
        });
    };
    let Ok(mut current_health_snapshot) = health_snapshot.write() else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "worker health lock is unavailable",
        });
    };
    let mut replacement_health_snapshot = WorkerHealthSnapshot::ready_with_replacement_model(
        model_id,
        capabilities,
        minimum_mlx_memory_ceiling_bytes,
        mtp_runtime_state,
        mtp_unavailable_reason,
        &current_health_snapshot,
    )
    .with_mtp_depth_status(mtp_depth_status)
    .with_speculative_prefill_runtime(
        speculative_prefill_runtime_state,
        speculative_prefill_unavailable_reason,
        speculative_prefill_draft_model_id,
        speculative_prefill_draft_model_revision,
    );
    replacement_health_snapshot.expert_memory_mode = expert_memory_mode;
    if let Some(runtime_configuration) = runtime_configuration {
        replacement_health_snapshot.worker_runtime_feature_configuration =
            Some(runtime_configuration);
    }
    *current_health_snapshot = replacement_health_snapshot;
    Ok(())
}
