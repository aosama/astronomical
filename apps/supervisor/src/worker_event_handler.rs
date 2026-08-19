//! Dispatches typed worker events to supervisor-owned request and health state.
//!
//! Process-scoped events (memory samples, readiness) are valid whenever their
//! payload is valid. Generation-scoped events additionally require a matching
//! active request. Model-swap waiting delegates here so both loops enforce the
//! same protocol rules instead of maintaining competing event allowlists.

use astronomical_ipc_protocol::WorkerEvent;
use std::sync::{Arc, RwLock};
use tokio::time::Instant;

use crate::{
    ChatGenerationStreamEvent, ExpertResidencySnapshot, GenerationPerformanceLog, WorkerActivity,
    WorkerControlError, WorkerHealthSnapshot,
    chat_generation_executor::try_send_stream_event,
    worker_completion_event::handle_worker_completion_event,
    worker_generation_output::{
        handle_worker_first_decode_completed, handle_worker_generation_progress,
        handle_worker_output, handle_worker_prompt_work_reuse,
    },
    worker_generation_preparation::handle_generation_preparation_started,
    worker_health::{
        clear_active_request_progress, clear_latest_mlx_memory_snapshot, publish_activity,
        publish_expert_memory_mode, publish_health, publish_latest_mlx_memory_snapshot,
        publish_mlx_memory_limit_changed, publish_mlx_memory_limit_rejection,
        publish_persistent_prompt_cache_stats,
    },
    worker_loop_types::ActiveGeneration,
    worker_prefill_progress::handle_worker_prefill_progress,
};

/// Applies one typed worker event to supervisor-owned request and health state.
///
/// Process-scoped events such as memory samples are valid whenever their own
/// payload is valid. Generation-scoped events additionally require a matching
/// active request. Model-swap waiting delegates here so both loops enforce the
/// same protocol rules instead of maintaining competing event allowlists.
pub(super) fn handle_worker_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_generation: &mut Option<ActiveGeneration>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration,
        } => {
            let Ok(mut worker_health_snapshot) = health_snapshot.write() else {
                return Err(protocol_violation("worker health lock is unavailable"));
            };
            worker_health_snapshot.worker_runtime_feature_configuration =
                Some(worker_runtime_feature_configuration);
        }
        WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode } => {
            publish_expert_memory_mode(health_snapshot, expert_memory_mode);
        }
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
        } => {
            if *is_ready || active_generation.is_some() {
                return Err(protocol_violation("duplicate worker readiness"));
            }
            *is_ready = true;
            tracing::info!(model = %model_id, "worker model is ready");
            *model_load_deadline = None;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::ready_with_model(
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
            );
        }
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
        } => {
            if *is_ready || active_generation.is_some() {
                return Err(protocol_violation("duplicate worker idle event"));
            }
            *is_ready = true;
            *model_load_deadline = None;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::ready_without_model_with_memory_limits(
                    machine_mlx_memory_ceiling_bytes,
                    effective_mlx_memory_ceiling_bytes,
                    minimum_mlx_memory_ceiling_bytes,
                ),
            );
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
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode,
            mlx_memory_snapshot,
        } => {
            publish_mlx_memory_limit_changed(
                health_snapshot,
                effective_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                expert_memory_mode,
                mlx_memory_snapshot,
            );
        }
        WorkerEvent::MlxMemoryLimitRejected {
            minimum_mlx_memory_ceiling_bytes,
            reason,
            ..
        } => {
            publish_mlx_memory_limit_rejection(
                health_snapshot,
                minimum_mlx_memory_ceiling_bytes,
                reason,
            );
        }
        WorkerEvent::ModelSwapped {
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
        } => {
            if !*is_ready {
                return Err(protocol_violation("model swapped before initial readiness"));
            }
            tracing::info!(model = %model_id, "worker model swapped");
            let mut ready_worker_health_snapshot =
                if let Ok(previous_health_snapshot) = health_snapshot.read() {
                    WorkerHealthSnapshot::ready_with_replacement_model(
                        model_id,
                        capabilities,
                        minimum_mlx_memory_ceiling_bytes,
                        mtp_runtime_state,
                        mtp_unavailable_reason,
                        &previous_health_snapshot,
                    )
                    .with_mtp_depth_status(mtp_depth_status)
                    .with_speculative_prefill_runtime(
                        speculative_prefill_runtime_state,
                        speculative_prefill_unavailable_reason,
                        speculative_prefill_draft_model_id,
                        speculative_prefill_draft_model_revision,
                    )
                } else {
                    let mut replacement_health_snapshot = WorkerHealthSnapshot::ready_with_model(
                        model_id,
                        capabilities,
                        mtp_runtime_state,
                        mtp_unavailable_reason,
                    );
                    replacement_health_snapshot =
                        replacement_health_snapshot.with_mtp_depth_status(mtp_depth_status);
                    replacement_health_snapshot = replacement_health_snapshot
                        .with_speculative_prefill_runtime(
                            speculative_prefill_runtime_state,
                            speculative_prefill_unavailable_reason,
                            speculative_prefill_draft_model_id,
                            speculative_prefill_draft_model_revision,
                        );
                    replacement_health_snapshot.minimum_mlx_memory_ceiling_bytes =
                        minimum_mlx_memory_ceiling_bytes;
                    replacement_health_snapshot
                };
            ready_worker_health_snapshot.expert_memory_mode = expert_memory_mode;
            publish_health(health_snapshot, ready_worker_health_snapshot);
        }
        WorkerEvent::ModelSwapFailed { .. } => {
            return Err(protocol_violation(
                "model swap failure outside model swap wait",
            ));
        }
        WorkerEvent::GenerationFinalized {
            request_id,
            expert_memory_mode,
            mlx_memory_snapshot,
            expert_residency,
        } => {
            let Some(active_request) = active_generation.as_mut() else {
                return Err(protocol_violation(
                    "finalized generation state without an active request",
                ));
            };
            if request_id != active_request.request_id {
                return Err(protocol_violation(
                    "finalized generation state request mismatch",
                ));
            }
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                active_request.maximum_mlx_peak_memory_bytes = Some(
                    active_request
                        .maximum_mlx_peak_memory_bytes
                        .unwrap_or(0)
                        .max(mlx_memory_snapshot.peak_memory_bytes),
                );
                active_request.last_mlx_active_memory_bytes =
                    Some(mlx_memory_snapshot.active_memory_bytes);
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            } else {
                clear_latest_mlx_memory_snapshot(health_snapshot);
            }
            if let Some(expert_memory_mode) = expert_memory_mode {
                publish_expert_memory_mode(health_snapshot, expert_memory_mode);
            }
            if let Some(expert_residency) = expert_residency {
                active_request.final_complete_expert_layer_count =
                    Some(expert_residency.complete_layer_count);
                active_request.final_complete_expert_payload_bytes =
                    Some(expert_residency.complete_layer_payload_bytes);
                active_request.final_partial_expert_layer_count =
                    Some(expert_residency.partial_layer_count);
                active_request.final_partial_expert_payload_bytes =
                    Some(expert_residency.partial_layer_payload_bytes);
                crate::worker_health::publish_worker_expert_residency(
                    health_snapshot,
                    expert_residency,
                );
            }
        }
        worker_output_event @ WorkerEvent::Output { .. } => {
            handle_worker_output(worker_output_event, health_snapshot, active_generation)?;
        }
        worker_prefill_progress_event @ WorkerEvent::PrefillProgress { .. } => {
            handle_worker_prefill_progress(
                worker_prefill_progress_event,
                health_snapshot,
                active_generation,
            )?;
        }
        WorkerEvent::GenerationPreparationStarted {
            request_id,
            total_layer_count,
            complete_layer_count,
            complete_layer_payload_bytes,
            partial_layer_count,
            partial_layer_payload_bytes,
        } => {
            handle_generation_preparation_started(
                request_id,
                ExpertResidencySnapshot {
                    total_layer_count,
                    complete_layer_count,
                    complete_layer_payload_bytes,
                    partial_layer_count,
                    partial_layer_payload_bytes,
                },
                health_snapshot,
                active_generation,
            )?;
        }
        worker_generation_progress_event @ WorkerEvent::GenerationProgress { .. } => {
            handle_worker_generation_progress(
                worker_generation_progress_event,
                health_snapshot,
                active_generation,
            )?;
        }
        worker_first_decode_event @ WorkerEvent::FirstDecodeCompleted { .. } => {
            handle_worker_first_decode_completed(worker_first_decode_event, active_generation)?;
        }
        worker_prompt_work_reuse_event @ WorkerEvent::PromptWorkReuse { .. } => {
            handle_worker_prompt_work_reuse(
                worker_prompt_work_reuse_event,
                health_snapshot,
                active_generation,
            )?;
        }
        WorkerEvent::Completed {
            request_id,
            prompt_token_count,
            generated_token_count,
            reasoning_token_count,
            cached_token_count,
            persistent_prompt_cache_diagnostics,
            reason,
        } => {
            handle_worker_completion_event(
                request_id,
                prompt_token_count,
                generated_token_count,
                reasoning_token_count,
                cached_token_count,
                persistent_prompt_cache_diagnostics,
                reason,
                health_snapshot,
                active_generation,
                performance_log,
            )?;
        }
        WorkerEvent::Failed { request_id, reason } => {
            let Some(active_request) = active_generation.as_ref() else {
                return Err(protocol_violation("failure without an active request"));
            };
            if request_id != active_request.request_id {
                return Err(protocol_violation("failure request mismatch"));
            }
            let Some(failed_request) = active_generation.take() else {
                return Ok(());
            };
            tracing::warn!(request_id = request_id.value(), failure_reason = ?reason,
                "worker generation failed");
            publish_activity(health_snapshot, WorkerActivity::Idle);
            clear_active_request_progress(health_snapshot);
            try_send_stream_event(
                &failed_request.stream_event_sender,
                ChatGenerationStreamEvent::Failed { reason },
            )?;
        }
        WorkerEvent::PersistentPromptCacheStats { .. } => {
            publish_persistent_prompt_cache_stats(health_snapshot, worker_event);
        }
        WorkerEvent::PromptCacheCleared { .. } => {
            return Err(protocol_violation(
                "prompt-cache clear acknowledgement outside cache-clear wait",
            ));
        }
    }
    Ok(())
}

pub(super) fn protocol_violation(description: &'static str) -> WorkerControlError {
    WorkerControlError::WorkerProtocolViolation { description }
}
