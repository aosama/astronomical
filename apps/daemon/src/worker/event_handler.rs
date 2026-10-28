//! Dispatches typed worker events to daemon-owned request and health state.
//!
//! Process-scoped events (memory samples, readiness) are valid whenever their
//! payload is valid. Generation-scoped events additionally require a matching
//! active request. Model-swap waiting delegates here so both loops enforce the
//! same protocol rules instead of maintaining competing event allowlists.

use astronomical_ipc_protocol::WorkerEvent;
use std::sync::{Arc, RwLock};
use tokio::time::Instant;

use crate::{
    ChatGenerationStreamEvent, CompletionAttributionLog, ExpertResidencySnapshot,
    GenerationPerformanceLog, WorkerActivity, WorkerControlError, WorkerHealthSnapshot,
    serving::chat_generation_executor::try_send_stream_event,
    worker::generation::completion_event::handle_worker_completion_event,
    worker::generation::output::{
        handle_worker_first_decode_completed, handle_worker_generation_progress,
        handle_worker_output, handle_worker_prompt_work_reuse,
    },
    worker::generation::prefill_progress::handle_worker_prefill_progress,
    worker::generation::preparation::handle_generation_preparation_started,
    worker::health::{
        clear_active_request_progress, clear_latest_mlx_memory_snapshot, publish_activity,
        publish_expert_memory_mode, publish_health, publish_latest_mlx_memory_snapshot,
        publish_mlx_memory_limit_changed, publish_mlx_memory_limit_rejection,
        publish_persistent_prompt_cache_stats, publish_worker_expert_residency,
    },
    worker::loop_types::{ActiveGeneration, ActiveWorkerRequest},
};

/// Applies one typed worker event to daemon-owned request and health state.
///
/// Process-scoped events such as memory samples are valid whenever their own
/// payload is valid. Generation-scoped events additionally require a matching
/// active request. Model-swap waiting delegates here so both loops enforce the
/// same protocol rules instead of maintaining competing event allowlists.
#[expect(clippy::too_many_lines)]
pub(super) fn handle_worker_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    model_load_deadline: &mut Option<Instant>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::RuntimeFeatureConfigurationApplied {
            worker_runtime_feature_configuration,
        } => {
            let Ok(mut worker_health_snapshot) = health_snapshot.write() else {
                return Err(protocol_violation("worker health lock is unavailable"));
            };
            let acknowledged_model_id = worker_runtime_feature_configuration
                .loaded_model
                .as_ref()
                .map(|loaded_model| loaded_model.model_id());
            if acknowledged_model_id != worker_health_snapshot.ready_model_id.as_deref() {
                return Err(protocol_violation(
                    "runtime policy acknowledgement does not match the published model",
                ));
            }
            if let Some(previous_configuration) = worker_health_snapshot
                .worker_runtime_feature_configuration
                .as_ref()
                && previous_configuration.loaded_model
                    != worker_runtime_feature_configuration.loaded_model
            {
                return Err(protocol_violation(
                    "runtime policy changed without an atomic model transition",
                ));
            }
            worker_health_snapshot.worker_runtime_feature_configuration =
                Some(worker_runtime_feature_configuration);
        }
        WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode } => {
            publish_expert_memory_mode(health_snapshot, expert_memory_mode);
        }
        WorkerEvent::Ready {
            model_id,
            capabilities,
        } => {
            if *is_ready || active_request.is_some() {
                return Err(protocol_violation("duplicate worker readiness"));
            }
            *is_ready = true;
            tracing::info!(model = %model_id, "worker model is ready");
            *model_load_deadline = None;
            publish_health(
                health_snapshot,
                WorkerHealthSnapshot::ready_with_model(model_id, capabilities),
            );
        }
        WorkerEvent::Idle {
            machine_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            wired_residency_budget_bytes,
        } => {
            if *is_ready || active_request.is_some() {
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
                    wired_residency_budget_bytes,
                ),
            );
        }
        WorkerEvent::MlxMemorySample {
            mlx_memory_snapshot,
            expert_residency,
        } => {
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                // Standalone samples are process-scoped and publish the public
                // status snapshot; when a chat request is active they also
                // carry decode-phase peaks no other event delivers. Folding
                // them here keeps the durable request record at least as high
                // as anything the status endpoint shows: the MLX peak counter
                // resets at allocator cleanups, so any single snapshot can
                // miss the true request maximum.
                if let Some(chat_request) = active_request
                    .as_mut()
                    .and_then(ActiveWorkerRequest::chat_mut)
                {
                    chat_request
                        .fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
                }
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            } else {
                clear_latest_mlx_memory_snapshot(health_snapshot);
            }
            if let Some(expert_residency) = expert_residency {
                publish_worker_expert_residency(health_snapshot, expert_residency);
            }
        }
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            expert_memory_mode,
            mlx_memory_snapshot,
            expert_residency,
        } => {
            publish_mlx_memory_limit_changed(
                health_snapshot,
                effective_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes,
                expert_memory_mode,
                mlx_memory_snapshot,
                expert_residency,
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
        WorkerEvent::MlxMemoryLimitRestartRequired {
            requested_mlx_memory_ceiling_bytes,
            effective_mlx_memory_ceiling_bytes,
            reason,
            ..
        } => {
            // The worker keeps its loaded runtime and old ceilings. The memory
            // transition belongs to a stop-first restart: an idle apply already
            // returned to the endpoint, and a queued apply defers to the queued
            // reconciliation, so only bounded health state changes here.
            tracing::warn!(
                requested_mlx_memory_ceiling_bytes,
                effective_mlx_memory_ceiling_bytes,
                reason = %reason,
                "worker reported that reaching the requested MLX ceiling requires a worker restart"
            );
            crate::worker::health::publish_mlx_memory_limit_restart_required(
                health_snapshot,
                requested_mlx_memory_ceiling_bytes,
                &reason,
            );
        }
        WorkerEvent::ModelSwapped { .. } => {
            return Err(protocol_violation(
                "model swap acknowledgement outside model swap wait",
            ));
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
            let chat_request = active_chat_request_mut(active_request)?;
            if request_id != chat_request.request_id {
                return Err(protocol_violation(
                    "finalized generation state request mismatch",
                ));
            }
            if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
                chat_request
                    .fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
                chat_request.last_mlx_active_memory_bytes =
                    Some(mlx_memory_snapshot.active_memory_bytes);
                publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
            } else {
                clear_latest_mlx_memory_snapshot(health_snapshot);
            }
            if let Some(expert_memory_mode) = expert_memory_mode {
                publish_expert_memory_mode(health_snapshot, expert_memory_mode);
            }
            if let Some(expert_residency) = expert_residency {
                chat_request.final_resident_expert_count =
                    Some(expert_residency.resident_expert_count);
                chat_request.final_resident_expert_payload_bytes =
                    Some(expert_residency.resident_expert_payload_bytes);
                crate::worker::health::publish_worker_expert_residency(
                    health_snapshot,
                    expert_residency,
                );
            }
        }
        worker_output_event @ WorkerEvent::Output { .. } => {
            handle_worker_output(
                worker_output_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )?;
        }
        worker_prefill_progress_event @ WorkerEvent::PrefillProgress { .. } => {
            handle_worker_prefill_progress(
                worker_prefill_progress_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )?;
        }
        WorkerEvent::GenerationPreparationStarted {
            request_id,
            total_layer_count,
            resident_expert_count,
            resident_expert_payload_bytes,
            mlx_memory_snapshot,
        } => {
            if let Some(mlx_memory_snapshot) = &mlx_memory_snapshot {
                publish_latest_mlx_memory_snapshot(health_snapshot, *mlx_memory_snapshot);
            }
            handle_generation_preparation_started(
                request_id,
                ExpertResidencySnapshot {
                    total_layer_count,
                    resident_expert_count,
                    resident_expert_payload_bytes,
                },
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )?;
        }
        worker_generation_progress_event @ WorkerEvent::GenerationProgress { .. } => {
            handle_worker_generation_progress(
                worker_generation_progress_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )?;
        }
        worker_first_decode_event @ WorkerEvent::FirstDecodeCompleted { .. } => {
            handle_worker_first_decode_completed(
                worker_first_decode_event,
                active_chat_request_mut(active_request)?,
            )?;
        }
        worker_prompt_work_reuse_event @ WorkerEvent::PromptWorkReuse { .. } => {
            handle_worker_prompt_work_reuse(
                worker_prompt_work_reuse_event,
                health_snapshot,
                active_chat_request(active_request)?,
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
                active_request,
                performance_log,
                completion_log,
            )?;
        }
        WorkerEvent::Failed { request_id, reason } => {
            let Some(chat_request) = active_request.as_ref().and_then(ActiveWorkerRequest::chat)
            else {
                return Err(protocol_violation("failure without an active request"));
            };
            if request_id != chat_request.request_id {
                return Err(protocol_violation("failure request mismatch"));
            }
            let Some(ActiveWorkerRequest::Chat(failed_request)) = active_request.take() else {
                return Err(protocol_violation(
                    "chat failure while image generation is active",
                ));
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
        image_event @ (WorkerEvent::ImageGenerationProgress { .. }
        | WorkerEvent::ImageGenerationCompleted { .. }
        | WorkerEvent::ImageGenerationFailed { .. }
        | WorkerEvent::ImageGenerationFinalized { .. }) => {
            crate::worker::image_event::handle_worker_image_event(
                image_event,
                health_snapshot,
                active_request,
                performance_log,
            )?;
        }
        embeddings_event @ (WorkerEvent::EmbeddingsCompleted { .. }
        | WorkerEvent::EmbeddingsFailed { .. }
        | WorkerEvent::EmbeddingsFinalized { .. }) => {
            crate::worker::embeddings_event::handle_worker_embeddings_event(
                embeddings_event,
                health_snapshot,
                active_request,
            )?;
        }
    }
    Ok(())
}

fn active_chat_request_mut(
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<&mut ActiveGeneration, WorkerControlError> {
    active_request
        .as_mut()
        .and_then(ActiveWorkerRequest::chat_mut)
        .ok_or_else(|| protocol_violation("chat event without an active chat request"))
}

fn active_chat_request(
    active_request: &Option<ActiveWorkerRequest>,
) -> Result<&ActiveGeneration, WorkerControlError> {
    active_request
        .as_ref()
        .and_then(ActiveWorkerRequest::chat)
        .ok_or_else(|| protocol_violation("chat event without an active chat request"))
}

pub(super) fn protocol_violation(description: &'static str) -> WorkerControlError {
    WorkerControlError::WorkerProtocolViolation { description }
}
