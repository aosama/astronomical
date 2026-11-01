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
    ChatGenerationStreamEvent, CompletionAttributionLog, GenerationPerformanceLog, WorkerActivity,
    WorkerControlError, WorkerHealthSnapshot,
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
        publish_expert_memory_mode, publish_expert_streaming_stats, publish_health,
        publish_latest_mlx_memory_snapshot, publish_mlx_memory_limit_changed,
        publish_mlx_memory_limit_rejection, publish_persistent_prompt_cache_stats,
        publish_worker_expert_residency,
    },
    worker::loop_types::{ActiveGeneration, ActiveWorkerRequest},
};

/// Applies one typed worker event to daemon-owned request and health state.
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
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    match worker_event {
        runtime_feature_event @ WorkerEvent::RuntimeFeatureConfigurationApplied { .. } => {
            apply_runtime_feature_configuration(runtime_feature_event, health_snapshot)?;
        }
        WorkerEvent::ExpertMemoryModeChanged { expert_memory_mode } => {
            publish_expert_memory_mode(health_snapshot, expert_memory_mode)
        }
        worker_ready_event @ WorkerEvent::Ready { .. } => {
            apply_worker_ready(
                worker_ready_event,
                health_snapshot,
                is_ready,
                active_request,
                model_load_deadline,
            )?;
        }
        worker_idle_event @ WorkerEvent::Idle { .. } => {
            apply_worker_idle(
                worker_idle_event,
                health_snapshot,
                is_ready,
                active_request,
                model_load_deadline,
            )?;
        }
        memory_sample_event @ WorkerEvent::MlxMemorySample { .. } => {
            apply_standalone_memory_sample(memory_sample_event, health_snapshot, active_request);
        }
        memory_limit_changed_event @ WorkerEvent::MlxMemoryLimitChanged { .. } => {
            apply_mlx_memory_limit_changed(memory_limit_changed_event, health_snapshot);
        }
        memory_limit_rejected_event @ WorkerEvent::MlxMemoryLimitRejected { .. } => {
            apply_mlx_memory_limit_rejected(memory_limit_rejected_event, health_snapshot);
        }
        restart_required_event @ WorkerEvent::MlxMemoryLimitRestartRequired { .. } => {
            apply_mlx_memory_limit_restart_required(restart_required_event, health_snapshot);
        }
        swap_wait_event @ (WorkerEvent::ModelSwapped { .. }
        | WorkerEvent::ModelSwapFailed { .. }) => {
            return Err(outside_model_swap_wait_violation(&swap_wait_event));
        }
        generation_finalized_event @ WorkerEvent::GenerationFinalized { .. } => {
            apply_generation_finalized(
                generation_finalized_event,
                health_snapshot,
                active_request,
            )?;
        }
        worker_progress_event @ (WorkerEvent::Output { .. }
        | WorkerEvent::PrefillProgress { .. }
        | WorkerEvent::GenerationPreparationStarted { .. }
        | WorkerEvent::GenerationProgress { .. }
        | WorkerEvent::FirstDecodeCompleted { .. }
        | WorkerEvent::PromptWorkReuse { .. }) => {
            dispatch_chat_progress_event(worker_progress_event, health_snapshot, active_request)?;
        }
        worker_completed_event @ WorkerEvent::Completed { .. } => {
            apply_worker_completed(
                worker_completed_event,
                health_snapshot,
                active_request,
                performance_log,
                completion_log,
            )?;
        }
        generation_failed_event @ WorkerEvent::Failed { .. } => {
            apply_generation_failed(generation_failed_event, health_snapshot, active_request)?;
        }
        observability_event @ (WorkerEvent::PersistentPromptCacheStats { .. }
        | WorkerEvent::ExpertStreamingStats { .. }) => {
            dispatch_observability_worker_event(observability_event, health_snapshot);
        }
        prompt_cache_cleared_event @ WorkerEvent::PromptCacheCleared { .. } => {
            return Err(outside_cache_clear_wait_violation(
                &prompt_cache_cleared_event,
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

/// Publishes prompt-cache and expert-streaming snapshots that ride alongside
/// generation. They are observations, so they stay out of the generation match
/// and do not change request ownership.
pub(in crate::worker) fn dispatch_observability_worker_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    if matches!(worker_event, WorkerEvent::PersistentPromptCacheStats { .. }) {
        publish_persistent_prompt_cache_stats(health_snapshot, worker_event);
        return;
    }
    if matches!(worker_event, WorkerEvent::ExpertStreamingStats { .. }) {
        publish_expert_streaming_stats(health_snapshot, worker_event);
    }
}

/// Applies a runtime-feature acknowledgement: the acknowledged model must
/// match the published ready model, and the loaded model must not have
/// changed outside an atomic transition. Mirrors the inlined arm.
fn apply_runtime_feature_configuration(
    runtime_feature_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration,
    } = runtime_feature_event
    else {
        return Ok(());
    };
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
        && previous_configuration.loaded_model != worker_runtime_feature_configuration.loaded_model
    {
        return Err(protocol_violation(
            "runtime policy changed without an atomic model transition",
        ));
    }
    worker_health_snapshot.worker_runtime_feature_configuration =
        Some(worker_runtime_feature_configuration);
    Ok(())
}

/// Applies worker readiness with a loaded model. Mirrors the inlined arm,
/// including the duplicate-readiness protocol violation.
fn apply_worker_ready(
    worker_ready_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    active_request: &mut Option<ActiveWorkerRequest>,
    model_load_deadline: &mut Option<Instant>,
) -> Result<(), WorkerControlError> {
    // The dispatch match routes only Ready events here; the let-else keeps
    // the function total without a silent catch-all arm.
    let WorkerEvent::Ready {
        model_id,
        capabilities,
    } = worker_ready_event
    else {
        return Ok(());
    };
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
    Ok(())
}

/// Applies worker readiness without a model, carrying the reported memory
/// ceilings. Mirrors the inlined arm.
fn apply_worker_idle(
    worker_idle_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    active_request: &mut Option<ActiveWorkerRequest>,
    model_load_deadline: &mut Option<Instant>,
) -> Result<(), WorkerControlError> {
    // Same routing contract as apply_worker_ready.
    let WorkerEvent::Idle {
        machine_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        wired_residency_budget_bytes,
    } = worker_idle_event
    else {
        return Ok(());
    };
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
    Ok(())
}

/// Applies one standalone MLX memory sample: publishes the public status
/// snapshot and folds decode-phase peaks into the active chat request's
/// durable record. The fold keeps the request record at least as high as
/// anything the status endpoint shows: the MLX peak counter resets at
/// allocator cleanups, so any single snapshot can miss the true request
/// maximum.
fn apply_standalone_memory_sample(
    memory_sample_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::MlxMemorySample {
        mlx_memory_snapshot,
        expert_residency,
    } = memory_sample_event
    else {
        return;
    };
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        if let Some(chat_request) = active_request
            .as_mut()
            .and_then(ActiveWorkerRequest::chat_mut)
        {
            chat_request.fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
        }
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    } else {
        clear_latest_mlx_memory_snapshot(health_snapshot);
    }
    if let Some(expert_residency) = expert_residency {
        publish_worker_expert_residency(health_snapshot, expert_residency);
    }
}

/// Records that the live worker cannot reach the requested ceiling without a
/// stop-first restart. The worker keeps its loaded runtime and old ceilings:
/// an idle apply already returned to the endpoint, and a queued apply defers
/// to the queued reconciliation, so only bounded health state changes here.
fn apply_mlx_memory_limit_restart_required(
    restart_required_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::MlxMemoryLimitRestartRequired {
        requested_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
        reason,
        ..
    } = restart_required_event
    else {
        return;
    };
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

/// Applies a generation-finalized event to the matching active chat request:
/// final peaks, last active memory, expert mode, and final residency. Mirrors
/// the inlined arm, including both request-mismatch violations.
fn apply_generation_finalized(
    generation_finalized_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::GenerationFinalized {
        request_id,
        expert_memory_mode,
        mlx_memory_snapshot,
        expert_residency,
    } = generation_finalized_event
    else {
        return Ok(());
    };
    let chat_request = active_chat_request_mut(active_request)?;
    if request_id != chat_request.request_id {
        return Err(protocol_violation(
            "finalized generation state request mismatch",
        ));
    }
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        chat_request.fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
        chat_request.last_mlx_active_memory_bytes = Some(mlx_memory_snapshot.active_memory_bytes);
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    } else {
        clear_latest_mlx_memory_snapshot(health_snapshot);
    }
    if let Some(expert_memory_mode) = expert_memory_mode {
        publish_expert_memory_mode(health_snapshot, expert_memory_mode);
    }
    if let Some(expert_residency) = expert_residency {
        chat_request.final_resident_expert_count = Some(expert_residency.resident_expert_count);
        chat_request.final_resident_expert_payload_bytes =
            Some(expert_residency.resident_expert_payload_bytes);
        crate::worker::health::publish_worker_expert_residency(health_snapshot, expert_residency);
    }
    Ok(())
}

/// Applies a generation failure: drops the matching chat request, publishes
/// idle, clears request progress, and delivers the failure to the stream.
/// Mirrors the inlined arm, including all three protocol violations.
fn apply_generation_failed(
    generation_failed_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::Failed { request_id, reason } = generation_failed_event else {
        return Ok(());
    };
    let Some(chat_request) = active_request.as_ref().and_then(ActiveWorkerRequest::chat) else {
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
    )
}

/// Publishes an accepted memory-limit change. Mirrors the inlined arm.
fn apply_mlx_memory_limit_changed(
    memory_limit_changed_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::MlxMemoryLimitChanged {
        effective_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        expert_memory_mode,
        mlx_memory_snapshot,
        expert_residency,
    } = memory_limit_changed_event
    else {
        return;
    };
    publish_mlx_memory_limit_changed(
        health_snapshot,
        effective_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        expert_memory_mode,
        mlx_memory_snapshot,
        expert_residency,
    );
}

/// Publishes a rejected memory limit. Mirrors the inlined arm.
fn apply_mlx_memory_limit_rejected(
    memory_limit_rejected_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::MlxMemoryLimitRejected {
        minimum_mlx_memory_ceiling_bytes,
        reason,
        ..
    } = memory_limit_rejected_event
    else {
        return;
    };
    publish_mlx_memory_limit_rejection(health_snapshot, minimum_mlx_memory_ceiling_bytes, reason);
}

/// Publishes the preparation snapshot and delegates the preparation start to
/// the request-scoped handler. Mirrors the inlined arm.
fn apply_generation_preparation_started(
    preparation_started_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::GenerationPreparationStarted {
        request_id,
        expert_residency,
        mlx_memory_snapshot,
    } = preparation_started_event
    else {
        return Ok(());
    };
    if let Some(mlx_memory_snapshot) = &mlx_memory_snapshot {
        publish_latest_mlx_memory_snapshot(health_snapshot, *mlx_memory_snapshot);
    }
    handle_generation_preparation_started(
        request_id,
        expert_residency.into(),
        health_snapshot,
        active_chat_request_mut(active_request)?,
    )
}

/// Delegates a completed generation to the completion handler. Mirrors the
/// inlined arm.
fn apply_worker_completed(
    worker_completed_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::Completed {
        request_id,
        prompt_token_count,
        generated_token_count,
        reasoning_token_count,
        cached_token_count,
        persistent_prompt_cache_diagnostics,
        reason,
    } = worker_completed_event
    else {
        return Ok(());
    };
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
    )
}

/// Dispatches the request-scoped chat progress events (output fragments,
/// prefill progress, generation progress, first decode, prompt work reuse) to
/// their dedicated handlers. Routed from the main dispatch match; total
/// without a silent catch-all.
fn dispatch_chat_progress_event(
    worker_progress_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    match worker_progress_event {
        preparation_started_event @ WorkerEvent::GenerationPreparationStarted { .. } => {
            apply_generation_preparation_started(
                preparation_started_event,
                health_snapshot,
                active_request,
            )
        }
        worker_output_event @ WorkerEvent::Output { .. } => handle_worker_output(
            worker_output_event,
            health_snapshot,
            active_chat_request_mut(active_request)?,
        ),
        worker_prefill_progress_event @ WorkerEvent::PrefillProgress { .. } => {
            handle_worker_prefill_progress(
                worker_prefill_progress_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )
        }
        worker_generation_progress_event @ WorkerEvent::GenerationProgress { .. } => {
            handle_worker_generation_progress(
                worker_generation_progress_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )
        }
        worker_first_decode_event @ WorkerEvent::FirstDecodeCompleted { .. } => {
            handle_worker_first_decode_completed(
                worker_first_decode_event,
                active_chat_request_mut(active_request)?,
            )
        }
        worker_prompt_work_reuse_event @ WorkerEvent::PromptWorkReuse { .. } => {
            handle_worker_prompt_work_reuse(
                worker_prompt_work_reuse_event,
                health_snapshot,
                active_chat_request(active_request)?,
            )
        }
        _ => Ok(()),
    }
}

/// The protocol violation for a model-swap acknowledgement or failure that
/// arrives outside model-swap wait. Pure so the per-variant message is
/// unit-testable.
pub(super) fn outside_model_swap_wait_violation(
    swap_wait_event: &WorkerEvent,
) -> WorkerControlError {
    let description = match swap_wait_event {
        WorkerEvent::ModelSwapped { .. } => "model swap acknowledgement outside model swap wait",
        WorkerEvent::ModelSwapFailed { .. } => "model swap failure outside model swap wait",
        _ => "unexpected event in the model-swap wait violation",
    };
    protocol_violation(description)
}

/// The protocol violation for a prompt-cache clear acknowledgement outside
/// cache-clear wait.
pub(super) fn outside_cache_clear_wait_violation(
    prompt_cache_cleared_event: &WorkerEvent,
) -> WorkerControlError {
    let _ = prompt_cache_cleared_event;
    protocol_violation("prompt-cache clear acknowledgement outside cache-clear wait")
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
