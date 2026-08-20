//! Applies worker events to the shared supervisor health snapshot.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    ExpertMemoryMode, WorkerEvent, WorkerExpertResidencySnapshot, WorkerMlxMemorySnapshot,
    WorkerPromptWorkReuse,
};

use super::{
    ActiveRequestProgress, ExpertResidencySnapshot, PendingPromptCacheClear, WorkerActivity,
    WorkerHealthSnapshot,
};

pub(crate) fn publish_health(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    worker_health_snapshot: WorkerHealthSnapshot,
) {
    if let Ok(mut health_snapshot) = health_snapshot.write() {
        *health_snapshot = worker_health_snapshot;
    }
}

pub(crate) fn publish_activity(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    worker_activity: WorkerActivity,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.activity = worker_activity;
    }
}

pub(crate) fn publish_active_request_progress(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    progress: ActiveRequestProgress,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.active_request_progress = Some(progress);
    }
}

pub(crate) fn clear_active_request_progress(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.active_request_progress = None;
    }
}

pub(crate) fn publish_expert_memory_mode(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    expert_memory_mode: ExpertMemoryMode,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.expert_memory_mode = Some(expert_memory_mode);
    }
}

pub(crate) fn publish_expert_residency(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    expert_residency: ExpertResidencySnapshot,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.expert_residency = Some(expert_residency);
    }
}

/// Converts and publishes the latest worker-owned expert topology snapshot.
pub(crate) fn publish_worker_expert_residency(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    expert_residency: WorkerExpertResidencySnapshot,
) {
    publish_expert_residency(
        health_snapshot,
        ExpertResidencySnapshot {
            total_layer_count: expert_residency.total_layer_count,
            complete_layer_count: expert_residency.complete_layer_count,
            complete_layer_payload_bytes: expert_residency.complete_layer_payload_bytes,
            partial_layer_count: expert_residency.partial_layer_count,
            partial_layer_payload_bytes: expert_residency.partial_layer_payload_bytes,
        },
    );
}

pub(crate) fn publish_persistent_prompt_cache_stats(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    persistent_prompt_cache_stats: WorkerEvent,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.persistent_prompt_cache_stats = Some(persistent_prompt_cache_stats);
    }
}

pub(crate) fn publish_latest_mlx_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    mlx_memory_snapshot: WorkerMlxMemorySnapshot,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.latest_mlx_memory_snapshot = Some(mlx_memory_snapshot);
    }
}

pub(crate) fn clear_latest_mlx_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.latest_mlx_memory_snapshot = None;
    }
}

pub(crate) fn publish_pending_mlx_memory_ceiling(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    pending_mlx_memory_ceiling_bytes: Option<u64>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes = pending_mlx_memory_ceiling_bytes;
    }
}

pub(crate) fn publish_pending_prompt_cache_clear(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    pending_prompt_cache_clear: Option<PendingPromptCacheClear>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.pending_prompt_cache_clear = pending_prompt_cache_clear;
    }
}

pub(crate) fn publish_mlx_memory_limit_changed(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    effective_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    expert_memory_mode: ExpertMemoryMode,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.mlx_memory_ceiling_bytes = effective_mlx_memory_ceiling_bytes;
        worker_health_snapshot.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes = None;
        if let Some(configuration_generation) = worker_health_snapshot
            .pending_configuration_generation
            .take()
            && let Some(worker_configuration) = worker_health_snapshot
                .worker_runtime_feature_configuration
                .as_mut()
        {
            worker_configuration.configuration_generation = configuration_generation;
        }
        worker_health_snapshot.mlx_memory_limit_error = None;
        worker_health_snapshot.expert_memory_mode = worker_health_snapshot
            .ready_model_id
            .as_ref()
            .map(|_| expert_memory_mode);
        worker_health_snapshot.latest_mlx_memory_snapshot = mlx_memory_snapshot;
    }
}

pub(crate) fn publish_mlx_memory_limit_rejection(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    minimum_mlx_memory_ceiling_bytes: u64,
    reason: String,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.minimum_mlx_memory_ceiling_bytes = minimum_mlx_memory_ceiling_bytes;
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes = None;
        worker_health_snapshot.pending_configuration_generation = None;
        worker_health_snapshot.mlx_memory_limit_error = Some(reason);
    }
}

pub(crate) fn record_serving_session(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    prompt_token_count: u32,
    cached_token_count: u32,
    prefill_tok_per_second: Option<f64>,
    generation_tok_per_second: Option<f64>,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot
            .serving_session
            .record_completed_request(
                prompt_token_count,
                cached_token_count,
                prefill_tok_per_second,
                generation_tok_per_second,
            );
    }
}

pub(crate) fn record_prompt_work_reuse(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    prompt_work_reuse: WorkerPromptWorkReuse,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot
            .serving_session
            .record_prompt_work_reuse(prompt_work_reuse);
    }
}
