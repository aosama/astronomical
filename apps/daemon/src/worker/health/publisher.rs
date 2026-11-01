//! Applies worker events to the shared daemon health snapshot.

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
    mut worker_health_snapshot: WorkerHealthSnapshot,
) {
    if let Ok(mut health_snapshot) = health_snapshot.write() {
        preserve_expert_streaming_observatory_across_health_replacement(
            &mut worker_health_snapshot,
            &health_snapshot,
        );
        *health_snapshot = worker_health_snapshot;
    }
}

/// Wholesale replacements (idle, ready, and load transitions) rebuild every
/// field from fresh worker events. The expert-streaming observatory is
/// daemon-accumulated state, so it survives every replacement except a
/// genuine model swap: only a replacement that claims a different ready model
/// than the snapshot already holds starts a fresh engine-load session.
/// Re-ready events during a load (existing claim not yet published) must not
/// reset the feed — that path was clearing `previous_mode` on the live daemon.
fn preserve_expert_streaming_observatory_across_health_replacement(
    replacement_snapshot: &mut WorkerHealthSnapshot,
    installed_snapshot: &WorkerHealthSnapshot,
) {
    let replacement_claims_different_model = match (
        replacement_snapshot.ready_model_id.as_ref(),
        installed_snapshot.ready_model_id.as_ref(),
    ) {
        (Some(replacement_model_id), Some(ready_model_id)) => {
            replacement_model_id != ready_model_id
        }
        _ => false,
    };
    if !replacement_claims_different_model
        && replacement_snapshot
            .expert_streaming_observatory
            .latest_stats
            .is_none()
    {
        replacement_snapshot.expert_streaming_observatory =
            installed_snapshot.expert_streaming_observatory.clone();
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
        worker_health_snapshot
            .expert_streaming_observatory
            .record_mode_change(expert_memory_mode, unix_millis_now());
    }
}

/// Current wall-clock in Unix milliseconds; the zero fallback only occurs
/// before the epoch, which no serving host exhibits.
fn unix_millis_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since_unix_epoch| since_unix_epoch.as_millis() as u64)
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
        ExpertResidencySnapshot::from(expert_residency),
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

pub(crate) fn publish_expert_streaming_stats(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    expert_streaming_stats_event: WorkerEvent,
) {
    let Some(worker_expert_streaming_stats) =
        super::expert_streaming::expert_streaming_stats_from_event(&expert_streaming_stats_event)
    else {
        return;
    };
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        tracing::info!(
            had_latest_stats = worker_health_snapshot
                .expert_streaming_observatory
                .latest_stats
                .is_some(),
            accumulated_prompt_tokens = worker_health_snapshot
                .expert_streaming_observatory
                .accumulated_prompt_token_count,
            snapshot_ptr = Arc::as_ptr(health_snapshot) as usize,
            "expert streaming stats recording"
        );
        worker_health_snapshot
            .expert_streaming_observatory
            .record_stats(worker_expert_streaming_stats, unix_millis_now());
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
    expert_residency: Option<WorkerExpertResidencySnapshot>,
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
        worker_health_snapshot.restart_required_mlx_memory_ceiling_bytes = None;
        worker_health_snapshot.expert_memory_mode = worker_health_snapshot
            .ready_model_id
            .as_ref()
            .map(|_| expert_memory_mode);
        worker_health_snapshot.latest_mlx_memory_snapshot = mlx_memory_snapshot;
        if let Some(expert_residency) = expert_residency {
            worker_health_snapshot.expert_residency = Some(expert_residency.into());
        }
    }
}

pub(crate) fn publish_mlx_memory_limit_restart_required(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    requested_mlx_memory_ceiling_bytes: u64,
    reason: &str,
) {
    if let Ok(mut worker_health_snapshot) = health_snapshot.write() {
        worker_health_snapshot.mlx_memory_limit_error = Some(format!(
            "restart required to reach the requested {requested_mlx_memory_ceiling_bytes}-byte ceiling: {reason}"
        ));
        worker_health_snapshot.restart_required_mlx_memory_ceiling_bytes =
            Some(requested_mlx_memory_ceiling_bytes);
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
        worker_health_snapshot.restart_required_mlx_memory_ceiling_bytes = None;
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
