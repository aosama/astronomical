//! Correlates the explicit prefill-to-decode preparation boundary with public health.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::RequestId;
use tokio::time::Instant;

use crate::{
    ActiveRequestProgress, ExpertResidencySnapshot, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot,
    worker_event_handler::protocol_violation,
    worker_health::{publish_active_request_progress, publish_activity, publish_expert_residency},
    worker_loop_types::ActiveGeneration,
};

pub(super) fn handle_generation_preparation_started(
    request_id: RequestId,
    expert_residency: ExpertResidencySnapshot,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveGeneration>,
) -> Result<(), WorkerControlError> {
    let Some(active_request) = active_generation.as_mut() else {
        return Err(protocol_violation(
            "generation preparation without an active request",
        ));
    };
    if active_request.request_id != request_id
        || expert_residency
            .complete_layer_count
            .saturating_add(expert_residency.partial_layer_count)
            > expert_residency.total_layer_count
        || (expert_residency.complete_layer_count == 0)
            != (expert_residency.complete_layer_payload_bytes == 0)
        || (expert_residency.partial_layer_count == 0)
            != (expert_residency.partial_layer_payload_bytes == 0)
        || active_request.generation_preparation_started_at.is_some()
    {
        return Err(protocol_violation(
            "generation preparation was duplicated or had a correlation, topology count, or payload mismatch",
        ));
    }
    let preparation_started_at = Instant::now();
    active_request.generation_preparation_started_at = Some(preparation_started_at);
    active_request.final_complete_expert_layer_count = Some(expert_residency.complete_layer_count);
    active_request.final_partial_expert_layer_count = Some(expert_residency.partial_layer_count);
    publish_activity(health_snapshot, WorkerActivity::GenerationPreparation);
    publish_expert_residency(health_snapshot, expert_residency);
    publish_active_request_progress(
        health_snapshot,
        ActiveRequestProgress::GenerationPreparation {
            request_started_at: active_request.request_started_at,
            preparation_started_at,
            total_layer_count: expert_residency.total_layer_count,
            complete_layer_count: expert_residency.complete_layer_count,
            partial_layer_count: expert_residency.partial_layer_count,
        },
    );
    Ok(())
}
