//! Correlates the explicit prefill-to-decode preparation boundary with public health.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::RequestId;
use tokio::time::Instant;

use crate::{
    ActiveRequestProgress, ExpertResidencySnapshot, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot,
    worker::event_handler::protocol_violation,
    worker::health::{publish_active_request_progress, publish_activity, publish_expert_residency},
    worker::loop_types::ActiveGeneration,
};

/// Moves the public status from prompt processing to generation
/// preparation. An engine that announces before reconciling decode residency
/// sends no residency; the status then keeps the last observed one until the
/// first decode observation replaces it.
pub(in crate::worker) fn handle_generation_preparation_started(
    request_id: RequestId,
    expert_residency: Option<ExpertResidencySnapshot>,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut ActiveGeneration,
) -> Result<(), WorkerControlError> {
    validate_preparation_announcement(request_id, expert_residency, active_request)?;
    let preparation_started_at = Instant::now();
    active_request.generation_preparation_started_at = Some(preparation_started_at);
    publish_activity(health_snapshot, WorkerActivity::GenerationPreparation);
    if let Some(expert_residency) = expert_residency {
        record_final_expert_residency(active_request, expert_residency);
        publish_expert_residency(health_snapshot, expert_residency);
    }
    publish_active_request_progress(
        health_snapshot,
        ActiveRequestProgress::GenerationPreparation {
            request_started_at: active_request.request_started_at,
            preparation_started_at,
            preparation_residency_counts: expert_residency.map(Into::into),
        },
    );
    Ok(())
}

/// One announcement per request, for this request, and a residency (when
/// present) whose count and payload agree about being empty.
fn validate_preparation_announcement(
    request_id: RequestId,
    expert_residency: Option<ExpertResidencySnapshot>,
    active_request: &ActiveGeneration,
) -> Result<(), WorkerControlError> {
    let has_inconsistent_residency = expert_residency.is_some_and(|expert_residency| {
        (expert_residency.resident_expert_count == 0)
            != (expert_residency.resident_expert_payload_bytes == 0)
    });
    if active_request.request_id != request_id
        || has_inconsistent_residency
        || active_request.generation_preparation_started_at.is_some()
    {
        return Err(protocol_violation(
            "generation preparation was duplicated or had a correlation or payload mismatch",
        ));
    }
    Ok(())
}

/// Keeps the request's final residency at the latest one observed, so the
/// performance record reports decode residency even when preparation was
/// announced before it was reconciled.
pub(in crate::worker) fn record_final_expert_residency(
    active_request: &mut ActiveGeneration,
    expert_residency: ExpertResidencySnapshot,
) {
    active_request.final_resident_expert_count = Some(expert_residency.resident_expert_count);
    active_request.final_resident_expert_payload_bytes =
        Some(expert_residency.resident_expert_payload_bytes);
}
