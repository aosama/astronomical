//! The worker's two readiness events: ready with a model, and idle without
//! one. Each replaces the published health snapshot wholesale, and both are
//! refused while a request is active, so a worker cannot announce readiness
//! mid-request.

use std::sync::{Arc, RwLock};

use ipc_protocol::WorkerEvent;
use tokio::time::Instant;

use crate::{
    WorkerControlError, WorkerHealthSnapshot,
    worker::event_handler::protocol_violation,
    worker::health::{publish_health, publish_neural_engine_prefill},
    worker::loop_types::ActiveWorkerRequest,
};

/// Routes the two readiness events, with or without a loaded model, which
/// share their arguments; kept apart so `handle_worker_event` stays within
/// the repository's 100-line function budget.
pub(in crate::worker) fn apply_worker_readiness_event(
    readiness_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    is_ready: &mut bool,
    active_request: &mut Option<ActiveWorkerRequest>,
    model_load_deadline: &mut Option<Instant>,
) -> Result<(), WorkerControlError> {
    if matches!(readiness_event, WorkerEvent::Ready { .. }) {
        apply_worker_ready(
            readiness_event,
            health_snapshot,
            is_ready,
            active_request,
            model_load_deadline,
        )
    } else {
        apply_worker_idle(
            readiness_event,
            health_snapshot,
            is_ready,
            active_request,
            model_load_deadline,
        )
    }
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
    // A worker that opted in announces this right after readiness.
    publish_neural_engine_prefill(health_snapshot, None);
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
