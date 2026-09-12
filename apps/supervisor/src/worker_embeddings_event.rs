//! Correlates embeddings events and releases shared ownership only after cleanup.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{MlxMemorySnapshotSource, WorkerEvent};

use crate::{
    WorkerControlError, WorkerHealthSnapshot,
    worker_event_handler::protocol_violation,
    worker_health::{
        clear_active_request_progress, publish_activity, publish_latest_mlx_memory_snapshot,
    },
    worker_loop_types::ActiveWorkerRequest,
};

pub(super) fn handle_worker_embeddings_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::EmbeddingsCompleted {
            request_id,
            embeddings,
            input_token_counts,
            elapsed_millis,
        } => {
            let active_embeddings = matching_active_embeddings(active_worker_request, request_id)?;
            if active_embeddings.terminal_outcome.is_some()
                || embeddings.len() != input_token_counts.len()
                || embeddings.iter().any(Vec::is_empty)
            {
                return Err(protocol_violation("invalid completed embeddings result"));
            }
            active_embeddings.terminal_outcome = Some(Ok(crate::EmbeddingsOutput {
                embeddings,
                input_token_counts,
                elapsed_millis,
            }));
            active_embeddings.terminal_received_at = Some(tokio::time::Instant::now());
            Ok(())
        }
        WorkerEvent::EmbeddingsFailed { request_id, reason } => {
            let active_embeddings = matching_active_embeddings(active_worker_request, request_id)?;
            if active_embeddings.terminal_outcome.is_some() {
                return Err(protocol_violation("duplicate embeddings terminal outcome"));
            }
            active_embeddings.terminal_outcome = Some(Err(reason));
            active_embeddings.terminal_received_at = Some(tokio::time::Instant::now());
            Ok(())
        }
        WorkerEvent::EmbeddingsFinalized {
            request_id,
            mlx_memory_snapshot,
            ..
        } => {
            let active_embeddings = matching_active_embeddings(active_worker_request, request_id)?;
            let Some(terminal_outcome) = active_embeddings.terminal_outcome.take() else {
                return Err(protocol_violation(
                    "embeddings finalized before any terminal outcome",
                ));
            };
            let result_sender = active_embeddings.embeddings_result_sender.clone();
            *active_worker_request = None;
            let _send_outcome = result_sender
                .try_send(terminal_outcome.map_err(crate::EmbeddingsExecutionError::WorkerFailure));
            publish_activity(health_snapshot, crate::WorkerActivity::Idle);
            clear_active_request_progress(health_snapshot);
            publish_embeddings_finalized_memory_snapshot(health_snapshot, mlx_memory_snapshot)?;
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Publishes the cleanup observation the embeddings engine captured, so the
/// status the menu paints carries the same unused-headroom split the journeys
/// assert (issue #510). The embeddings engine owns no experts, context
/// state, or drafter, so any nonzero owner there is a protocol violation.
pub(super) fn publish_embeddings_finalized_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    mlx_memory_snapshot: Option<astronomical_ipc_protocol::WorkerMlxMemorySnapshot>,
) -> Result<(), WorkerControlError> {
    let Some(mlx_memory_snapshot) = mlx_memory_snapshot else {
        return Ok(());
    };
    let attributed_memory_bytes = mlx_memory_snapshot
        .expert_payload_bytes
        .saturating_add(mlx_memory_snapshot.model_core_payload_bytes)
        .saturating_add(mlx_memory_snapshot.context_state_payload_bytes)
        .saturating_add(mlx_memory_snapshot.speculative_prefill_draft_memory_bytes);
    let effective_ceiling_bytes = health_snapshot
        .read()
        .ok()
        .map(|snapshot| snapshot.mlx_memory_ceiling_bytes)
        .unwrap_or(0);
    if mlx_memory_snapshot.source != MlxMemorySnapshotSource::Finalized
        || mlx_memory_snapshot.allocator_cache_memory_bytes != 0
        || mlx_memory_snapshot.active_memory_bytes > mlx_memory_snapshot.peak_memory_bytes
        || attributed_memory_bytes > mlx_memory_snapshot.active_memory_bytes
        || mlx_memory_snapshot.expert_payload_bytes != 0
        || mlx_memory_snapshot.context_state_payload_bytes != 0
        || mlx_memory_snapshot.speculative_prefill_draft_memory_bytes != 0
        || (effective_ceiling_bytes > 0
            && mlx_memory_snapshot.active_memory_bytes > effective_ceiling_bytes)
    {
        return Err(protocol_violation(
            "invalid embeddings finalization MLX memory snapshot",
        ));
    }
    publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    Ok(())
}

fn matching_active_embeddings(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: astronomical_ipc_protocol::RequestId,
) -> Result<&mut crate::worker_loop_types::ActiveEmbeddingsGeneration, WorkerControlError> {
    let Some(active_request) = active_worker_request else {
        return Err(protocol_violation(
            "received an embeddings event without an active request",
        ));
    };
    if active_request.request_id() != request_id {
        return Err(protocol_violation(
            "embeddings event request identifier mismatch",
        ));
    }
    let Some(active_embeddings) = active_request.embeddings_mut() else {
        return Err(protocol_violation(
            "embeddings events require an active embeddings request",
        ));
    };
    Ok(active_embeddings)
}
