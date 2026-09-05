//! Correlates embeddings events and releases shared ownership only after cleanup.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::WorkerEvent;

use crate::{
    WorkerControlError, WorkerHealthSnapshot,
    worker_event_handler::protocol_violation,
    worker_health::{clear_active_request_progress, publish_activity},
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
        WorkerEvent::EmbeddingsFinalized { request_id, .. } => {
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
            Ok(())
        }
        _ => Ok(()),
    }
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
