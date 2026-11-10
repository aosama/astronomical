//! Correlates decisions events and releases shared ownership only after cleanup.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{RequestId, WorkerEvent};

use crate::{
    DecisionsExecutionError, DecisionsOutput, WorkerControlError, WorkerHealthSnapshot,
    worker::embeddings_event::publish_single_shot_finalized_memory_snapshot,
    worker::event_handler::protocol_violation,
    worker::health::{clear_active_request_progress, publish_activity},
    worker::loop_types::{ActiveDecisionsGeneration, ActiveWorkerRequest},
};

pub(super) fn handle_worker_decisions_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::DecisionsCompleted {
            request_id,
            question_outcomes,
            input_token_count,
            elapsed_millis,
        } => {
            let active_decisions = matching_active_decisions(active_worker_request, request_id)?;
            if active_decisions.terminal_outcome.is_some() || question_outcomes.is_empty() {
                return Err(protocol_violation("invalid completed decisions result"));
            }
            active_decisions.terminal_outcome = Some(Ok(DecisionsOutput {
                question_outcomes,
                input_token_count,
                elapsed_millis,
            }));
            active_decisions.terminal_received_at = Some(tokio::time::Instant::now());
            Ok(())
        }
        WorkerEvent::DecisionsFailed { request_id, reason } => {
            let active_decisions = matching_active_decisions(active_worker_request, request_id)?;
            if active_decisions.terminal_outcome.is_some() {
                return Err(protocol_violation("duplicate decisions terminal outcome"));
            }
            active_decisions.terminal_outcome = Some(Err(reason));
            active_decisions.terminal_received_at = Some(tokio::time::Instant::now());
            Ok(())
        }
        WorkerEvent::DecisionsFinalized {
            request_id,
            mlx_memory_snapshot,
            ..
        } => {
            let active_decisions = matching_active_decisions(active_worker_request, request_id)?;
            let Some(terminal_outcome) = active_decisions.terminal_outcome.take() else {
                return Err(protocol_violation(
                    "decisions finalized before any terminal outcome",
                ));
            };
            let result_sender = active_decisions.decisions_result_sender.clone();
            *active_worker_request = None;
            let client_outcome = match terminal_outcome {
                Ok(decisions_output) => Ok(decisions_output),
                Err(decision_failure) => {
                    Err(DecisionsExecutionError::WorkerFailure(decision_failure))
                }
            };
            let _send_outcome = result_sender.try_send(client_outcome);
            publish_activity(health_snapshot, crate::WorkerActivity::Idle);
            clear_active_request_progress(health_snapshot);
            // A decision engine owns no experts or context state either, so
            // its cleanup observation obeys the embeddings rules exactly.
            publish_single_shot_finalized_memory_snapshot(health_snapshot, mlx_memory_snapshot)?;
            Ok(())
        }
        _ => Ok(()),
    }
}

fn matching_active_decisions(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: RequestId,
) -> Result<&mut ActiveDecisionsGeneration, WorkerControlError> {
    let Some(active_request) = active_worker_request else {
        return Err(protocol_violation(
            "received a decisions event without an active request",
        ));
    };
    let Some(active_decisions) = active_request.decisions_mut() else {
        return Err(protocol_violation(
            "received a decisions event while another kind of request is active",
        ));
    };
    if active_decisions.request_id != request_id {
        return Err(protocol_violation(
            "received a decisions event for a different request",
        ));
    }
    Ok(active_decisions)
}

#[cfg(test)]
#[path = "decisions_event_tests.rs"]
mod decisions_event_tests;
