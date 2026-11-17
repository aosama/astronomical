//! Correlates decisions events and releases shared ownership only after cleanup.

use std::sync::{Arc, RwLock};

use astronomical_decision_contract::DecisionFailure;
use astronomical_ipc_protocol::{RequestId, WorkerEvent, WorkerMlxMemorySnapshot};

use crate::worker::request_admission::SINGLE_SHOT_EXECUTION_TIMEOUT;
use crate::worker::undelivered_training_result::discard_undelivered_handover_file;
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
            record_terminal_outcome(
                active_decisions,
                Ok(DecisionsOutput::Answered {
                    question_outcomes,
                    input_token_count,
                    elapsed_millis,
                }),
            )
        }
        WorkerEvent::DecisionTrainingProgress {
            request_id,
            stage,
            completed,
            total,
        } => {
            let active_decisions = matching_active_decisions(active_worker_request, request_id)?;
            forward_training_progress(
                active_decisions,
                DecisionsOutput::TrainingProgress {
                    stage,
                    completed,
                    total,
                },
            );
            Ok(())
        }
        WorkerEvent::DecisionModelTrained {
            request_id,
            outcome,
            elapsed_millis,
        } => record_terminal_outcome(
            matching_active_decisions(active_worker_request, request_id)?,
            Ok(DecisionsOutput::ModelTrained {
                outcome,
                elapsed_millis,
            }),
        ),
        WorkerEvent::DecisionsFailed { request_id, reason } => record_terminal_outcome(
            matching_active_decisions(active_worker_request, request_id)?,
            Err(reason),
        ),
        WorkerEvent::DecisionsFinalized {
            request_id,
            mlx_memory_snapshot,
            ..
        } => finalize_decisions(
            request_id,
            mlx_memory_snapshot,
            health_snapshot,
            active_worker_request,
        ),
        _ => Ok(()),
    }
}

/// Holds the request's one terminal outcome until the worker finalizes it.
fn record_terminal_outcome(
    active_decisions: &mut ActiveDecisionsGeneration,
    terminal_outcome: Result<DecisionsOutput, DecisionFailure>,
) -> Result<(), WorkerControlError> {
    if active_decisions.terminal_outcome.is_some() {
        return Err(protocol_violation("duplicate decisions terminal outcome"));
    }
    active_decisions.terminal_outcome = Some(terminal_outcome);
    active_decisions.terminal_received_at = Some(tokio::time::Instant::now());
    Ok(())
}

/// Passes a training step on to the caller. A training reports every step,
/// and each report shows the engine is alive, so the stuck-engine deadline
/// moves on with it.
fn forward_training_progress(
    active_decisions: &mut ActiveDecisionsGeneration,
    training_progress: DecisionsOutput,
) {
    active_decisions.execution_deadline =
        tokio::time::Instant::now() + SINGLE_SHOT_EXECUTION_TIMEOUT;
    let result_sender = &active_decisions.decisions_result_sender;
    // Progress may be dropped when the caller reads slowly; one slot always
    // stays free for the result.
    if result_sender.capacity() > 1 {
        let _progress_outcome = result_sender.try_send(Ok(training_progress));
    }
}

/// Sends the terminal outcome to the caller once the worker has cleaned up,
/// and releases the worker.
fn finalize_decisions(
    request_id: RequestId,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
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
        Err(decision_failure) => Err(DecisionsExecutionError::WorkerFailure(decision_failure)),
    };
    if let Err(send_error) = result_sender.try_send(client_outcome)
        && let Ok(undelivered_output) = send_error.into_inner()
    {
        // The caller stopped waiting, so nothing will read the trained
        // tensors the worker wrote for it.
        discard_undelivered_handover_file(&undelivered_output, &std::env::temp_dir());
    }
    publish_activity(health_snapshot, crate::WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    // A decision engine owns no experts or context state either, so its
    // cleanup observation obeys the embeddings rules exactly.
    publish_single_shot_finalized_memory_snapshot(health_snapshot, mlx_memory_snapshot)
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
