//! Unit contracts for correlating a worker's decisions events with the one
//! active request: what is delivered to the caller, when the worker slot is
//! released, and which event orders are protocol violations.

use std::sync::{Arc, RwLock};

use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingOutcome, DecisionTrainingStage, RunEnding, RunSummary,
};
use astronomical_decision_contract::{
    DecisionAnswer, DecisionFailure, NoulAnswer, QuestionInputCoverage, QuestionOutcome,
};
use astronomical_ipc_protocol::{
    RequestId, WorkerDecisionCapabilities, WorkerEvent, WorkerModelCapabilities,
};
use tokio::sync::{Semaphore, mpsc};

use super::handle_worker_decisions_event;
use crate::worker::loop_types::{ActiveDecisionsGeneration, ActiveWorkerRequest};
use crate::{DecisionsExecutionError, DecisionsOutput, WorkerControlError, WorkerHealthSnapshot};

const ACTIVE_REQUEST_ID: u64 = 4;

type DecisionsResultReceiver = mpsc::Receiver<Result<DecisionsOutput, DecisionsExecutionError>>;

fn decision_health_snapshot() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("decision-model"),
        WorkerModelCapabilities::decisions(WorkerDecisionCapabilities {
            maximum_tokens_per_question: 512,
        }),
    )))
}

fn active_decisions_request() -> (Option<ActiveWorkerRequest>, DecisionsResultReceiver) {
    active_decisions_request_with_room(1)
}

/// An active request whose result channel holds `channel_capacity` messages.
fn active_decisions_request_with_room(
    channel_capacity: usize,
) -> (Option<ActiveWorkerRequest>, DecisionsResultReceiver) {
    let (decisions_result_sender, decisions_result_receiver) = mpsc::channel(channel_capacity);
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    let active_decisions = ActiveDecisionsGeneration {
        _active_generation_permit: active_generation_permit,
        request_id: RequestId::new(ACTIVE_REQUEST_ID),
        terminal_received_at: None,
        execution_deadline: tokio::time::Instant::now(),
        decisions_result_sender,
        terminal_outcome: None,
    };
    (
        Some(ActiveWorkerRequest::Decisions(active_decisions)),
        decisions_result_receiver,
    )
}

fn urgent_outcome() -> QuestionOutcome {
    QuestionOutcome {
        question_name: "urgent".to_owned(),
        answer: DecisionAnswer::Noul(NoulAnswer {
            probability_true: 0.75,
            answer_confidence: 0.75,
        }),
        option_logits: vec![-0.5, 0.5],
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 8,
            state_tokens_read: 8,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
        head_inputs: None,
        training_record: None,
    }
}

fn completed_event(request_identifier: u64, outcomes: Vec<QuestionOutcome>) -> WorkerEvent {
    WorkerEvent::DecisionsCompleted {
        request_id: RequestId::new(request_identifier),
        question_outcomes: outcomes,
        input_token_count: 20,
        elapsed_millis: 6,
    }
}

fn finalized_event(request_identifier: u64) -> WorkerEvent {
    WorkerEvent::DecisionsFinalized {
        request_id: RequestId::new(request_identifier),
        elapsed_millis: 6,
        mlx_memory_snapshot: None,
    }
}

fn assert_protocol_violation(
    handling_outcome: Result<(), WorkerControlError>,
    expected_description: &'static str,
) {
    match handling_outcome {
        Err(WorkerControlError::WorkerProtocolViolation { description }) => {
            assert_eq!(description, expected_description);
        }
        other => panic!("expected a worker protocol violation, got {other:?}"),
    }
}

#[tokio::test]
async fn should_deliver_the_answers_only_once_the_request_is_finalized() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, mut decisions_result_receiver) = active_decisions_request();

    handle_worker_decisions_event(
        completed_event(ACTIVE_REQUEST_ID, vec![urgent_outcome()]),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("a completed event for the active request should be accepted");
    assert!(
        decisions_result_receiver.try_recv().is_err(),
        "the caller must not hear before the worker has cleaned up"
    );
    assert!(
        active_worker_request.is_some(),
        "the worker slot stays owned until finalization"
    );

    handle_worker_decisions_event(
        finalized_event(ACTIVE_REQUEST_ID),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("the finalization should be accepted");

    assert!(
        active_worker_request.is_none(),
        "finalization frees the slot"
    );
    let delivered_output = decisions_result_receiver
        .try_recv()
        .expect("the answers should be delivered at finalization")
        .expect("the delivered outcome should be the answers");
    let DecisionsOutput::Answered {
        question_outcomes,
        input_token_count,
        ..
    } = delivered_output
    else {
        panic!("an answer request delivered training results");
    };
    assert_eq!(question_outcomes, vec![urgent_outcome()]);
    assert_eq!(input_token_count, 20);
}

#[tokio::test]
async fn should_deliver_the_workers_failure_at_finalization() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, mut decisions_result_receiver) = active_decisions_request();
    let failure_reason = DecisionFailure::invalid_request("question `urgent` needs instructions");

    handle_worker_decisions_event(
        WorkerEvent::DecisionsFailed {
            request_id: RequestId::new(ACTIVE_REQUEST_ID),
            reason: failure_reason.clone(),
        },
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("a failed event for the active request should be accepted");
    handle_worker_decisions_event(
        finalized_event(ACTIVE_REQUEST_ID),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("the finalization should be accepted");

    assert_eq!(
        decisions_result_receiver.try_recv(),
        Ok(Err(DecisionsExecutionError::WorkerFailure(failure_reason)))
    );
}

#[tokio::test]
async fn should_reject_a_completed_event_that_answers_no_question() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, _decisions_result_receiver) = active_decisions_request();

    let handling_outcome = handle_worker_decisions_event(
        completed_event(ACTIVE_REQUEST_ID, Vec::new()),
        &health_snapshot,
        &mut active_worker_request,
    );

    assert_protocol_violation(handling_outcome, "invalid completed decisions result");
}

#[tokio::test]
async fn should_reject_a_second_terminal_outcome() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, _decisions_result_receiver) = active_decisions_request();
    handle_worker_decisions_event(
        completed_event(ACTIVE_REQUEST_ID, vec![urgent_outcome()]),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("the first terminal outcome should be accepted");

    let handling_outcome = handle_worker_decisions_event(
        WorkerEvent::DecisionsFailed {
            request_id: RequestId::new(ACTIVE_REQUEST_ID),
            reason: DecisionFailure::EngineBusy,
        },
        &health_snapshot,
        &mut active_worker_request,
    );

    assert_protocol_violation(handling_outcome, "duplicate decisions terminal outcome");
}

#[tokio::test]
async fn should_reject_a_finalization_that_precedes_any_outcome() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, _decisions_result_receiver) = active_decisions_request();

    let handling_outcome = handle_worker_decisions_event(
        finalized_event(ACTIVE_REQUEST_ID),
        &health_snapshot,
        &mut active_worker_request,
    );

    assert_protocol_violation(
        handling_outcome,
        "decisions finalized before any terminal outcome",
    );
}

#[tokio::test]
async fn should_reject_an_event_for_a_different_request() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, _decisions_result_receiver) = active_decisions_request();

    let handling_outcome = handle_worker_decisions_event(
        completed_event(ACTIVE_REQUEST_ID + 1, vec![urgent_outcome()]),
        &health_snapshot,
        &mut active_worker_request,
    );

    assert_protocol_violation(
        handling_outcome,
        "received a decisions event for a different request",
    );
}

#[tokio::test]
async fn should_reject_an_event_when_no_request_is_active() {
    let health_snapshot = decision_health_snapshot();
    let mut active_worker_request = None;

    let handling_outcome = handle_worker_decisions_event(
        completed_event(ACTIVE_REQUEST_ID, vec![urgent_outcome()]),
        &health_snapshot,
        &mut active_worker_request,
    );

    assert_protocol_violation(
        handling_outcome,
        "received a decisions event without an active request",
    );
}

fn training_progress_event(completed: u64) -> WorkerEvent {
    WorkerEvent::DecisionTrainingProgress {
        request_id: RequestId::new(ACTIVE_REQUEST_ID),
        stage: DecisionTrainingStage::Training,
        completed,
        total: 120,
    }
}

fn trained_outcome() -> DecisionTrainingOutcome {
    DecisionTrainingOutcome {
        examples: Vec::new(),
        trained_tensors: Vec::new(),
        adapter_tensors: Vec::new(),
        runs: vec![RunSummary {
            seed: 1,
            best_choosing_loss: 0.4,
            best_epoch: 8,
            epochs_run: 14,
            ending: RunEnding::StoppedImproving,
        }],
        best_seed: 1,
        largest_parity_gap: 0.0,
        parity_gap_beyond_rounding: None,
        indistinguishable_question_count: 0,
    }
}

fn execution_deadline(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
) -> tokio::time::Instant {
    active_worker_request
        .as_mut()
        .and_then(ActiveWorkerRequest::decisions_mut)
        .expect("the training should still be active")
        .execution_deadline
}

#[tokio::test(start_paused = true)]
async fn should_forward_training_progress_and_move_the_stuck_engine_deadline() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, mut decisions_result_receiver) =
        active_decisions_request_with_room(4);
    let deadline_before = execution_deadline(&mut active_worker_request);
    tokio::time::advance(std::time::Duration::from_secs(30)).await;

    handle_worker_decisions_event(
        training_progress_event(7),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("progress for the active training should be accepted");

    assert!(execution_deadline(&mut active_worker_request) > deadline_before);
    assert!(matches!(
        decisions_result_receiver.try_recv(),
        Ok(Ok(DecisionsOutput::TrainingProgress {
            stage: DecisionTrainingStage::Training,
            completed: 7,
            total: 120,
        }))
    ));
}

#[tokio::test]
async fn should_drop_progress_rather_than_fill_the_slot_the_result_needs() {
    let health_snapshot = decision_health_snapshot();
    let (mut active_worker_request, mut decisions_result_receiver) =
        active_decisions_request_with_room(1);
    handle_worker_decisions_event(
        training_progress_event(1),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("progress for the active training should be accepted");
    handle_worker_decisions_event(
        WorkerEvent::DecisionModelTrained {
            request_id: RequestId::new(ACTIVE_REQUEST_ID),
            outcome: Box::new(trained_outcome()),
            elapsed_millis: 900,
        },
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("the trained result should be accepted");
    handle_worker_decisions_event(
        finalized_event(ACTIVE_REQUEST_ID),
        &health_snapshot,
        &mut active_worker_request,
    )
    .expect("the finalization should be accepted");

    let Ok(Ok(DecisionsOutput::ModelTrained { outcome, .. })) =
        decisions_result_receiver.try_recv()
    else {
        panic!("the trained result must be the first and only delivery");
    };
    assert_eq!(*outcome, trained_outcome());
    assert!(active_worker_request.is_none());
}

#[tokio::test]
async fn should_reject_progress_for_a_training_that_is_not_running() {
    let health_snapshot = decision_health_snapshot();
    let mut active_worker_request = None;
    assert_protocol_violation(
        handle_worker_decisions_event(
            training_progress_event(1),
            &health_snapshot,
            &mut active_worker_request,
        ),
        "received a decisions event without an active request",
    );
}
