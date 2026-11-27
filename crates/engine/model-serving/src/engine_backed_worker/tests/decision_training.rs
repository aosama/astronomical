//! User-journey coverage for training a decision model's final layers in
//! the worker: progress per step, then the trained result, then
//! finalization; a model that cannot train refuses; a failing step ends
//! the training and drops its state.

use decision_contract::DecisionFailure;
use decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, DecisionTrainingStage, RunEnding, RunSummary,
};
use ipc_protocol::{DecisionTrainingCommand, RequestId, WorkerCommand, WorkerEvent};

use super::decisions::{
    DECISION_MODEL_ID, DecisionsWorkerHarness, close, start_worker_and_swap_in_a_trainable_model,
};
use crate::decision_engine::DecisionTrainingStep;

fn progress(stage: DecisionTrainingStage, completed: u64, total: u64) -> DecisionTrainingStep {
    DecisionTrainingStep::Progress {
        stage,
        completed,
        total,
    }
}

fn trained_outcome() -> DecisionTrainingOutcome {
    DecisionTrainingOutcome {
        examples: Vec::new(),
        trained_tensors: Vec::new(),
        adapter_tensors: Vec::new(),
        runs: vec![RunSummary {
            seed: 2,
            best_choosing_loss: 0.31,
            best_epoch: 12,
            epochs_run: 18,
            ending: RunEnding::StoppedImproving,
        }],
        best_seed: 2,
        largest_parity_gap: 0.000_3,
        parity_gap_beyond_rounding: None,
        recomputed_text_count: 0,
        trained_tensors_file: None,
        indistinguishable_question_count: 1,
    }
}

async fn send_training(harness: &mut DecisionsWorkerHarness, request_number: u64) {
    harness
        .worker
        .commands
        .send_command(&WorkerCommand::TrainDecisionModel(
            DecisionTrainingCommand {
                request_id: RequestId::new(request_number),
                model: DECISION_MODEL_ID.to_owned(),
                job: DecisionTrainingJob::default(),
            },
        ))
        .await
        .expect("the training command should reach the worker");
}

async fn swapped_trainable_model(
    scripted_training_steps: Vec<Result<DecisionTrainingStep, DecisionFailure>>,
) -> DecisionsWorkerHarness {
    let mut harness = start_worker_and_swap_in_a_trainable_model(scripted_training_steps).await;
    assert!(matches!(
        harness.worker.next_event().await,
        WorkerEvent::ModelSwapped { .. }
    ));
    harness
}

#[tokio::test]
async fn should_report_each_step_then_the_trained_result_then_finalize() {
    let mut harness = swapped_trainable_model(vec![
        Ok(progress(DecisionTrainingStage::Reading, 1, 1)),
        Ok(progress(DecisionTrainingStage::Training, 40, 120)),
        Ok(DecisionTrainingStep::Completed(Box::new(trained_outcome()))),
    ])
    .await;
    send_training(&mut harness, 61).await;

    let mut reported_steps = Vec::new();
    for _ in 0..2 {
        let progress_event = harness.worker.next_event().await;
        let WorkerEvent::DecisionTrainingProgress {
            request_id,
            stage,
            completed,
            total,
        } = progress_event
        else {
            panic!("each step should be reported: {progress_event:?}");
        };
        assert_eq!(request_id, RequestId::new(61));
        reported_steps.push((stage, completed, total));
    }
    assert_eq!(
        reported_steps,
        [
            (DecisionTrainingStage::Reading, 1, 1),
            (DecisionTrainingStage::Training, 40, 120)
        ]
    );
    let trained_event = harness.worker.next_event().await;
    let WorkerEvent::DecisionModelTrained { outcome, .. } = trained_event else {
        panic!("the training should end with its result: {trained_event:?}");
    };
    assert_eq!(*outcome, trained_outcome());
    assert!(matches!(
        harness.worker.next_event().await,
        WorkerEvent::DecisionsFinalized { request_id, .. } if request_id == RequestId::new(61)
    ));
    close(harness).await;
}

#[tokio::test]
async fn should_refuse_to_train_a_model_whose_final_layers_cannot_be_trained() {
    let mut harness = swapped_trainable_model(Vec::new()).await;
    send_training(&mut harness, 62).await;

    let refusal_event = harness.worker.next_event().await;
    let WorkerEvent::DecisionsFailed { reason, .. } = refusal_event else {
        panic!("the worker should refuse the training: {refusal_event:?}");
    };
    assert!(reason.to_string().contains("cannot be trained"), "{reason}");
    assert!(matches!(
        harness.worker.next_event().await,
        WorkerEvent::DecisionsFinalized { .. }
    ));
    close(harness).await;
}

#[tokio::test]
async fn should_end_a_training_whose_step_fails_and_drop_its_state() {
    let mut harness = swapped_trainable_model(vec![
        Ok(progress(DecisionTrainingStage::Reading, 1, 2)),
        Err(DecisionFailure::fatal_execution(
            "the GPU copy of the final layers scores 0.00876 away from the model",
        )),
    ])
    .await;
    send_training(&mut harness, 63).await;

    assert!(matches!(
        harness.worker.next_event().await,
        WorkerEvent::DecisionTrainingProgress { completed: 1, .. }
    ));
    let failure_event = harness.worker.next_event().await;
    let WorkerEvent::DecisionsFailed { reason, .. } = failure_event else {
        panic!("the failing step should end the training: {failure_event:?}");
    };
    assert!(reason.to_string().contains("0.00876"), "{reason}");
    assert!(matches!(
        harness.worker.next_event().await,
        WorkerEvent::DecisionsFinalized { .. }
    ));
    assert!(
        *harness
            .was_training_cancelled
            .lock()
            .expect("the cancellation flag should not be poisoned"),
        "a failed training must not leave its state behind"
    );
    close(harness).await;
}

/// Sends a limit change and returns every event up to the request's
/// finalization, split into the limit answers and the rest.
async fn events_after_a_limit_change_until_finalized(
    harness: &mut DecisionsWorkerHarness,
) -> (Vec<WorkerEvent>, Vec<WorkerEvent>) {
    harness
        .worker
        .commands
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 4_000,
            configuration_generation: "generation-during-training".to_owned(),
        })
        .await
        .expect("the limit change should reach the worker");
    let mut limit_answers = Vec::new();
    let mut other_events = Vec::new();
    loop {
        let worker_event = harness.worker.next_event().await;
        match worker_event {
            WorkerEvent::MlxMemoryLimitChanged { .. }
            | WorkerEvent::MlxMemoryLimitRejected { .. } => limit_answers.push(worker_event),
            WorkerEvent::DecisionsFinalized { .. } => return (limit_answers, other_events),
            _ => other_events.push(worker_event),
        }
    }
}

#[tokio::test]
async fn should_refuse_a_memory_limit_change_during_training_and_finish_the_training() {
    let mut harness = swapped_trainable_model(vec![
        Ok(progress(DecisionTrainingStage::Reading, 1, 3)),
        Ok(progress(DecisionTrainingStage::Training, 2, 3)),
        Ok(progress(DecisionTrainingStage::Training, 3, 3)),
        Ok(DecisionTrainingStep::Completed(Box::new(trained_outcome()))),
    ])
    .await;
    send_training(&mut harness, 64).await;
    assert!(matches!(
        harness.worker.next_event().await,
        WorkerEvent::DecisionTrainingProgress { completed: 1, .. }
    ));

    let (limit_answers, other_events) =
        events_after_a_limit_change_until_finalized(&mut harness).await;

    let [WorkerEvent::MlxMemoryLimitRejected { reason, .. }] = limit_answers.as_slice() else {
        panic!("the change should be refused once: {limit_answers:?}");
    };
    assert!(reason.contains("try again when it ends"), "{reason}");
    assert!(
        other_events
            .iter()
            .any(|worker_event| matches!(worker_event, WorkerEvent::DecisionModelTrained { .. })),
        "the training must complete after the refusal: {other_events:?}"
    );
    close(harness).await;
}
