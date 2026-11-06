use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, RequestId, WorkerEvent};

use super::{CancellationEventReaction, is_progress_of_request, react_to_cancellation_event};
use crate::worker::health::{WorkerHealthSnapshot, WorkerHealthStatus};

fn first_decode_of(request_number: u64) -> WorkerEvent {
    WorkerEvent::FirstDecodeCompleted {
        request_id: RequestId::new(request_number),
        elapsed_millis: 1,
    }
}

#[test]
fn should_recognize_progress_of_the_request_being_cancelled() {
    assert!(is_progress_of_request(
        &first_decode_of(7),
        RequestId::new(7)
    ));
}

#[test]
fn should_not_recognize_progress_of_another_request() {
    assert!(!is_progress_of_request(
        &first_decode_of(8),
        RequestId::new(7)
    ));
}

#[test]
fn should_not_treat_an_event_without_a_request_as_progress() {
    let prefill_event = WorkerEvent::NeuralEnginePrefillChanged {
        neural_engine_prefill: NeuralEnginePrefillStatus::Off,
    };

    assert!(!is_progress_of_request(&prefill_event, RequestId::new(7)));
}

#[test]
fn should_publish_a_neural_engine_status_found_during_a_cancellation() {
    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )));
    let preparing = NeuralEnginePrefillStatus::Preparing {
        completed_program_count: 1,
        total_program_count: 2,
        elapsed_ms: 1,
        estimated_remaining_ms: None,
        is_waiting_to_measure: false,
    };

    let reaction = react_to_cancellation_event(
        &health_snapshot,
        WorkerEvent::NeuralEnginePrefillChanged {
            neural_engine_prefill: preparing.clone(),
        },
        RequestId::new(7),
        false,
    )
    .expect("this event is expected during a cancellation");

    assert!(matches!(reaction, CancellationEventReaction::KeepDraining));
    assert_eq!(
        health_snapshot
            .read()
            .expect("the snapshot reads")
            .neural_engine_prefill,
        Some(preparing)
    );
}

#[test]
fn should_count_a_chunk_that_finishes_during_a_cancellation() {
    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )));

    let reaction = react_to_cancellation_event(
        &health_snapshot,
        WorkerEvent::NeuralEngineChunkFinished {
            report: astronomical_ipc_protocol::NeuralEngineChunkReport {
                placement: astronomical_ipc_protocol::NeuralEngineChunkPlacement::BothEngines,
                chunk_token_count: 2_048,
                neural_engine_token_count: 960,
                elapsed_millis: 350,
                is_prompt_warm: true,
            },
        },
        RequestId::new(7),
        false,
    )
    .expect("this event is expected during a cancellation");

    assert!(matches!(reaction, CancellationEventReaction::KeepDraining));
    let totals = health_snapshot
        .read()
        .expect("the snapshot reads")
        .serving_session
        .neural_engine
        .clone();
    assert_eq!(totals.neural_engine_token_count, 960);
    assert_eq!(totals.prompt_token_count, 2_048);
}

#[test]
fn should_refuse_progress_of_another_request_during_a_cancellation() {
    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )));

    assert!(
        react_to_cancellation_event(
            &health_snapshot,
            first_decode_of(8),
            RequestId::new(7),
            false,
        )
        .is_err()
    );
}
