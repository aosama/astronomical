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
