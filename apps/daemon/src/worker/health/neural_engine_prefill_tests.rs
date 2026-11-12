//! The Neural Engine status a daemon publishes for the loaded model.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerModelCapabilities};

use super::{WorkerHealthSnapshot, publish_neural_engine_prefill};

fn no_capabilities() -> WorkerModelCapabilities {
    WorkerModelCapabilities {
        chat: None,
        image_generation: None,
        embeddings: None,
        decisions: None,
        speech: None,
    }
}

fn ready_snapshot() -> WorkerHealthSnapshot {
    WorkerHealthSnapshot::ready_with_model(String::from("first-model"), no_capabilities())
}

#[test]
fn should_start_with_no_status_until_a_worker_reports_one() {
    assert_eq!(ready_snapshot().neural_engine_prefill, None);
}

#[test]
fn should_publish_the_status_a_worker_reports() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot()));
    let status = NeuralEnginePrefillStatus::Unavailable {
        reason: "only dense models are supported".to_owned(),
    };

    publish_neural_engine_prefill(&health_snapshot, Some(status.clone()));

    assert_eq!(
        health_snapshot
            .read()
            .expect("the snapshot reads")
            .neural_engine_prefill,
        Some(status)
    );
}

#[test]
fn should_clear_the_status_when_a_replacement_model_is_ready() {
    let mut first_model_snapshot = ready_snapshot();
    first_model_snapshot.neural_engine_prefill = Some(NeuralEnginePrefillStatus::Active {
        layer_program_count: 32,
        neural_engine_tokens_per_chunk: 960,
        chunk_token_count: 2048,
        measured_memory_bytes: 1,
        measured_speed_percent: None,
    });

    let replacement_snapshot = WorkerHealthSnapshot::ready_with_replacement_model(
        String::from("second-model"),
        no_capabilities(),
        1,
        &first_model_snapshot,
    );

    assert_eq!(replacement_snapshot.neural_engine_prefill, None);
}
