use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerEvent};

use super::EngineLoadResult;

#[test]
fn should_announce_nothing_when_the_setting_is_off() {
    assert_eq!(EngineLoadResult::new().neural_engine_prefill_event(), None);
}

#[test]
fn should_announce_an_unavailable_status_with_its_reason() {
    let status = NeuralEnginePrefillStatus::Unavailable {
        reason: "only dense models are supported".to_owned(),
    };

    let event = EngineLoadResult::new()
        .with_neural_engine_prefill_status(status.clone())
        .neural_engine_prefill_event();

    assert_eq!(
        event,
        Some(WorkerEvent::NeuralEnginePrefillChanged {
            neural_engine_prefill: status
        })
    );
}

#[test]
fn should_announce_an_active_status() {
    let status = NeuralEnginePrefillStatus::Active {
        layer_program_count: 32,
        neural_engine_tokens_per_chunk: 960,
        chunk_token_count: 2048,
        measured_memory_bytes: 1,
        measured_speed_percent: None,
    };

    let event = EngineLoadResult::new()
        .with_neural_engine_prefill_status(status.clone())
        .neural_engine_prefill_event();

    assert_eq!(
        event,
        Some(WorkerEvent::NeuralEnginePrefillChanged {
            neural_engine_prefill: status
        })
    );
}

#[test]
fn should_explain_an_unsupported_engine_only_when_the_user_opted_in() {
    assert_eq!(
        EngineLoadResult::neural_engine_prefill_unsupported_when_enabled(false),
        NeuralEnginePrefillStatus::Off
    );
    assert!(matches!(
        EngineLoadResult::neural_engine_prefill_unsupported_when_enabled(true),
        NeuralEnginePrefillStatus::Unavailable { .. }
    ));
}
