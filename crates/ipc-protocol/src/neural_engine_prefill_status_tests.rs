use super::NeuralEnginePrefillStatus;

#[test]
fn should_serialize_the_active_state_with_its_tag() {
    let status = NeuralEnginePrefillStatus::Active {
        layer_program_count: 32,
        neural_engine_tokens_per_chunk: 896,
        chunk_token_count: 2048,
        estimated_memory_bytes: 7,
    };

    assert_eq!(
        serde_json::to_value(&status).expect("the status serializes"),
        serde_json::json!({
            "state": "active",
            "layer_program_count": 32,
            "neural_engine_tokens_per_chunk": 896,
            "chunk_token_count": 2048,
            "estimated_memory_bytes": 7
        })
    );
}

#[test]
fn should_round_trip_an_unavailable_reason() {
    let status = NeuralEnginePrefillStatus::Unavailable {
        reason: "dense models only".to_owned(),
    };

    let encoded = serde_json::to_string(&status).expect("the status serializes");

    assert_eq!(
        serde_json::from_str::<NeuralEnginePrefillStatus>(&encoded).expect("the status parses"),
        status
    );
}

#[test]
fn should_refuse_an_unknown_state() {
    assert!(serde_json::from_str::<NeuralEnginePrefillStatus>(r#"{"state":"maybe"}"#).is_err());
}

#[test]
fn should_round_trip_the_status_event() {
    let event = crate::WorkerEvent::NeuralEnginePrefillChanged {
        neural_engine_prefill: NeuralEnginePrefillStatus::Unavailable {
            reason: "only dense models are supported".to_owned(),
        },
    };

    let encoded = serde_json::to_string(&event).expect("the event serializes");

    assert_eq!(
        serde_json::from_str::<crate::WorkerEvent>(&encoded).expect("the event parses"),
        event
    );
    assert_eq!(event.diagnostic_summary(), "neural_engine_prefill_changed");
}
