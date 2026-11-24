use super::{NeuralEngineChunkPlacement, NeuralEngineChunkReport};

fn report(placement: NeuralEngineChunkPlacement) -> NeuralEngineChunkReport {
    NeuralEngineChunkReport {
        placement,
        chunk_token_count: 2_048,
        neural_engine_token_count: 960,
        elapsed_millis: 350,
        is_prompt_warm: true,
    }
}

#[test]
fn should_name_each_placement_in_snake_case() {
    assert_eq!(
        serde_json::to_value(NeuralEngineChunkPlacement::GpuMemoryTight)
            .expect("the placement serializes"),
        serde_json::json!("gpu_memory_tight")
    );
    assert_eq!(
        serde_json::to_value(NeuralEngineChunkPlacement::BothEngines)
            .expect("the placement serializes"),
        serde_json::json!("both_engines")
    );
}

#[test]
fn should_round_trip_the_chunk_event() {
    let event = crate::WorkerEvent::NeuralEngineChunkFinished {
        report: report(NeuralEngineChunkPlacement::GpuWakingPrograms),
    };

    let encoded = serde_json::to_string(&event).expect("the event serializes");

    assert_eq!(
        serde_json::from_str::<crate::WorkerEvent>(&encoded).expect("the event parses"),
        event
    );
    assert_eq!(event.diagnostic_summary(), "neural_engine_chunk_finished");
}

#[test]
fn should_refuse_an_unknown_placement() {
    assert!(serde_json::from_str::<NeuralEngineChunkPlacement>(r#""maybe""#).is_err());
}

#[test]
fn should_refuse_a_report_with_an_unknown_field() {
    let mut document =
        serde_json::to_value(report(NeuralEngineChunkPlacement::BothEngines)).expect("it encodes");
    document["surprise"] = serde_json::json!(1);
    assert!(serde_json::from_value::<NeuralEngineChunkReport>(document).is_err());
}
