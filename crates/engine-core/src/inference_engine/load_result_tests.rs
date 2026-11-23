use ipc_protocol::{
    DraftModelGeneration, DraftModelLocation, DraftModelStatus, NeuralEnginePrefillStatus,
    WorkerEvent,
};

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

#[test]
fn should_announce_no_drafter_state_for_an_engine_that_never_drafts() {
    assert_eq!(EngineLoadResult::new().draft_model_event(), None);
}

#[test]
fn should_announce_that_the_loaded_model_drafts() {
    let drafting_status = DraftModelStatus::Drafting {
        generation: DraftModelGeneration::Dflash,
        location: DraftModelLocation::Bundled,
        tree_node_budget: 4,
    };
    let load_result = EngineLoadResult::new().with_draft_model_status(drafting_status.clone());
    assert_eq!(
        load_result.draft_model_event(),
        Some(WorkerEvent::DraftModelChanged {
            draft_model: drafting_status
        })
    );
}

#[test]
fn should_announce_a_model_without_a_drafter_rather_than_stay_silent() {
    let load_result = EngineLoadResult::new().with_draft_model_status(DraftModelStatus::None);
    assert_eq!(
        load_result.draft_model_event(),
        Some(WorkerEvent::DraftModelChanged {
            draft_model: DraftModelStatus::None
        })
    );
}
