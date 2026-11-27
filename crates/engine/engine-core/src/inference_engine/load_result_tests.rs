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

#[test]
fn should_announce_no_custom_kernels_for_an_engine_without_them() {
    let load_result = EngineLoadResult::new();
    assert_eq!(load_result.custom_metal_kernels_event(), None);
    assert!(load_result.engine_status_events().is_empty());
}

#[test]
fn should_announce_the_custom_kernel_report_after_the_drafter() {
    let custom_metal_kernels = ipc_protocol::CustomMetalKernelReport {
        probe_verdicts: vec![ipc_protocol::CustomMetalKernelProbeVerdict {
            family: "gated_delta_sequence".to_owned(),
            outcome: ipc_protocol::CustomMetalKernelProbeOutcome::Passed,
        }],
        in_use: vec!["gated_delta_sequence".to_owned()],
    };
    let load_result = EngineLoadResult::new()
        .with_draft_model_status(DraftModelStatus::None)
        .with_custom_metal_kernel_report(custom_metal_kernels.clone());
    let engine_status_events = load_result.engine_status_events();
    assert_eq!(
        engine_status_events.len(),
        2,
        "the prefill setting is off, so it announces nothing"
    );
    assert!(matches!(
        engine_status_events[0],
        WorkerEvent::DraftModelChanged { .. }
    ));
    assert_eq!(
        engine_status_events[1],
        WorkerEvent::CustomMetalKernelsChanged {
            custom_metal_kernels
        }
    );
}
