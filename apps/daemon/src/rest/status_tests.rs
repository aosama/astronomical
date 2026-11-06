//! Beside-source coverage for the pure status-label units. The JSON
//! attachment units are thin field mappings exercised by the REST
//! integration journeys.

use super::{expert_memory_mode_label, state_directory_label};
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::ExpertMemoryMode;

#[test]
fn should_label_the_standard_stable_state_directory() {
    assert_eq!(
        state_directory_label(true, AstronomicalRuntimeInstance::Stable),
        "~/.astronomical"
    );
}

#[test]
fn should_label_the_standard_development_state_directory() {
    assert_eq!(
        state_directory_label(true, AstronomicalRuntimeInstance::Development),
        "~/.astronomical-dev"
    );
}

#[test]
fn should_label_a_custom_directory_regardless_of_channel() {
    assert_eq!(
        state_directory_label(false, AstronomicalRuntimeInstance::Stable),
        "custom"
    );
    assert_eq!(
        state_directory_label(false, AstronomicalRuntimeInstance::Development),
        "custom"
    );
}

#[test]
fn should_label_every_expert_memory_mode() {
    assert_eq!(
        expert_memory_mode_label(ExpertMemoryMode::Resident),
        "resident"
    );
    assert_eq!(expert_memory_mode_label(ExpertMemoryMode::Hybrid), "hybrid");
    assert_eq!(expert_memory_mode_label(ExpertMemoryMode::Paged), "paged");
}

#[test]
fn should_report_neural_engine_totals_under_their_public_names() {
    let mut totals =
        crate::serving::neural_engine_session_totals::NeuralEngineSessionTotals::empty();
    totals.note_prefill_status(Some(
        &astronomical_ipc_protocol::NeuralEnginePrefillStatus::Active {
            layer_program_count: 28,
            neural_engine_tokens_per_chunk: 960,
            chunk_token_count: 2_048,
            measured_memory_bytes: 1,
            measured_speed_percent: Some(128),
        },
    ));
    totals.record_chunk(astronomical_ipc_protocol::NeuralEngineChunkReport {
        placement: astronomical_ipc_protocol::NeuralEngineChunkPlacement::BothEngines,
        chunk_token_count: 2_048,
        neural_engine_token_count: 960,
        elapsed_millis: 1_000,
        is_prompt_warm: true,
    });

    assert_eq!(
        super::neural_engine_session_facts(&totals),
        serde_json::json!({
            "prompt_token_count": 2_048,
            "neural_engine_token_count": 960,
            "estimated_saved_ms": 280,
            "last_chunk_placement": "both_engines",
        })
    );
}
