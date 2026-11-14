//! Beside-source coverage for the pure status-label units. The JSON
//! attachment units are thin field mappings exercised by the REST
//! integration journeys.

use super::{expert_memory_mode_label, prompt_progress_elapsed_millis};
use astronomical_ipc_protocol::ExpertMemoryMode;

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

#[test]
fn should_report_the_idle_unload_span_with_no_countdown_before_the_clock_runs() {
    let worker_handle = crate::WorkerHandle::unavailable();
    worker_handle.set_idle_model_unload_seconds(60);
    let mut status_json = serde_json::json!({});

    super::attach_idle_model_unload(&mut status_json, &worker_handle);

    assert_eq!(
        status_json["idle_model_unload"],
        serde_json::json!({ "after_seconds": 60, "remaining_seconds": null })
    );
}

#[test]
fn should_report_a_zero_span_as_staying_loaded() {
    let worker_handle = crate::WorkerHandle::unavailable();
    let mut status_json = serde_json::json!({});

    super::attach_idle_model_unload(&mut status_json, &worker_handle);

    assert_eq!(status_json["idle_model_unload"]["after_seconds"], 0);
    assert!(status_json["idle_model_unload"]["remaining_seconds"].is_null());
}

#[test]
fn should_run_the_prompt_clock_live_while_prompt_tokens_remain() {
    let request_elapsed = std::time::Duration::from_millis(9_000);
    assert_eq!(
        prompt_progress_elapsed_millis(4_096, 11_125, 2_000, request_elapsed),
        9_000
    );
}

#[test]
fn should_stop_the_prompt_clock_at_the_measured_time_once_every_prompt_token_is_processed() {
    let request_elapsed = std::time::Duration::from_millis(10_000);
    assert_eq!(
        prompt_progress_elapsed_millis(20, 20, 1_158, request_elapsed),
        1_158
    );
}

#[test]
fn should_never_report_less_than_the_measured_prompt_time() {
    let request_elapsed = std::time::Duration::from_millis(500);
    assert_eq!(
        prompt_progress_elapsed_millis(10, 20, 1_000, request_elapsed),
        1_000
    );
}

#[test]
fn should_stop_the_prompt_clock_for_an_empty_prompt() {
    let request_elapsed = std::time::Duration::from_millis(700);
    assert_eq!(prompt_progress_elapsed_millis(0, 0, 0, request_elapsed), 0);
}
