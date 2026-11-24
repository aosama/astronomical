use ipc_protocol::ExpertMemoryMode;
use serial_test::file_serial;

use super::super::fixture_model_test_support::load_fixture_text_model;
use super::super::{PrismHadamardTextModel, request_decoder_state_evaluation_arrays};
use crate::PerformanceAttribution;
use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::memory::telemetry::MlxActiveMemoryBreakdown;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;
use crate::qwen3_5_core::shared_execution::text_model::Qwen3_5TextModel;

/// The fixture prompt: two tokens read as a prompt chunk, then one more
/// forwarded for its logits, as the shared loop does for injected feedback.
const PROMPT_CHUNK_TOKEN_IDS: [u32; 2] = [1, 2];
const FORWARDED_TOKEN_ID: u32 = 3;

#[test]
#[file_serial]
fn should_produce_the_same_logits_as_the_prism_forward() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;

    let mut shared_request_decoder_state = text_model
        .empty_request_decoder_state()
        .expect("the fixture layout should create decoder state");
    text_model
        .prefill_chunk_with_performance_attribution(
            &PROMPT_CHUNK_TOKEN_IDS,
            0,
            &mut shared_request_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the shared prefill chunk should run");
    let shared_logits = text_model
        .build_forward_chunk_with_performance_attribution(
            &[FORWARDED_TOKEN_ID],
            PROMPT_CHUNK_TOKEN_IDS.len() as u32,
            &mut shared_request_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the shared forward chunk should build");

    let prism_logits = prism_reference_logits(text_model);
    let shared_values = float32_values(text_model, &shared_logits);
    let prism_values = float32_values(text_model, &prism_logits);
    assert_eq!(shared_values.len(), prism_values.len());
    for (logit_index, (shared_value, prism_value)) in
        shared_values.iter().zip(&prism_values).enumerate()
    {
        assert_eq!(
            shared_value.to_bits(),
            prism_value.to_bits(),
            "logit {logit_index} differs between the shared and Prism routes"
        );
    }
}

#[test]
#[file_serial]
fn should_submit_the_generated_token_with_every_live_state_root() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    let mut request_decoder_state = text_model
        .empty_request_decoder_state()
        .expect("the fixture layout should create decoder state");
    let logits = text_model
        .build_forward_chunk_with_performance_attribution(
            &PROMPT_CHUNK_TOKEN_IDS,
            0,
            &mut request_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the forward chunk should build");
    let generated_token = text_model
        .greedy_token_lazily(&logits)
        .expect("the greedy token should build");

    text_model
        .async_evaluate_generation(&generated_token, &request_decoder_state)
        .expect("the generation should submit");

    // Every root the submission named is evaluated once its token is read.
    generated_token
        .item_u32()
        .expect("the submitted token should evaluate");
    let evaluation_roots =
        request_decoder_state_evaluation_arrays(&request_decoder_state, &generated_token);
    assert_eq!(evaluation_roots.len(), 9);
}

#[test]
#[file_serial]
fn should_report_a_fully_resident_dense_model_without_a_governor() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    let pack_payload_bytes = text_model.pack().total_payload_bytes();

    assert_eq!(text_model.expert_memory_mode(), ExpertMemoryMode::Resident);
    assert_eq!(
        text_model.observed_expert_memory_mode(),
        ExpertMemoryMode::Resident
    );
    assert_eq!(
        text_model.expert_residency_telemetry_for_breakdown(&MlxActiveMemoryBreakdown::default()),
        ExpertResidencyTelemetry::default()
    );
    assert!(text_model.latest_memory_ledger_reading().is_none());
    assert!(text_model.governed_memory_reserve_snapshot().is_none());
    assert_eq!(text_model.held_expert_payload_bytes(), 0);
    assert_eq!(
        text_model.held_model_core_payload_bytes(),
        pack_payload_bytes
    );
    assert_eq!(
        text_model.resident_model_payload_byte_count(),
        pack_payload_bytes
    );
    assert_eq!(
        text_model
            .minimum_mlx_memory_ceiling_bytes()
            .expect("the floor should compute"),
        crate::memory::dense_resident_safe_minimum_mlx_memory_ceiling_bytes(pack_payload_bytes)
    );
    let configured_limit_bytes = text_model
        .runtime
        .configured_memory_limit_bytes()
        .expect("the MLX limit should read");
    assert_eq!(
        text_model.mlx_active_memory_ceiling_bytes(),
        configured_limit_bytes as u64
    );
    assert!(text_model.memory_snapshot_for_telemetry().is_ok());
}

/// The same prompt through Prism's own forward, as its engine runs it.
fn prism_reference_logits(text_model: &PrismHadamardTextModel) -> mlx_bridge::MlxArray {
    let mut prism_request_decoder_state = text_model
        .empty_request_decoder_state()
        .expect("the fixture layout should create decoder state");
    let prompt_tokens = text_model
        .runtime
        .array_from_i32(&[1, 2], &[1, 2])
        .expect("the prompt tokens should build");
    let final_hidden_state_row = text_model
        .forward_hidden_states(&prompt_tokens, 0, &mut prism_request_decoder_state)
        .expect("the Prism prompt forward should build");
    text_model
        .evaluate_outputs(&[&final_hidden_state_row])
        .expect("the Prism prompt forward should evaluate");
    let forwarded_token = text_model
        .runtime
        .array_from_i32(&[3], &[1, 1])
        .expect("the forwarded token should build");
    text_model
        .forward_pass(&forwarded_token, 2, &mut prism_request_decoder_state)
        .expect("the Prism forward should build")
}

/// The logits widened to float32 without rounding, so their bits compare.
fn float32_values(text_model: &PrismHadamardTextModel, logits: &mlx_bridge::MlxArray) -> Vec<f32> {
    text_model
        .runtime
        .astype(logits, mlx_bridge::MlxDtype::Float32)
        .and_then(|float32_logits| float32_logits.to_vec_f32())
        .expect("the logits should evaluate")
}
