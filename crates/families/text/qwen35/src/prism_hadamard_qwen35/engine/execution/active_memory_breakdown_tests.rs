use serial_test::serial;

use super::prism_active_memory_breakdown;
use crate::prism_hadamard_qwen35::model::fixture_model_test_support::load_fixture_text_model;
use crate::qwen3_5_core::model::memory_breakdown::Qwen3_5MemoryOwnership;

/// Active-memory readings around the fixture's few-megabyte payload: none at
/// all and one byte (the reconciliation clamps every owner), about the
/// weights alone, and far more than every owner together.
const ACTIVE_MEMORY_READINGS_BYTES: [u64; 4] = [0, 1, 4 * 1024 * 1024, 1_073_741_824];

#[test]
#[serial]
fn should_compose_the_engine_breakdown_through_the_shared_memory_ownership() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    let mut request_decoder_state = text_model
        .empty_request_decoder_state()
        .expect("the fixture layout should create decoder state");
    let prompt_tokens = text_model
        .runtime
        .array_from_i32(&[1, 2, 3], &[1, 3])
        .expect("the prompt tokens should build");
    let logits = text_model
        .forward_pass(&prompt_tokens, 0, &mut request_decoder_state)
        .expect("the prompt forward should build");
    text_model
        .evaluate_outputs(&[&logits])
        .expect("the prompt forward should evaluate");
    assert!(
        request_decoder_state.payload_byte_count() > 0,
        "the forward must leave live decoder state to charge"
    );
    let measured_active_memory_bytes = u64::try_from(
        text_model
            .runtime
            .memory_snapshot()
            .expect("the MLX memory reading should succeed")
            .active_memory_bytes(),
    )
    .expect("active memory fits u64");

    for active_memory_bytes in ACTIVE_MEMORY_READINGS_BYTES
        .into_iter()
        .chain([measured_active_memory_bytes])
    {
        let engine_breakdown = prism_active_memory_breakdown(
            text_model,
            active_memory_bytes,
            Some(&request_decoder_state),
        );
        let shared_breakdown =
            text_model.active_memory_breakdown(&request_decoder_state, 0, active_memory_bytes, 0);
        assert_eq!(
            shared_breakdown, engine_breakdown,
            "the breakdowns differ at {active_memory_bytes} active bytes"
        );
        assert_eq!(shared_breakdown.expert_payload_bytes, 0);
    }
}

#[test]
#[serial]
fn should_charge_no_context_state_between_requests() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    let idle_request_decoder_state = text_model
        .empty_request_decoder_state()
        .expect("the fixture layout should create decoder state");
    let engine_breakdown = prism_active_memory_breakdown(text_model, 1_073_741_824, None);
    assert_eq!(engine_breakdown.context_state_payload_bytes, 0);
    assert_eq!(
        text_model.active_memory_breakdown(&idle_request_decoder_state, 0, 1_073_741_824, 0),
        engine_breakdown
    );
    assert_eq!(
        text_model.finalized_active_memory_breakdown(1_073_741_824, 0, 0),
        engine_breakdown
    );
}
