use serial_test::serial;

use super::starting_request_decoder_state;
use crate::PerformanceAttribution;
use crate::prism_hadamard_qwen35::model::fixture_model_test_support::load_fixture_text_model;
use engine_core::astro_ssd_prompt_cache::AstroSsdPromptCacheCounters;

// Positive and edge: with no store open, a request starts from position zero
// with one unallocated decoder-state owner per fixture decoder layer.
#[test]
#[serial]
fn should_start_a_request_from_position_zero_when_no_store_is_open() {
    let fixture = load_fixture_text_model();
    let mut astro_ssd_prompt_cache_counters = AstroSsdPromptCacheCounters::default();

    let (request_decoder_state, restore_outcome) = starting_request_decoder_state(
        &fixture.text_model,
        None,
        &[1, 2, 3],
        &mut astro_ssd_prompt_cache_counters,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("a request without a store should start");

    assert_eq!(restore_outcome.restored_token_count, 0);
    assert!(
        restore_outcome
            .last_restored_astro_ssd_prompt_cache_block_key
            .is_none()
    );
    assert_eq!(
        request_decoder_state.layer_count(),
        fixture.text_model.weights.decoder_layers.len()
    );
    assert_eq!(request_decoder_state.payload_byte_count(), 0);
}
