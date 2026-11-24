use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::starting_request_decoder_state;
use crate::PerformanceAttribution;
use crate::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    remove_fixture_directory, write_fixture_pack,
};
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use engine_core::astro_ssd_prompt_cache::AstroSsdPromptCacheCounters;

// Positive and edge: with no store open, a request starts from position zero
// with one unallocated decoder-state owner per fixture decoder layer.
#[test]
#[serial]
fn should_start_a_request_from_position_zero_when_no_store_is_open() {
    let fixture_directory = write_fixture_pack(|_| {});
    let fixture_runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the fixture memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let validated_pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let fixture_model = PrismHadamardTextModel::new(fixture_runtime, validated_pack)
        .expect("the fixture pack should load");
    let mut astro_ssd_prompt_cache_counters = AstroSsdPromptCacheCounters::default();

    let (request_decoder_state, restore_outcome) = starting_request_decoder_state(
        &fixture_model,
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
        fixture_model.weights.decoder_layers.len()
    );
    assert_eq!(request_decoder_state.payload_byte_count(), 0);
    remove_fixture_directory(&fixture_directory);
}
