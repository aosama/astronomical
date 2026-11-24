use super::{pending_load_for, prompt_processing_chunk_sizer_for};
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    remove_fixture_directory, write_fixture_pack,
};
use crate::prism_hadamard_qwen35::astro_decoder_state_cache_layout::PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS;
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    FixtureMemoryLimits, pending_fixture_execution, serving_settings_with,
};

const ONE_GB: usize = 1_073_741_824;

#[test]
fn should_carry_the_load_knobs_from_the_chunking_settings() {
    let pending_load = pending_load_for(&serving_settings_with(512, 3), 8 * ONE_GB, ONE_GB);
    assert_eq!(pending_load.effective_mlx_memory_ceiling_bytes, 8 * ONE_GB);
    assert_eq!(pending_load.mlx_buffer_cache_limit_bytes, ONE_GB);
    assert_eq!(
        pending_load.configured_astro_ssd_prompt_cache_block_token_count,
        Some(128)
    );
    assert_eq!(pending_load.ssd_prompt_cache_common_prefix_stride_blocks, 3);
    assert!(pending_load.neural_engine_mlp_prefill_requested);
}

#[test]
fn should_floor_the_stride_and_the_prefill_chunk_at_one() {
    let serving_settings = serving_settings_with(0, 0);
    let pending_load = pending_load_for(&serving_settings, 8 * ONE_GB, ONE_GB);
    assert_eq!(pending_load.ssd_prompt_cache_common_prefix_stride_blocks, 1);
    let prompt_processing_chunk_sizer = prompt_processing_chunk_sizer_for(&serving_settings)
        .expect("a zero chunk is floored at one token, never refused");
    assert_eq!(
        prompt_processing_chunk_sizer.maximum_prompt_processing_chunk_size_tokens(),
        1
    );
}

#[test]
fn should_keep_prism_limits_where_the_shared_settings_derive_others() {
    let fixture_directory = write_fixture_pack(|_| {});
    let execution = pending_fixture_execution(
        &fixture_directory,
        &serving_settings_with(512, 1),
        &FixtureMemoryLimits {
            mlx_memory_ceiling_bytes: 8 * ONE_GB,
            mlx_buffer_cache_bytes: ONE_GB,
        },
    );

    let execution_core = &execution.execution_core;
    // The configured window is Prism's hard limit, not the pack's 4096.
    assert_eq!(execution_core.maximum_position_count, 2048);
    assert_eq!(execution_core.hard_maximum_position_count, 2048);
    assert_eq!(execution_core.context_memory_reservation_bytes_per_token, 0);
    assert_eq!(
        execution_core.full_attention_kv_state_growth_tokens,
        PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS as i32
    );
    assert_eq!(execution.maximum_context_tokens(), 2048);
    assert_eq!(execution.prompt_processing_chunk_tokens(), 512);
    assert_eq!(execution.configured_mlx_buffer_cache_bytes, ONE_GB as u64);
    assert!(execution_core.model.is_none() && execution_core.active_request.is_none());
    remove_fixture_directory(&fixture_directory);
}
