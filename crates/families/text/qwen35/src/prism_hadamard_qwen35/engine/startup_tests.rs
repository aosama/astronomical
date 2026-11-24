use ipc_protocol::WorkerChunkingConfiguration;

use super::{
    PrismHadamardQwen35ServingSettings, pending_load_for, prompt_processing_chunk_sizer_for,
};
use crate::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    remove_fixture_directory, write_fixture_pack,
};
use crate::prism_hadamard_qwen35::astro_decoder_state_cache_layout::PRISM_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS;
use crate::prism_hadamard_qwen35::engine::{
    PrismHadamardQwen35InferenceExecution, PrismHadamardQwen35PendingStartup,
};
use crate::{PerformanceAttribution, PerformanceAttributionLog, Qwen3_5ExecutionCoreSettings};

const ONE_GB: usize = 1_073_741_824;

/// The serving settings the worker hands over, with a chunking
/// configuration whose growth step differs from Prism's own.
fn serving_settings_with(
    prompt_processing_chunk_tokens: u32,
    ssd_prompt_cache_common_prefix_stride_blocks: u32,
) -> PrismHadamardQwen35ServingSettings {
    PrismHadamardQwen35ServingSettings {
        maximum_context_tokens: Some(2048),
        maximum_output_tokens: Some(256),
        prompt_processing_chunk_tokens,
        enable_thinking: true,
        chunking: WorkerChunkingConfiguration {
            experimental_neural_engine_prefill_enabled: true,
            ssd_prompt_cache_common_prefix_stride_blocks,
            ssd_prompt_cache_block_tokens: Some(128),
            full_attention_key_value_growth_tokens: 1024,
            fixed_prompt_processing_chunk_size_tokens: prompt_processing_chunk_tokens,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 64,
            prefill_graph_submission_layer_interval: 0,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
            experimental_decode_stage_attribution_enabled: false,
            experimental_prefill_stage_attribution_enabled: false,
            experimental_in_memory_prefix_adoption_enabled: false,
        },
        astro_ssd_prompt_cache_enabled: false,
        astro_ssd_prompt_cache_config: None,
        performance_attribution_log_path: None,
    }
}

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
    let validated_pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let serving_settings = serving_settings_with(512, 1);
    let validated_execution_core_settings = Qwen3_5ExecutionCoreSettings {
        validated_artifact: validated_pack,
        active_memory_limit_bytes: 8 * ONE_GB,
        mlx_buffer_cache_limit_bytes: ONE_GB,
        astro_ssd_prompt_cache_disk_store_config: None,
        prompt_processing_chunk_sizer: prompt_processing_chunk_sizer_for(&serving_settings)
            .expect("the chunk size is positive"),
        think_end_token_id: 248_069,
        model_directory: fixture_directory.clone(),
        maximum_context_tokens: 2048,
        chunking: serving_settings.chunking.clone(),
        model_loading_performance_attribution: PerformanceAttribution::disabled(),
        performance_attribution_log: PerformanceAttributionLog::disabled(),
    }
    .validate()
    .expect("the fixture settings should validate");

    let execution =
        PrismHadamardQwen35InferenceExecution::pending(PrismHadamardQwen35PendingStartup {
            validated_execution_core_settings,
            pending_load: pending_load_for(&serving_settings, 8 * ONE_GB, ONE_GB),
        });

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
