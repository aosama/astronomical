//! The folded-ternary engine built over the sealed fixture pack, for tests
//! that drive startup, load and whole requests without a real model.

use std::path::{Path, PathBuf};

use ipc_protocol::WorkerChunkingConfiguration;

use super::startup::{pending_load_for, prompt_processing_chunk_sizer_for};
use super::{
    PrismHadamardQwen35InferenceExecution, PrismHadamardQwen35PendingStartup,
    PrismHadamardQwen35ServingSettings,
};
use crate::Qwen3_5ExecutionCoreSettings;
use crate::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    FIXTURE_VOCAB_SIZE, FixturePackMutation, remove_fixture_directory, write_fixture_pack,
};
use crate::{MlxInferenceExecution, PerformanceAttribution, PerformanceAttributionLog};

/// The fixture's think-end marker: the last vocabulary id, so a thinking
/// request's forced transition can end with it inside the vocabulary.
pub(super) const FIXTURE_THINK_END_TOKEN_ID: u32 = FIXTURE_VOCAB_SIZE - 1;

/// The serving settings the worker hands over, with a chunking
/// configuration whose growth step differs from Prism's own.
pub(super) fn serving_settings_with(
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

/// The MLX limits a fixture engine runs under: the fixture's few megabytes
/// of weights fit with room to spare, and every fixture test in the process
/// installs these same limits.
pub(super) struct FixtureMemoryLimits {
    pub(super) mlx_memory_ceiling_bytes: usize,
    pub(super) mlx_buffer_cache_bytes: usize,
}

/// An engine over the fixture pack at `fixture_directory`, validated and
/// built exactly as startup builds it, with no model loaded yet.
pub(super) fn pending_fixture_execution(
    fixture_directory: &Path,
    serving_settings: &PrismHadamardQwen35ServingSettings,
    fixture_memory_limits: &FixtureMemoryLimits,
) -> PrismHadamardQwen35InferenceExecution {
    let validated_pack = FoldedPackValidator::new()
        .validate(fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let validated_execution_core_settings = Qwen3_5ExecutionCoreSettings {
        validated_artifact: validated_pack,
        active_memory_limit_bytes: fixture_memory_limits.mlx_memory_ceiling_bytes,
        mlx_buffer_cache_limit_bytes: fixture_memory_limits.mlx_buffer_cache_bytes,
        astro_ssd_prompt_cache_disk_store_config: None,
        prompt_processing_chunk_sizer: prompt_processing_chunk_sizer_for(serving_settings)
            .expect("the chunk size is positive"),
        think_end_token_id: FIXTURE_THINK_END_TOKEN_ID,
        model_directory: fixture_directory.to_path_buf(),
        maximum_context_tokens: 2048,
        chunking: serving_settings.chunking.clone(),
        model_loading_performance_attribution: PerformanceAttribution::disabled(),
        performance_attribution_log: PerformanceAttributionLog::disabled(),
    }
    .validate()
    .expect("the fixture settings should validate");
    PrismHadamardQwen35InferenceExecution::pending(PrismHadamardQwen35PendingStartup {
        validated_execution_core_settings,
        pending_load: pending_load_for(
            serving_settings,
            fixture_memory_limits.mlx_memory_ceiling_bytes,
            fixture_memory_limits.mlx_buffer_cache_bytes,
        ),
    })
}

/// A loaded fixture engine that removes its pack directory when dropped.
pub(super) struct LoadedFixtureEngine {
    pub(super) execution: PrismHadamardQwen35InferenceExecution,
    fixture_directory: PathBuf,
}

impl Drop for LoadedFixtureEngine {
    fn drop(&mut self) {
        remove_fixture_directory(&self.fixture_directory);
    }
}

/// Builds and loads the engine over a fixture pack `pack_mutation` shapes.
pub(super) fn loaded_fixture_engine(
    pack_mutation: impl FnOnce(&mut FixturePackMutation),
) -> LoadedFixtureEngine {
    let fixture_directory = write_fixture_pack(pack_mutation);
    let fixture_memory_limits = FixtureMemoryLimits {
        mlx_memory_ceiling_bytes: 512 * 1024 * 1024,
        mlx_buffer_cache_bytes: 512 * 1024 * 1024,
    };
    let mut execution = pending_fixture_execution(
        &fixture_directory,
        &serving_settings_with(64, 1),
        &fixture_memory_limits,
    );
    execution.load().expect("the fixture engine should load");
    LoadedFixtureEngine {
        execution,
        fixture_directory,
    }
}
