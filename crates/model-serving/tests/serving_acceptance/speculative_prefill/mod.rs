mod fail_closed;
mod large_prompt_activation_overrun;
mod memory_admission;
mod persistent_cache;
mod representative_generation;
pub(crate) mod support;
mod tool_control;
mod tool_process_prompt;
mod tool_process_restart;
mod vanished_target_state;
mod visual_tool;

use std::path::Path;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::PersistentPromptCacheDiskStoreConfig;
use representative_generation::{
    build_loaded_representative_engine, generate_representative_measurement,
};
pub(crate) use support::prepare_representative_prompt;

const SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS: u32 = 8_192;
pub(super) const SPECULATIVE_PREFILL_KEEP_PERCENTAGE: u32 = 20;
pub(super) const SPECULATIVE_PREFILL_SELECTION_CHUNK_TOKEN_COUNT: u32 = 32;
pub(super) const SPECULATIVE_PREFILL_MANDATORY_TRAILING_TOKEN_COUNT: u32 = 512;
pub(super) const SPECULATIVE_PREFILL_LOOKAHEAD_TOKEN_COUNT: u32 = 8;
pub(super) const SPECULATIVE_PREFILL_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT: u32 = 13;
pub(crate) struct RepresentativePrompt {
    pub(crate) prompt_token_ids: Vec<u32>,
    pub(crate) image_pad_token_id: u32,
    pub(crate) processed_visual_images: Vec<astronomical_model_serving::Qwen3_5ProcessedImage>,
    pub(crate) ordinary_target_prefill_control_span_token_count: usize,
    pub(crate) sampling_temperature_thousandths: u16,
    pub(crate) sampling_top_p_thousandths: u16,
    pub(crate) sampling_seed: Option<u64>,
}

pub(super) struct RepresentativeGenerationMeasurement {
    pub(super) generated_token_ids: Vec<u32>,
    pub(super) speculative_prefill_draft_scoring_elapsed_seconds: f64,
    pub(super) speculative_prefill_fallback_count: u64,
    pub(super) speculative_prefill_draft_persistent_prefix_restored_token_count: u64,
    pub(super) speculative_prefill_drafter_eligible_token_count: u64,
    pub(super) restored_target_persistent_prompt_cache_token_count: u64,
    pub(super) speculative_prefill_target_persistent_state_write_count: u64,
    pub(super) speculative_prefill_target_persistent_state_restored_token_count: u64,
    pub(super) speculative_prefill_draft_scored_suffix_token_count: u64,
    pub(super) speculative_prefill_ordinary_control_span_token_count: u64,
    pub(super) speculative_prefill_selected_token_count: u64,
    pub(super) speculative_prefill_selected_token_positions: Vec<usize>,
    pub(super) speculative_prefill_mandatory_visual_token_count: u64,
    pub(super) speculative_prefill_context_target_expert_reclaimed_payload_bytes: u64,
    pub(super) speculative_prefill_draft_target_expert_reclaimed_payload_bytes: u64,
    pub(super) speculative_prefill_target_expert_repopulated_payload_bytes: u64,
    pub(super) speculative_prefill_request_scoped_draft_release_elapsed_seconds: f64,
}

pub(super) async fn run_representative_generation(
    target_model_directory: &Path,
    draft_model_directory: &Path,
    draft_model_id: &str,
    representative_prompt: &RepresentativePrompt,
    speculative_prefill_enabled: bool,
    maximum_output_token_count: u16,
    speculative_prefill_keep_percentage: u32,
    request_id: RequestId,
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    mlx_memory_limits: astronomical_runtime_integration::MlxMemoryLimits,
) -> RepresentativeGenerationMeasurement {
    run_representative_generation_with_selection_chunk_token_count(
        target_model_directory,
        draft_model_directory,
        draft_model_id,
        representative_prompt,
        speculative_prefill_enabled,
        maximum_output_token_count,
        speculative_prefill_keep_percentage,
        SPECULATIVE_PREFILL_SELECTION_CHUNK_TOKEN_COUNT,
        request_id,
        persistent_prompt_cache_disk_store_config,
        mlx_memory_limits,
    )
    .await
}

pub(super) async fn run_representative_generation_with_selection_chunk_token_count(
    target_model_directory: &Path,
    draft_model_directory: &Path,
    draft_model_id: &str,
    representative_prompt: &RepresentativePrompt,
    speculative_prefill_enabled: bool,
    maximum_output_token_count: u16,
    speculative_prefill_keep_percentage: u32,
    speculative_prefill_selection_chunk_token_count: u32,
    request_id: RequestId,
    persistent_prompt_cache_disk_store_config: Option<PersistentPromptCacheDiskStoreConfig>,
    mlx_memory_limits: astronomical_runtime_integration::MlxMemoryLimits,
) -> RepresentativeGenerationMeasurement {
    let mut loaded_representative_engine = build_loaded_representative_engine(
        target_model_directory,
        draft_model_directory,
        draft_model_id,
        speculative_prefill_enabled,
        speculative_prefill_keep_percentage,
        speculative_prefill_selection_chunk_token_count,
        persistent_prompt_cache_disk_store_config,
        mlx_memory_limits,
    )
    .await;
    generate_representative_measurement(
        &mut loaded_representative_engine,
        representative_prompt,
        maximum_output_token_count,
        request_id,
    )
    .await
}
