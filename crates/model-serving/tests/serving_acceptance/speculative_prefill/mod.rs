mod fail_closed;
mod memory_admission;
mod persistent_cache;
pub(crate) mod support;
mod tool_control;
mod tool_process_prompt;
mod tool_process_restart;
mod visual_tool;

use std::path::Path;

use astronomical_ipc_protocol::{RequestId, WorkerSpeculativePrefillConfiguration};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
pub(crate) use support::prepare_representative_prompt;

const SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS: u32 = 8_192;
pub(super) const SPECULATIVE_PREFILL_KEEP_PERCENTAGE: u32 = 20;
const SPECULATIVE_PREFILL_SELECTION_CHUNCK_TOKEN_COUNT: u32 = 32;
const SPECULATIVE_PREFILL_MANDATORY_TRAILING_TOKEN_COUNT: u32 = 512;
const SPECULATIVE_PREFILL_LOOKAHEAD_TOKEN_COUNT: u32 = 8;
const SPECULATIVE_PREFILL_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT: u32 = 13;
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
        SPECULATIVE_PREFILL_SELECTION_CHUNCK_TOKEN_COUNT,
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
    let performance_attribution_directory = tempfile::tempdir()
        .expect("the representative measurement should create an attribution directory");
    let performance_attribution_log_path = performance_attribution_directory
        .path()
        .join("performance-attribution.jsonl");
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(target_model_directory, 20_480)
        .expect("the representative target artifact should validate before loading");
    let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the representative target tokenizer should load before engine construction");
    let target_think_end_token_id = target_tokenizer.think_end_token_id();
    let speculative_prefill = WorkerSpeculativePrefillConfiguration {
        enabled: speculative_prefill_enabled,
        target_model_id: speculative_prefill_enabled
            .then(|| validated_target_artifact.model_id().to_owned()),
        draft_model_id: speculative_prefill_enabled.then(|| draft_model_id.to_owned()),
        draft_model_directory: speculative_prefill_enabled
            .then(|| draft_model_directory.to_path_buf()),
        minimum_prompt_tokens: SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS,
        keep_percentage: speculative_prefill_keep_percentage,
        selection_chunk_token_count: speculative_prefill_selection_chunk_token_count,
        mandatory_trailing_token_count: SPECULATIVE_PREFILL_MANDATORY_TRAILING_TOKEN_COUNT,
        lookahead_token_count: SPECULATIVE_PREFILL_LOOKAHEAD_TOKEN_COUNT,
        importance_pooling_kernel_token_count:
            SPECULATIVE_PREFILL_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT,
    };
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_target_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        persistent_prompt_cache_disk_store_config,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(32)
            .expect("the representative prefill chunk size should be valid"),
        target_think_end_token_id,
        target_model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        true,
        false,
        speculative_prefill,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the representative attribution log should open"),
    )
    .expect("the representative SpecPrefill engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the representative target and optional draft should load");
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                representative_prompt.prompt_token_ids.clone(),
                maximum_output_token_count,
                representative_prompt.sampling_temperature_thousandths,
                representative_prompt.sampling_top_p_thousandths,
                representative_prompt.sampling_seed,
            )
            .with_image_pad_token_id(representative_prompt.image_pad_token_id)
            .with_processed_visual_images(representative_prompt.processed_visual_images.clone())
            .with_ordinary_target_prefill_control_span_token_count(
                representative_prompt.ordinary_target_prefill_control_span_token_count,
            )
            .with_thinking_configuration(false, None, Vec::new(), Vec::new())
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the representative prompt should be admitted");
    let mut observed_prompt_work_reuse = qwen3_5_engine
        .prompt_work_reuse_for_tests(request_id)
        .await
        .expect("the representative prompt work should remain inspectable after admission");
    let mut generated_token_ids = Vec::with_capacity(usize::from(maximum_output_token_count));
    let mut speculative_prefill_selected_token_positions = Vec::new();
    loop {
        let generated_token = qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .unwrap_or_else(|generation_error| {
                panic!(
                    "the representative request should continue generating after {} output tokens: {generation_error}",
                    generated_token_ids.len(),
                )
            });
        let generation_has_finalized = matches!(
            &generated_token,
            GeneratedToken::TokenId {
                generation_finalization: Some(_),
                ..
            } | GeneratedToken::EndOfSequence
        );
        if matches!(&generated_token, GeneratedToken::PrefillProgress { .. })
            && let Some(selected_token_positions) = qwen3_5_engine
                .speculative_prefill_selected_token_positions_for_tests(request_id)
                .await
                .expect("the representative request selected positions should remain inspectable")
        {
            speculative_prefill_selected_token_positions = selected_token_positions;
        }
        if let GeneratedToken::PrefillProgress {
            prompt_work_reuse, ..
        } = &generated_token
        {
            observed_prompt_work_reuse = prompt_work_reuse.clone();
        }
        match &generated_token {
            GeneratedToken::TokenId { token_id, .. } => {
                generated_token_ids.push(*token_id);
            }
            GeneratedToken::PrefillProgress { .. }
            | GeneratedToken::PromptProcessingPhaseStarted { .. }
            | GeneratedToken::GenerationPreparationStarted { .. }
            | GeneratedToken::EndOfSequence => {}
        }
        if generated_token_ids.len() >= usize::from(maximum_output_token_count)
            || generation_has_finalized
        {
            break;
        }
    }
    let attribution_report_documents =
        crate::serving_acceptance::support::performance_attribution::read_attribution_report_documents(
            &performance_attribution_log_path,
        );
    let generation_report =
        crate::serving_acceptance::support::performance_attribution::generation_report_for_request(
            &attribution_report_documents,
            request_id.value(),
        );
    let operation_elapsed_seconds = |operation_identifier| {
        crate::serving_acceptance::support::performance_attribution::operation_total_elapsed_nanoseconds(
            generation_report,
            operation_identifier,
        ) as f64
            / 1_000_000_000.0
    };
    let performance_counter_amount = |counter_identifier| {
        crate::serving_acceptance::support::performance_attribution::counter_amount(
            generation_report,
            counter_identifier,
        )
    };
    RepresentativeGenerationMeasurement {
        generated_token_ids,
        speculative_prefill_draft_scoring_elapsed_seconds: operation_elapsed_seconds(
            "speculative_prefill_draft_scoring",
        ),
        speculative_prefill_fallback_count: performance_counter_amount(
            "speculative_prefill_fallback_count",
        ),
        speculative_prefill_draft_persistent_prefix_restored_token_count:
            performance_counter_amount(
                "speculative_prefill_draft_persistent_prefix_restored_token_count",
            ),
        speculative_prefill_drafter_eligible_token_count: observed_prompt_work_reuse
            .drafter_eligible_token_count,
        restored_target_persistent_prompt_cache_token_count: performance_counter_amount(
            "restored_persistent_prompt_cache_token_count",
        ),
        speculative_prefill_target_persistent_state_write_count: performance_counter_amount(
            "speculative_prefill_target_persistent_state_write_count",
        ),
        speculative_prefill_target_persistent_state_restored_token_count:
            performance_counter_amount(
                "speculative_prefill_target_persistent_state_restored_token_count",
            ),
        speculative_prefill_draft_scored_suffix_token_count: performance_counter_amount(
            "speculative_prefill_draft_scored_suffix_token_count",
        ),
        speculative_prefill_ordinary_control_span_token_count: performance_counter_amount(
            "speculative_prefill_ordinary_control_span_token_count",
        ),
        speculative_prefill_selected_token_count: performance_counter_amount(
            "speculative_prefill_selected_token_count",
        ),
        speculative_prefill_selected_token_positions,
        speculative_prefill_mandatory_visual_token_count: performance_counter_amount(
            "speculative_prefill_mandatory_visual_token_count",
        ),
        speculative_prefill_context_target_expert_reclaimed_payload_bytes:
            performance_counter_amount(
                "speculative_prefill_context_target_expert_reclaimed_payload_bytes",
            ),
        speculative_prefill_draft_target_expert_reclaimed_payload_bytes: performance_counter_amount(
            "speculative_prefill_draft_target_expert_reclaimed_payload_bytes",
        ),
        speculative_prefill_target_expert_repopulated_payload_bytes: performance_counter_amount(
            "speculative_prefill_target_expert_repopulated_payload_bytes",
        ),
        speculative_prefill_request_scoped_draft_release_elapsed_seconds: operation_elapsed_seconds(
            "speculative_prefill_request_scoped_draft_release",
        ),
    }
}
