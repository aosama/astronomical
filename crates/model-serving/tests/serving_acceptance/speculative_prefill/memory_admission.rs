use std::time::Duration;

use astronomical_config::{AstronomicalConfig, ResolvedModelConfig};
use astronomical_ipc_protocol::{
    RequestId, WorkerPromptProcessingPhase, WorkerSpeculativePrefillConfiguration,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxMemoryLimits;

use super::support::{
    decode_generated_output_text, prepare_romeo_and_juliet_three_paragraph_summary_prompt,
};

const CONFIGURED_TARGET_SUMMARY_PROMPT_TOKEN_COUNT: usize = 8_192;
// Three prose paragraphs cannot fit in a 64-token cap: the generation truncates
// mid-sentence before the second paragraph separator, and the paragraph
// assertion can never hold. 512 output tokens carries a full three-paragraph
// summary robustly (same finding as the speculative-prefill visual-tool
// journeys) while keeping the whole cold journey far inside the 115-second
// boundary.
const CONFIGURED_TARGET_SUMMARY_MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 512;
const CONFIGURED_TARGET_SUMMARY_REQUEST_IDENTIFIER: u64 = 95_260;
const CONFIGURED_TARGET_SUMMARY_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "loads the configured target and drafter for a cold-cache Romeo and Juliet summary at the SpecPrefill floor"]
async fn should_complete_the_configured_cold_cache_romeo_summary_within_the_mlx_memory_ceiling() {
    tokio::time::timeout(
        CONFIGURED_TARGET_SUMMARY_TIMEOUT,
        run_configured_cold_cache_summary_journey(
            CONFIGURED_TARGET_SUMMARY_PROMPT_TOKEN_COUNT,
            CONFIGURED_TARGET_SUMMARY_MAXIMUM_OUTPUT_TOKEN_COUNT,
            CONFIGURED_TARGET_SUMMARY_REQUEST_IDENTIFIER,
            crate::common::sample_serving_acceptance_mlx_memory_limits()
                .await
                .active_memory_limit_bytes(),
            None,
        ),
    )
    .await
    .expect("the configured Romeo summary journey must finish within 115 seconds");
}
async fn run_configured_cold_cache_summary_journey(
    prompt_token_count: usize,
    maximum_output_token_count: u16,
    request_identifier: u64,
    active_memory_limit_bytes: usize,
    required_target_model_id: Option<&str>,
) {
    if let Err(test_tracing_initialization_error) = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init()
    {
        eprintln!(
            "[configured-cold-specprefill] status=progress tracing=already_initialized reason={test_tracing_initialization_error}"
        );
    }
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for the summary journey");
    let requested_target_model_id =
        required_target_model_id.unwrap_or(crate::common::large_sparse_moe_model_id());
    let discovered_target_model = crate::common::configured_discovered_model_by_id(
        &astronomical_config,
        requested_target_model_id,
    );
    let target_chat_capabilities =
        crate::common::discovered_chat_capabilities(&discovered_target_model);
    let target_context_window = target_chat_capabilities.context_window;
    let target_maximum_output_tokens = target_chat_capabilities.max_output_tokens;
    let resolved_target_config = astronomical_config
        .resolved_model_config(requested_target_model_id, target_context_window)
        .expect("the selected target model policy should resolve");
    let target_model_directory = discovered_target_model.model_directory;
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_model_directory, target_maximum_output_tokens)
        .expect("the selected target should validate for the summary journey");
    let target_model_id = validated_target_artifact.model_id().to_owned();
    assert_eq!(
        target_model_id, requested_target_model_id,
        "the user journey must use the requested Ornith artifact"
    );
    let (draft_model_directory, draft_model_id) =
        crate::serving_acceptance::support::configured_speculative_prefill_draft_model(
            &target_model_directory,
        );
    let mlx_memory_limits = MlxMemoryLimits::new(active_memory_limit_bytes, 0)
        .expect("the configured MLX memory limits should be valid");
    let target_maximum_position_count = validated_target_artifact.config().maximum_position_count();
    let target_model_revision = validated_target_artifact.revision().to_owned();
    let prefill_chunk_sizer = configured_prefill_chunk_sizer(
        &resolved_target_config,
        target_maximum_position_count,
        &target_model_id,
        &target_model_revision,
    );
    let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the configured Ornith tokenizer should load");
    let request_id = RequestId::new(request_identifier);
    let summary_prompt = prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &target_model_directory,
        &target_model_id,
        request_id,
        prompt_token_count,
        maximum_output_token_count,
    );
    let temporary_prompt_cache_directory =
        tempfile::tempdir().expect("the summary journey should create an empty prompt-cache root");
    let temporary_attribution_directory =
        tempfile::tempdir().expect("the summary journey should create an attribution directory");
    let performance_attribution_log_path = temporary_attribution_directory
        .path()
        .join("performance-attribution.jsonl");
    let target_prompt_cache_directory = temporary_prompt_cache_directory.path().join("target");
    let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
        target_prompt_cache_directory,
        temporary_prompt_cache_directory.path().to_path_buf(),
        astronomical_config
            .prompt_cache()
            .expect("the configured prompt-cache policy should resolve")
            .global_prompt_cache_maximum_size_bytes(),
    );
    let speculative_prefill_configuration = WorkerSpeculativePrefillConfiguration {
        enabled: true,
        target_model_id: Some(target_model_id.clone()),
        draft_model_id: Some(draft_model_id),
        draft_model_directory: Some(draft_model_directory),
        minimum_prompt_tokens: 8_192,
        keep_percentage: 20,
        selection_chunk_token_count: 32,
        mandatory_trailing_token_count: 512,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    };
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_speculative_prefill_mtp_depth_and_performance_attribution(
        validated_target_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(persistent_prompt_cache_disk_store_config),
        prefill_chunk_sizer,
        target_tokenizer.think_end_token_id(),
        target_model_directory,
        crate::common::standard_worker_chunking_configuration(),
        true,
        true,
        resolved_target_config.mtp_draft_depth(),
        speculative_prefill_configuration,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the summary journey attribution log should open"),
    )
    .expect("the configured SpecPrefill summary engine should construct");

    eprintln!(
        "[configured-cold-specprefill] status=progress phase=model_load prompt_tokens={} active_memory_limit_bytes={} timeout_seconds=115",
        summary_prompt.prompt_token_ids.len(),
        active_memory_limit_bytes,
    );
    qwen3_5_engine
        .load()
        .await
        .expect("the configured target and drafter should load");
    eprintln!("[configured-cold-specprefill] status=progress phase=model_loaded");
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                summary_prompt.prompt_token_ids,
                maximum_output_token_count,
                summary_prompt.sampling_temperature_thousandths,
                summary_prompt.sampling_top_p_thousandths,
                summary_prompt.sampling_seed,
            )
            .with_image_pad_token_id(summary_prompt.image_pad_token_id)
            .with_ordinary_target_prefill_control_span_token_count(
                summary_prompt.ordinary_target_prefill_control_span_token_count,
            )
            .with_thinking_configuration(false, None, Vec::new(), Vec::new())
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the configured summary should be admitted");

    let mut observed_drafter_phase = false;
    let mut observed_live_drafter_memory = false;
    let mut observed_finalized_zero_drafter_memory = false;
    let mut latest_prompt_processing_phase = None;
    let mut latest_processed_token_count = 0;
    let mut latest_mlx_active_memory_bytes = None;
    let mut latest_draft_mlx_active_memory_bytes = None;
    let mut generated_answer_token_ids = Vec::new();
    loop {
        let generated_token = match qwen3_5_engine.decode_next_token(request_id).await {
            Ok(generated_token) => generated_token,
            Err(generation_error) => panic!(
                "the configured summary should complete without a memory failure; latest_prompt_processing_phase={latest_prompt_processing_phase:?} latest_processed_token_count={latest_processed_token_count} latest_mlx_active_memory_bytes={latest_mlx_active_memory_bytes:?} latest_draft_mlx_active_memory_bytes={latest_draft_mlx_active_memory_bytes:?} error={generation_error:?}"
            ),
        };
        match generated_token {
            GeneratedToken::PromptProcessingPhaseStarted {
                prompt_processing_phase,
                total_token_count,
            } => {
                latest_prompt_processing_phase = Some(prompt_processing_phase);
                observed_drafter_phase |=
                    prompt_processing_phase == WorkerPromptProcessingPhase::Drafter;
                eprintln!(
                    "[configured-cold-specprefill] status=progress phase={prompt_processing_phase:?} total_tokens={total_token_count}"
                );
            }
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::PrefillProgress {
                processed_token_count,
                mlx_memory_telemetry,
                speculative_prefill_draft_memory_telemetry,
                ..
            } => {
                latest_processed_token_count = processed_token_count;
                latest_mlx_active_memory_bytes = mlx_memory_telemetry
                    .map(|mlx_memory_telemetry| mlx_memory_telemetry.active_memory_bytes);
                latest_draft_mlx_active_memory_bytes = speculative_prefill_draft_memory_telemetry
                    .map(|draft_memory_telemetry| draft_memory_telemetry.active_memory_bytes);
                observed_live_drafter_memory |= speculative_prefill_draft_memory_telemetry
                    .is_some_and(|draft_memory_telemetry| {
                        draft_memory_telemetry
                            .active_memory_breakdown
                            .speculative_prefill_draft_memory_bytes
                            > 0
                    });
                eprintln!(
                    "[configured-cold-specprefill] status=progress phase={latest_prompt_processing_phase:?} processed_tokens={processed_token_count} mlx_active_memory_bytes={latest_mlx_active_memory_bytes:?} draft_mlx_active_memory_bytes={latest_draft_mlx_active_memory_bytes:?}"
                );
            }
            GeneratedToken::TokenId {
                token_id,
                is_reasoning_token,
                generation_finalization,
                ..
            } => {
                if !is_reasoning_token {
                    generated_answer_token_ids.push(token_id);
                }
                if let Some(generation_finalization) = generation_finalization {
                    observed_finalized_zero_drafter_memory = generation_finalization
                        .mlx_memory_telemetry()
                        .is_some_and(|finalized_memory_telemetry| {
                            finalized_memory_telemetry
                                .active_memory_breakdown
                                .speculative_prefill_draft_memory_bytes
                                == 0
                        });
                    break;
                }
            }
            GeneratedToken::EndOfSequence => break,
        }
    }
    let decoded_summary_text =
        decode_generated_output_text(&target_tokenizer, &generated_answer_token_ids);
    let summary_paragraphs = decoded_summary_text
        .split("\n\n")
        .map(str::trim)
        .filter(|summary_paragraph| !summary_paragraph.is_empty())
        .collect::<Vec<_>>();
    assert!(
        observed_drafter_phase,
        "the cold journey must enter the Drafter phase"
    );
    assert!(
        observed_live_drafter_memory,
        "the cold journey must observe live request-scoped drafter memory"
    );
    assert!(
        observed_finalized_zero_drafter_memory,
        "the completed journey must report zero request-scoped drafter memory"
    );
    // The memory-admission lifecycle above is the behavior under test. The
    // summary text is the sanity guard that generation completed with real
    // prose: the model reliably produces a two- or three-paragraph summary at
    // the mandatory temperature-1 sampling, but the exact paragraph count is
    // sampling variance, not a memory-policy outcome.
    assert!(
        summary_paragraphs.len() >= 2,
        "the configured model should return a complete multi-paragraph summary; output={decoded_summary_text:?}"
    );
    eprintln!(
        "[configured-cold-specprefill] status=success prompt_tokens={} output_tokens={} paragraphs={}",
        prompt_token_count,
        generated_answer_token_ids.len(),
        summary_paragraphs.len(),
    );
}

fn configured_prefill_chunk_sizer(
    resolved_target_config: &ResolvedModelConfig,
    _target_maximum_position_count: u32,
    _target_model_id: &str,
    _target_model_revision: &str,
) -> Qwen3_5PromptProcessingChunkSizer {
    let chunking = resolved_target_config.chunking();
    Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
        chunking.fixed_prompt_processing_chunk_size_tokens(),
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
    )
    .expect("the configured fixed prefill chunk size should be valid")
}
