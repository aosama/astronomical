use std::io::Cursor;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId, WorkerSpeculativePrefillConfiguration,
};
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PerformanceAttribution, PerformanceAttributionLog, PersistentPromptCacheDiskStoreConfig,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxMemoryLimits;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

const CACHE_DELETED_VISUAL_PROMPT_MINIMUM_TOKEN_COUNT: usize = 47_000;
const CACHE_DELETED_VISUAL_PROMPT_MAXIMUM_TOKEN_COUNT: usize = 49_000;
const CACHE_DELETED_VISUAL_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const CACHE_DELETED_VISUAL_REQUEST_IDENTIFIER: u64 = 95_200;
const RESTORED_VISUAL_REQUEST_IDENTIFIER: u64 = 95_201;
const CACHE_DELETED_VISUAL_MLX_MEMORY_LIMIT_BYTES: usize = 30_000_000_000;
const ROMEO_AND_JULIET_SOURCE_TEXT: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "loads Ornith and proves cache-deleted 48K visual SpecPrefill draft scoring under the configured 30 GB MLX ceiling"]
async fn should_complete_cache_deleted_and_restored_48k_visual_speculative_prefill_without_target_only_fallback()
 {
    tokio::time::timeout(
        CACHE_DELETED_VISUAL_QUALIFICATION_TIMEOUT,
        run_cache_deleted_visual_speculative_prefill_qualification(),
    )
    .await
    .expect("the cache-deleted visual SpecPrefill qualification must finish within 115 seconds");
}

async fn run_cache_deleted_visual_speculative_prefill_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
    let (draft_model_directory, draft_model_id) =
        super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
    let validated_draft_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&draft_model_directory, 1)
        .expect("the visual draft artifact should validate");
    let draft_model_revision = validated_draft_artifact.revision().to_owned();
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_model_directory, 1)
        .expect("the cache-deleted visual target artifact should validate");
    let target_model_id = validated_target_artifact.model_id().to_owned();
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the cache-deleted visual tokenizer should load");
    let visual_request = cache_deleted_visual_request(
        &tokenizer,
        &target_model_id,
        CACHE_DELETED_VISUAL_REQUEST_IDENTIFIER,
    )
    .with_performance_attribution(PerformanceAttribution::enabled());
    assert!(
        visual_request.input_token_ids().len() >= CACHE_DELETED_VISUAL_PROMPT_MINIMUM_TOKEN_COUNT,
        "expected at least {} prompt tokens but prepared {}",
        CACHE_DELETED_VISUAL_PROMPT_MINIMUM_TOKEN_COUNT,
        visual_request.input_token_ids().len()
    );
    assert!(
        visual_request.input_token_ids().len() <= CACHE_DELETED_VISUAL_PROMPT_MAXIMUM_TOKEN_COUNT,
        "expected at most {} prompt tokens but prepared {}",
        CACHE_DELETED_VISUAL_PROMPT_MAXIMUM_TOKEN_COUNT,
        visual_request.input_token_ids().len()
    );
    let temporary_prompt_cache_directory =
        tempfile::tempdir().expect("the cache-deleted visual test should create an empty cache");
    let temporary_attribution_directory = tempfile::tempdir()
        .expect("the cache-deleted visual test should create an attribution directory");
    let performance_attribution_log_path = temporary_attribution_directory
        .path()
        .join("performance-attribution.jsonl");
    let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
        temporary_prompt_cache_directory.path().to_path_buf(),
        temporary_prompt_cache_directory.path().to_path_buf(),
        50_000_000_000,
    );
    let mlx_memory_limits = MlxMemoryLimits::new(CACHE_DELETED_VISUAL_MLX_MEMORY_LIMIT_BYTES, 0)
        .expect("the configured 30 GB MLX limits should be valid");
    let speculative_prefill = WorkerSpeculativePrefillConfiguration {
        enabled: true,
        target_model_id: Some(target_model_id.clone()),
        draft_model_id: Some(draft_model_id.clone()),
        draft_model_directory: Some(draft_model_directory.clone()),
        minimum_prompt_tokens: 8_192,
        keep_percentage: 20,
        selection_chunck_token_count: 32,
        mandatory_trailing_token_count: 512,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    };
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer_and_speculative_prefill_and_performance_attribution(
        validated_target_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(persistent_prompt_cache_disk_store_config.clone()),
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(4_096)
            .expect("the visual qualification prefill chunk size should be valid"),
        tokenizer.think_end_token_id(),
        target_model_directory.clone(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
        false,
        speculative_prefill,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the visual qualification attribution log should open"),
    )
    .expect("the cache-deleted visual SpecPrefill engine should construct");
    eprintln!("[cache-deleted-visual-specprefill] status=loading-model timeout_seconds=115");
    qwen3_5_engine
        .load()
        .await
        .expect("the target and startup-validated draft should load");
    eprintln!(
        "[cache-deleted-visual-specprefill] status=prefill prompt_tokens={} cache_state=deleted",
        visual_request.input_token_ids().len()
    );
    qwen3_5_engine
        .start_generation(visual_request)
        .await
        .expect("the cache-deleted visual prompt should start with SpecPrefill enabled");
    loop {
        match qwen3_5_engine
            .decode_next_token(RequestId::new(CACHE_DELETED_VISUAL_REQUEST_IDENTIFIER))
            .await
            .expect("the cache-deleted visual prompt should not fall back or fail")
        {
            GeneratedToken::TokenId { .. } => {
                eprintln!("[cache-deleted-visual-specprefill] status=output-token-generated");
                break;
            }
            GeneratedToken::EndOfSequence => break,
            GeneratedToken::PrefillProgress {
                completed_prefill_chunck_tokens,
                ..
            } => eprintln!(
                "[cache-deleted-visual-specprefill] status=prefill-progress completed_prefill_chunck_tokens={completed_prefill_chunck_tokens}"
            ),
        }
    }
    let attribution_report_documents =
        super::performance_attribution::read_attribution_report_documents(
            &performance_attribution_log_path,
        );
    let generation_report = super::performance_attribution::generation_report_for_request(
        &attribution_report_documents,
        CACHE_DELETED_VISUAL_REQUEST_IDENTIFIER,
    );
    assert!(
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            generation_report,
            "speculative_prefill_request_scoped_draft_load",
        ) > 0,
        "the cache-deleted visual request must load a request-scoped draft after expert eviction"
    );
    assert!(
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            generation_report,
            "speculative_prefill_draft_scoring",
        ) > 0,
        "the cache-deleted visual request must complete draft scoring"
    );
    assert!(
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            generation_report,
            "speculative_prefill_request_scoped_draft_release",
        ) > 0,
        "the cache-deleted visual request must release the draft before target paging"
    );
    assert_eq!(
        super::performance_attribution::counter_amount(
            generation_report,
            "speculative_prefill_fallback_count",
        ),
        0,
        "RAM pressure must evict target experts and never convert this request to target-only"
    );
    drop(qwen3_5_engine);
    let draft_kv_block_directory = temporary_prompt_cache_directory
        .path()
        .join(&draft_model_id)
        .join(&draft_model_revision)
        .join("kv_blocks");
    assert!(
        std::fs::read_dir(&draft_kv_block_directory)
            .expect("the cold visual request should create the draft KV directory")
            .next()
            .is_some(),
        "the cold visual request must publish reusable draft KV blocks"
    );

    let restored_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_model_directory, 1)
        .expect("the restored-cache visual target artifact should validate");
    let restored_visual_request = cache_deleted_visual_request(
        &tokenizer,
        &target_model_id,
        RESTORED_VISUAL_REQUEST_IDENTIFIER,
    )
    .with_performance_attribution(PerformanceAttribution::enabled());
    let restored_performance_attribution_log_path = temporary_attribution_directory
        .path()
        .join("restored-performance-attribution.jsonl");
    let restored_mlx_memory_limits =
        MlxMemoryLimits::new(CACHE_DELETED_VISUAL_MLX_MEMORY_LIMIT_BYTES, 0)
            .expect("the restored-cache 30 GB MLX limits should be valid");
    let restored_speculative_prefill = WorkerSpeculativePrefillConfiguration {
        enabled: true,
        target_model_id: Some(target_model_id),
        draft_model_id: Some(draft_model_id),
        draft_model_directory: Some(draft_model_directory),
        minimum_prompt_tokens: 8_192,
        keep_percentage: 20,
        selection_chunck_token_count: 32,
        mandatory_trailing_token_count: 512,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    };
    let mut restored_qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer_and_speculative_prefill_and_performance_attribution(
        restored_target_artifact,
        restored_mlx_memory_limits.active_memory_limit_bytes(),
        restored_mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(persistent_prompt_cache_disk_store_config),
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(4_096)
            .expect("the restored visual qualification prefill chunk size should be valid"),
        tokenizer.think_end_token_id(),
        target_model_directory,
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
        false,
        restored_speculative_prefill,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&restored_performance_attribution_log_path, true)
            .expect("the restored visual qualification attribution log should open"),
    )
    .expect("the restored-cache visual SpecPrefill engine should construct");
    eprintln!("[restored-visual-specprefill] status=loading-model timeout_seconds=115");
    restored_qwen3_5_engine
        .load()
        .await
        .expect("the restored-cache target and startup-validated draft should load");
    eprintln!(
        "[restored-visual-specprefill] status=prefill prompt_tokens={} cache_state=restored",
        restored_visual_request.input_token_ids().len()
    );
    restored_qwen3_5_engine
        .start_generation(restored_visual_request)
        .await
        .expect("the restored-cache visual prompt should start with SpecPrefill enabled");
    loop {
        let generated_token = restored_qwen3_5_engine
            .decode_next_token(RequestId::new(RESTORED_VISUAL_REQUEST_IDENTIFIER));
        let generated_token = generated_token
            .await
            .expect("the restored-cache visual prompt should not fall back or fail");
        match generated_token {
            GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => break,
            GeneratedToken::PrefillProgress {
                completed_prefill_chunck_tokens,
                ..
            } => eprintln!(
                "[restored-visual-specprefill] status=prefill-progress completed_prefill_chunck_tokens={completed_prefill_chunck_tokens}"
            ),
        }
    }
    let restored_attribution_report_documents =
        super::performance_attribution::read_attribution_report_documents(
            &restored_performance_attribution_log_path,
        );
    let restored_generation_report = super::performance_attribution::generation_report_for_request(
        &restored_attribution_report_documents,
        RESTORED_VISUAL_REQUEST_IDENTIFIER,
    );
    assert_eq!(
        super::performance_attribution::counter_amount(
            restored_generation_report,
            "restored_persistent_prompt_cache_token_count",
        ),
        0,
        "the restarted sparse visual request must not require target-cache population"
    );
    assert!(
        super::performance_attribution::counter_amount(
            restored_generation_report,
            "speculative_prefill_target_persistent_state_restored_token_count",
        ) > 0,
        "the restarted visual request must restore its selection-bound target state"
    );
    assert_eq!(
        super::performance_attribution::counter_amount(
            restored_generation_report,
            "speculative_prefill_draft_persistent_prefix_restored_token_count",
        ),
        0,
        "an exact visual target-state hit must not report drafter reuse"
    );
    assert_eq!(
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            restored_generation_report,
            "speculative_prefill_request_scoped_draft_load",
        ),
        0,
        "an exact visual target-state hit must not load a request-scoped drafter"
    );
    assert_eq!(
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            restored_generation_report,
            "speculative_prefill_draft_scoring",
        ),
        0,
        "an exact visual target-state hit must not repeat drafter scoring"
    );
    assert_eq!(
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            restored_generation_report,
            "speculative_prefill_request_scoped_draft_release",
        ),
        0,
        "an exact visual target-state hit has no request-scoped drafter to release"
    );
    assert_eq!(
        super::performance_attribution::counter_amount(
            restored_generation_report,
            "speculative_prefill_fallback_count",
        ),
        0,
        "the restored visual request must not convert to target-only under RAM pressure"
    );
    eprintln!("[cache-deleted-and-restored-visual-specprefill] status=success");
}

fn cache_deleted_visual_request(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    request_identifier: u64,
) -> astronomical_model_serving::Qwen3_5InferenceRequest {
    let source_material = format!(
        "{}{}",
        ROMEO_AND_JULIET_SOURCE_TEXT.repeat(6),
        ROMEO_AND_JULIET_SOURCE_TEXT
            .chars()
            .take(4_000)
            .collect::<String>()
    );
    tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(request_identifier),
                model: target_model_id.to_owned(),
                messages: vec![ChatMessage::User {
                    content: format!(
                        "Use both images and the supplied Romeo and Juliet source material to write a concise literary analysis.\n\n{source_material}"
                    ),
                    images: vec![one_pixel_png(64, 96, 128), one_pixel_png(160, 96, 32)],
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: 1,
                    temperature_thousandths: Some(0),
                    top_p_thousandths: Some(1_000),
                    seed: Some(1),
                    thinking_budget: None,
                },
            },
            false,
        )
        .expect("the cache-deleted visual prompt should prepare")
}

fn one_pixel_png(red: u8, green: u8, blue: u8) -> ChatImageInput {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([red, green, blue]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the visual qualification image should encode");
    ChatImageInput {
        mime_type: "image/png".to_owned(),
        decoded_bytes: encoded_image_bytes.into_inner(),
    }
}
