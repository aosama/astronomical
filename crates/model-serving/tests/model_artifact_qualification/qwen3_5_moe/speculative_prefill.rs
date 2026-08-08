use std::path::Path;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerSpeculativePrefillConfiguration,
};
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PerformanceAttribution, PerformanceAttributionLog, PersistentPromptCacheDiskStoreConfig,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer,
    Qwen3_5Tokenizer,
};
use astronomical_runtime_integration::MlxRuntime;

const REPRESENTATIVE_INPUT_TOKEN_COUNT: usize = 8_192;
const REPRESENTATIVE_OUTPUT_TOKEN_COUNT: u16 = 1_024;
const PARITY_OUTPUT_TOKEN_COUNT: u16 = 64;
const SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS: u32 = 8_192;
pub(super) const SPECULATIVE_PREFILL_KEEP_PERCENTAGE: u32 = 20;
const SPECULATIVE_PREFILL_SELECTION_CHUNCK_TOKEN_COUNT: u32 = 32;
const SPECULATIVE_PREFILL_MANDATORY_TRAILING_TOKEN_COUNT: u32 = 512;
const SPECULATIVE_PREFILL_LOOKAHEAD_TOKEN_COUNT: u32 = 8;
const SPECULATIVE_PREFILL_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT: u32 = 13;
const REPRESENTATIVE_SOURCE_TEXT: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "measures target-only and draft-assisted 8K/1K SpecPrefill behavior"]
async fn should_measure_representative_speculative_prefill_against_a_target_only_control() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let benchmark_prompt = prepare_representative_prompt(&target_model_directory);
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

        eprintln!(
            "[speculative-prefill-release] status=progress phase=target_only_control prompt_tokens={} output_tokens={} ETA_seconds=115",
            benchmark_prompt.prompt_token_ids.len(),
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
        );
        let target_only_before_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &benchmark_prompt,
            false,
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_100),
            None,
            mlx_memory_limits,
        )
        .await;
        clear_reclaimable_mlx_memory(mlx_memory_limits);

        eprintln!(
            "[speculative-prefill-release] status=progress phase=speculative_prefill prompt_tokens={} output_tokens={} ETA_seconds=75",
            benchmark_prompt.prompt_token_ids.len(),
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
        );
        let speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &benchmark_prompt,
            true,
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_101),
            None,
            mlx_memory_limits,
        )
        .await;
        clear_reclaimable_mlx_memory(mlx_memory_limits);

        assert_eq!(
            target_only_before_measurement.generated_token_ids.len(),
            usize::from(REPRESENTATIVE_OUTPUT_TOKEN_COUNT),
        );
        assert_eq!(
            speculative_prefill_measurement.generated_token_ids.len(),
            usize::from(REPRESENTATIVE_OUTPUT_TOKEN_COUNT),
        );
        assert_eq!(
            speculative_prefill_measurement.speculative_prefill_fallback_count,
            0,
            "the representative sparse run must not fall back to target-only execution",
        );
        assert!(
            !target_only_before_measurement.decoded_output_text.trim().is_empty(),
            "the target-only Shakespeare output must decode to readable text",
        );
        assert!(
            !speculative_prefill_measurement.decoded_output_text.trim().is_empty(),
            "the SpecPrefill Shakespeare output must decode to readable text",
        );
        eprintln!(
            "[speculative-prefill-quality] status=output model=target_only begin\n{}\n[speculative-prefill-quality] status=output model=target_only end",
            target_only_before_measurement.decoded_output_text,
        );
        eprintln!(
            "[speculative-prefill-quality] status=output model=speculative_prefill begin\n{}\n[speculative-prefill-quality] status=output model=speculative_prefill end",
            speculative_prefill_measurement.decoded_output_text,
        );
        let target_only_baseline_measurement = &target_only_before_measurement;
        let throughput_ratio = speculative_prefill_measurement
            .generation_tokens_per_second()
            / target_only_baseline_measurement
                .generation_tokens_per_second()
                .max(f64::EPSILON);
        let total_request_speedup_ratio = target_only_baseline_measurement
            .total_request_elapsed_seconds
            / speculative_prefill_measurement
                .total_request_elapsed_seconds
                .max(f64::EPSILON);
        assert_within_memory_limits(
            "target_only_before",
            &target_only_before_measurement,
            mlx_memory_limits,
        );
        assert_within_memory_limits(
            "speculative_prefill",
            &speculative_prefill_measurement,
            mlx_memory_limits,
        );
        eprintln!(
            "[speculative-prefill-release] status=sample prompt_tokens={} output_tokens={} target_only_total_request_seconds={:.3} speculative_prefill_total_request_seconds={:.3} target_only_prefill_seconds={:.3} speculative_prefill_prefill_seconds={:.3} target_only_decode_seconds={:.3} speculative_prefill_decode_seconds={:.3} speculative_prefill_draft_scoring_seconds={:.3} speculative_prefill_sparse_input_assembly_seconds={:.3} speculative_prefill_sparse_target_seconds={:.3} speculative_prefill_fallback_count={} target_only_mtp_accepted={} target_only_mtp_rejected={} speculative_prefill_mtp_accepted={} speculative_prefill_mtp_rejected={} target_only_expert_disk_page_loads={} speculative_prefill_expert_disk_page_loads={} target_only_expert_cache_misses={} speculative_prefill_expert_cache_misses={} target_only_expert_cache_evictions={} speculative_prefill_expert_cache_evictions={} target_only_generation_tokens_per_second={:.2} speculative_prefill_generation_tokens_per_second={:.2} throughput_ratio={:.3} total_request_speedup_ratio={:.3} target_only_active_mlx_bytes={} speculative_prefill_active_mlx_bytes={} target_only_peak_mlx_bytes={} speculative_prefill_peak_mlx_bytes={}",
            benchmark_prompt.prompt_token_ids.len(),
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            target_only_before_measurement.total_request_elapsed_seconds,
            speculative_prefill_measurement.total_request_elapsed_seconds,
            target_only_before_measurement.prompt_prefill_elapsed_seconds,
            speculative_prefill_measurement.prompt_prefill_elapsed_seconds,
            target_only_baseline_measurement.decode_elapsed_seconds,
            speculative_prefill_measurement.decode_elapsed_seconds,
            speculative_prefill_measurement.speculative_prefill_draft_scoring_elapsed_seconds,
            speculative_prefill_measurement.speculative_prefill_sparse_input_assembly_elapsed_seconds,
            speculative_prefill_measurement.speculative_prefill_sparse_target_elapsed_seconds,
            speculative_prefill_measurement.speculative_prefill_fallback_count,
            target_only_baseline_measurement.mtp_accepted_draft_count,
            target_only_baseline_measurement.mtp_rejected_draft_count,
            speculative_prefill_measurement.mtp_accepted_draft_count,
            speculative_prefill_measurement.mtp_rejected_draft_count,
            target_only_baseline_measurement.expert_weight_disk_page_load_count,
            speculative_prefill_measurement.expert_weight_disk_page_load_count,
            target_only_baseline_measurement.expert_weight_memory_cache_miss_count,
            speculative_prefill_measurement.expert_weight_memory_cache_miss_count,
            target_only_baseline_measurement.expert_weight_memory_cache_eviction_count,
            speculative_prefill_measurement.expert_weight_memory_cache_eviction_count,
            target_only_baseline_measurement.generation_tokens_per_second(),
            speculative_prefill_measurement.generation_tokens_per_second(),
            throughput_ratio,
            total_request_speedup_ratio,
            target_only_baseline_measurement.maximum_active_memory_bytes,
            speculative_prefill_measurement.maximum_active_memory_bytes,
            target_only_baseline_measurement.maximum_peak_memory_bytes,
            speculative_prefill_measurement.maximum_peak_memory_bytes,
        );
        assert!(throughput_ratio.is_finite() && throughput_ratio > 0.0);
        assert!(total_request_speedup_ratio.is_finite() && total_request_speedup_ratio > 0.0);
        eprintln!("[speculative-prefill-release] status=success");
    })
    .await
    .expect("the representative SpecPrefill control comparison should finish within 115 seconds");
}

#[tokio::test]
#[ignore = "qualifies exact greedy parity for full-retention SpecPrefill"]
async fn should_preserve_target_only_greedy_output_with_full_speculative_prefill_retention() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let parity_prompt = prepare_representative_prompt(&target_model_directory);
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

        eprintln!("[speculative-prefill-parity] status=progress phase=target_only ETA_seconds=55");
        let target_only_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &parity_prompt,
            false,
            PARITY_OUTPUT_TOKEN_COUNT,
            100,
            RequestId::new(95_110),
            None,
            mlx_memory_limits,
        )
        .await;
        clear_reclaimable_mlx_memory(mlx_memory_limits);
        eprintln!(
            "[speculative-prefill-parity] status=progress phase=full_retention ETA_seconds=55"
        );
        let speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &parity_prompt,
            true,
            PARITY_OUTPUT_TOKEN_COUNT,
            100,
            RequestId::new(95_111),
            None,
            mlx_memory_limits,
        )
        .await;

        assert_eq!(
            speculative_prefill_measurement.generated_token_ids,
            target_only_measurement.generated_token_ids,
            "full-retention SpecPrefill must preserve exact target-only greedy output",
        );
        assert_eq!(
            speculative_prefill_measurement.speculative_prefill_fallback_count,
            0
        );
        assert!(
            speculative_prefill_measurement.speculative_prefill_draft_scoring_elapsed_seconds > 0.0
        );
        eprintln!("[speculative-prefill-parity] status=success");
    })
    .await
    .expect("the full-retention SpecPrefill parity gate should finish within 115 seconds");
}

pub(super) struct RepresentativePrompt {
    pub(super) prompt_token_ids: Vec<u32>,
    pub(super) image_pad_token_id: u32,
}

pub(super) struct RepresentativeGenerationMeasurement {
    pub(super) generated_token_ids: Vec<u32>,
    decoded_output_text: String,
    total_request_elapsed_seconds: f64,
    prompt_prefill_elapsed_seconds: f64,
    decode_elapsed_seconds: f64,
    speculative_prefill_draft_scoring_elapsed_seconds: f64,
    speculative_prefill_sparse_input_assembly_elapsed_seconds: f64,
    speculative_prefill_sparse_target_elapsed_seconds: f64,
    mtp_accepted_draft_count: u64,
    mtp_rejected_draft_count: u64,
    expert_weight_disk_page_load_count: u64,
    expert_weight_memory_cache_miss_count: u64,
    expert_weight_memory_cache_eviction_count: u64,
    maximum_active_memory_bytes: u64,
    maximum_peak_memory_bytes: u64,
    pub(super) speculative_prefill_fallback_count: u64,
    pub(super) speculative_prefill_draft_persistent_prefix_restored_token_count: u64,
    pub(super) restored_target_persistent_prompt_cache_token_count: u64,
    pub(super) speculative_prefill_target_persistent_state_write_count: u64,
    pub(super) speculative_prefill_target_persistent_state_restored_token_count: u64,
    pub(super) speculative_prefill_draft_scored_suffix_token_count: u64,
    pub(super) speculative_prefill_context_target_expert_reclaimed_payload_bytes: u64,
    pub(super) speculative_prefill_draft_target_expert_reclaimed_payload_bytes: u64,
    pub(super) speculative_prefill_target_expert_repopulated_payload_bytes: u64,
    pub(super) speculative_prefill_request_scoped_draft_release_elapsed_seconds: f64,
}

impl RepresentativeGenerationMeasurement {
    fn generation_tokens_per_second(&self) -> f64 {
        self.generated_token_ids.len() as f64 / self.total_request_elapsed_seconds.max(f64::EPSILON)
    }
}

fn decode_generated_output_text(
    tokenizer: &Qwen3_5Tokenizer,
    generated_token_ids: &[u32],
) -> String {
    let mut token_decoder = tokenizer.incremental_decoder();
    let mut decoded_output_text = String::new();
    for generated_token_id in generated_token_ids {
        if let Some(decoded_fragment) = token_decoder
            .push_token(*generated_token_id)
            .expect("the generated Shakespeare output should decode")
        {
            decoded_output_text.push_str(&decoded_fragment);
        }
    }
    if let Some(decoded_fragment) = token_decoder
        .finish()
        .expect("the generated Shakespeare output should flush")
    {
        decoded_output_text.push_str(&decoded_fragment);
    }
    decoded_output_text
}

pub(super) fn prepare_representative_prompt(model_directory: &Path) -> RepresentativePrompt {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the representative SpecPrefill target artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the representative SpecPrefill tokenizer should load");
    let mut repeated_source_material = String::new();
    let prepared_chat_request = loop {
        if !repeated_source_material.is_empty() {
            repeated_source_material.push_str("\n\n");
        }
        repeated_source_material.push_str(REPRESENTATIVE_SOURCE_TEXT);
        let prepared_chat_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_000),
                    model: validated_artifact.model_id().to_owned(),
                    messages: vec![ChatMessage::User {
                        content: format!(
                            "Write a coherent literary synopsis of Romeo and Juliet from the source material. Preserve the characters, relationships, major events, and tragic ending. Do not discuss these instructions or repeat the source text. Write at least seven hundred words.\n\nSource material:\n{repeated_source_material}"
                        ),
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
                        temperature_thousandths: Some(0),
                        top_p_thousandths: Some(1_000),
                        seed: None,
                        thinking_budget: None,
                    },
                },
                false,
            )
            .expect("the representative SpecPrefill prompt should prepare");
        if prepared_chat_request.input_token_ids().len() >= REPRESENTATIVE_INPUT_TOKEN_COUNT {
            break prepared_chat_request;
        }
    };
    let complete_prompt_token_ids = prepared_chat_request.input_token_ids().to_vec();
    let assistant_suffix_start = complete_prompt_token_ids
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .expect("the representative prompt should contain the assistant suffix marker");
    let assistant_suffix_token_ids = &complete_prompt_token_ids[assistant_suffix_start..];
    let retained_prompt_prefix_token_count = REPRESENTATIVE_INPUT_TOKEN_COUNT
        .checked_sub(assistant_suffix_token_ids.len())
        .expect("the assistant suffix should fit the representative input budget");
    assert!(retained_prompt_prefix_token_count < assistant_suffix_start);
    let mut prompt_token_ids =
        complete_prompt_token_ids[..retained_prompt_prefix_token_count].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix_token_ids);
    assert_eq!(prompt_token_ids.len(), REPRESENTATIVE_INPUT_TOKEN_COUNT);
    RepresentativePrompt {
        prompt_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
    }
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
        selection_chunck_token_count: SPECULATIVE_PREFILL_SELECTION_CHUNCK_TOKEN_COUNT,
        mandatory_trailing_token_count: SPECULATIVE_PREFILL_MANDATORY_TRAILING_TOKEN_COUNT,
        lookahead_token_count: SPECULATIVE_PREFILL_LOOKAHEAD_TOKEN_COUNT,
        importance_pooling_kernel_token_count:
            SPECULATIVE_PREFILL_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT,
    };
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer_and_speculative_prefill_and_performance_attribution(
        validated_target_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        persistent_prompt_cache_disk_store_config,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(32)
            .expect("the representative prefill chunk size should be valid"),
        target_think_end_token_id,
        target_model_directory.to_path_buf(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
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
    let generation_started_at = Instant::now();
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                representative_prompt.prompt_token_ids.clone(),
                maximum_output_token_count,
            )
            .with_image_pad_token_id(representative_prompt.image_pad_token_id)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the representative prompt should be admitted");
    let mut generated_token_ids = Vec::with_capacity(usize::from(maximum_output_token_count));
    let mut maximum_active_memory_bytes = 0_u64;
    let mut maximum_peak_memory_bytes = 0_u64;
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
        let memory_telemetry = match &generated_token {
            GeneratedToken::TokenId {
                token_id,
                mlx_memory_telemetry,
                ..
            } => {
                generated_token_ids.push(*token_id);
                mlx_memory_telemetry.as_ref()
            }
            GeneratedToken::PrefillProgress {
                mlx_memory_telemetry,
                ..
            } => mlx_memory_telemetry.as_ref(),
            GeneratedToken::EndOfSequence => None,
        };
        if let Some(memory_telemetry) = memory_telemetry {
            maximum_active_memory_bytes =
                maximum_active_memory_bytes.max(memory_telemetry.active_memory_bytes);
            maximum_peak_memory_bytes =
                maximum_peak_memory_bytes.max(memory_telemetry.peak_memory_bytes);
        }
        if generated_token_ids.len() >= usize::from(maximum_output_token_count)
            || matches!(generated_token, GeneratedToken::EndOfSequence)
        {
            break;
        }
    }
    let total_request_elapsed_seconds = generation_started_at.elapsed().as_secs_f64();
    let decoded_output_text = decode_generated_output_text(&target_tokenizer, &generated_token_ids);
    let attribution_report_documents =
        super::performance_attribution::read_attribution_report_documents(
            &performance_attribution_log_path,
        );
    let generation_report = super::performance_attribution::generation_report_for_request(
        &attribution_report_documents,
        request_id.value(),
    );
    let operation_elapsed_seconds = |operation_identifier| {
        super::performance_attribution::operation_total_elapsed_nanoseconds(
            generation_report,
            operation_identifier,
        ) as f64
            / 1_000_000_000.0
    };
    let performance_counter_amount = |counter_identifier| {
        super::performance_attribution::counter_amount(generation_report, counter_identifier)
    };
    RepresentativeGenerationMeasurement {
        generated_token_ids,
        decoded_output_text,
        total_request_elapsed_seconds,
        prompt_prefill_elapsed_seconds: operation_elapsed_seconds("prompt_prefill_advance_span"),
        decode_elapsed_seconds: operation_elapsed_seconds("decode_advance_span"),
        speculative_prefill_draft_scoring_elapsed_seconds: operation_elapsed_seconds(
            "speculative_prefill_draft_scoring",
        ),
        speculative_prefill_sparse_input_assembly_elapsed_seconds: operation_elapsed_seconds(
            "speculative_prefill_sparse_input_assembly",
        ),
        speculative_prefill_sparse_target_elapsed_seconds: operation_elapsed_seconds(
            "speculative_prefill_sparse_target_forward",
        ),
        mtp_accepted_draft_count: performance_counter_amount("mtp_accepted_draft_count"),
        mtp_rejected_draft_count: performance_counter_amount("mtp_rejected_draft_count"),
        expert_weight_disk_page_load_count: performance_counter_amount(
            "expert_weight_disk_page_load_count",
        ),
        expert_weight_memory_cache_miss_count: performance_counter_amount(
            "expert_weight_memory_cache_miss_count",
        ),
        expert_weight_memory_cache_eviction_count: performance_counter_amount(
            "expert_weight_memory_cache_eviction_count",
        ),
        maximum_active_memory_bytes,
        maximum_peak_memory_bytes,
        speculative_prefill_fallback_count: performance_counter_amount(
            "speculative_prefill_fallback_count",
        ),
        speculative_prefill_draft_persistent_prefix_restored_token_count:
            performance_counter_amount(
                "speculative_prefill_draft_persistent_prefix_restored_token_count",
            ),
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

fn clear_reclaimable_mlx_memory(
    mlx_memory_limits: astronomical_runtime_integration::MlxMemoryLimits,
) {
    MlxRuntime::initialize(mlx_memory_limits)
        .expect("the representative benchmark should re-enter the configured MLX runtime")
        .clear_allocator_cache()
        .expect("the representative benchmark should clear reclaimable MLX memory");
}

fn assert_within_memory_limits(
    measurement_name: &str,
    measurement: &RepresentativeGenerationMeasurement,
    mlx_memory_limits: astronomical_runtime_integration::MlxMemoryLimits,
) {
    let configured_memory_limit_bytes = mlx_memory_limits.active_memory_limit_bytes() as u64;
    assert!(
        measurement.maximum_active_memory_bytes <= configured_memory_limit_bytes,
        "{measurement_name} active MLX memory must remain within the configured limit",
    );
    assert!(
        measurement.maximum_peak_memory_bytes
            <= configured_memory_limit_bytes + configured_memory_limit_bytes / 100,
        "{measurement_name} peak MLX memory must remain within the one-percent allowance",
    );
}
