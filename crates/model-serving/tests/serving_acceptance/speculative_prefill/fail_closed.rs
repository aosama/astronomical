use std::time::Duration;

use astronomical_ipc_protocol::{RequestId, WorkerSpeculativePrefillConfiguration};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionLog, PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer,
    Qwen3_5SpeculativePrefillFailureStageForTests, Qwen3_5Tokenizer,
};

use super::{SPECULATIVE_PREFILL_KEEP_PERCENTAGE, prepare_representative_prompt};

const FORCED_FAILURE_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const RECOVERY_PROGRESS_LOG_INTERVAL: usize = 64;

#[tokio::test]
#[ignore = "loads the target and proves an unavailable configured drafter stops model activation"]
async fn should_stop_model_activation_when_the_configured_drafter_is_unavailable() {
    tokio::time::timeout(FORCED_FAILURE_ACCEPTANCE_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 1)
            .expect("the unavailable-drafter target artifact should validate");
        let target_model_id = validated_target_artifact.model_id().to_owned();
        let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the unavailable-drafter target tokenizer should load");
        let unavailable_draft_directory = tempfile::tempdir()
            .expect("the unavailable-drafter journey should create an empty draft directory");
        let mlx_memory_limits =
            crate::common::sample_serving_acceptance_mlx_memory_limits().await;
        let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_target_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(32)
                .expect("the unavailable-drafter prefill chunk size should be valid"),
            target_tokenizer.think_end_token_id(),
            target_model_directory,
            crate::common::standard_worker_chunking_configuration(),
            true,
            false,
            WorkerSpeculativePrefillConfiguration {
                enabled: true,
                target_model_id: Some(target_model_id),
                draft_model_id: Some("unavailable-draft-model".to_owned()),
                draft_model_directory: Some(unavailable_draft_directory.path().to_path_buf()),
                minimum_prompt_tokens: 8_192,
                keep_percentage: SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
                selection_chunck_token_count: 32,
                mandatory_trailing_token_count: 512,
                lookahead_token_count: 8,
                importance_pooling_kernel_token_count: 13,
            },
            PerformanceAttribution::disabled(),
            PerformanceAttributionLog::disabled(),
        )
        .expect("the unavailable-drafter engine should construct before model activation");

        let model_activation_failure = qwen3_5_engine
            .load()
            .await
            .expect_err("an unavailable configured drafter must stop model activation");
        assert!(matches!(
            model_activation_failure,
            InferenceEngineError::Fatal {
                reason: ref model_loading_failure_reason,
            }
                if model_loading_failure_reason == "configured SpecPrefill failed during draft model artifact validation; model use was stopped"
                    && !model_loading_failure_reason.contains(
                        unavailable_draft_directory.path().to_string_lossy().as_ref()
                    )
        ));
        eprintln!(
            "[speculative-prefill-fail-closed] status=success stage=draft_model_artifact_validation"
        );
    })
    .await
    .expect("the unavailable configured drafter journey should finish within 115 seconds");
}

#[tokio::test]
#[ignore = "loads the configured target and drafter and recovers one sparse target active-memory rejection"]
async fn should_recover_sparse_target_memory_pressure_without_target_only_retry() {
    tokio::time::timeout(FORCED_FAILURE_ACCEPTANCE_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let (draft_model_directory, draft_model_id) =
            crate::serving_acceptance::support::configured_speculative_prefill_draft_model(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 1)
            .expect("the memory-recovery target artifact should validate");
        let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the memory-recovery target tokenizer should load");
        let target_model_id = validated_target_artifact.model_id().to_owned();
        let target_image_pad_token_id = target_tokenizer.image_pad_token_id();
        let representative_prompt = prepare_representative_prompt(&target_model_directory);
        let mlx_memory_limits =
            crate::common::sample_serving_acceptance_mlx_memory_limits().await;
        let attribution_directory = tempfile::tempdir()
            .expect("the memory-recovery acceptance should create an attribution directory");
        let performance_attribution_log_path = attribution_directory
            .path()
            .join("performance-attribution.jsonl");
        let persistent_prompt_cache_directory = tempfile::tempdir()
            .expect("the memory-recovery acceptance should create an SSD cache directory");
        let persistent_prompt_cache_config = PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().join("target"),
            persistent_prompt_cache_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );
        let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_target_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            Some(persistent_prompt_cache_config),
            Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(32)
                .expect("the memory-recovery prefill chunk size should be valid"),
            target_tokenizer.think_end_token_id(),
            target_model_directory,
            crate::common::standard_worker_chunking_configuration(),
            true,
            false,
            WorkerSpeculativePrefillConfiguration {
                enabled: true,
                target_model_id: Some(target_model_id),
                draft_model_id: Some(draft_model_id),
                draft_model_directory: Some(draft_model_directory),
                minimum_prompt_tokens: 8_192,
                keep_percentage: SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
                selection_chunck_token_count: 32,
                mandatory_trailing_token_count: 512,
                lookahead_token_count: 8,
                importance_pooling_kernel_token_count: 13,
            },
            PerformanceAttribution::enabled(),
            PerformanceAttributionLog::open(&performance_attribution_log_path, true)
                .expect("the memory-recovery attribution log should open"),
        )
        .expect("the memory-recovery engine should construct");
        eprintln!("[speculative-prefill-memory-recovery] status=loading-model");
        qwen3_5_engine
            .load()
            .await
            .expect("the configured target and drafter should load before memory recovery");

        let recoverable_memory_pressure_request_id = RequestId::new(95_490);
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(
                    recoverable_memory_pressure_request_id,
                    representative_prompt.prompt_token_ids,
                    1,
                )
                .with_image_pad_token_id(target_image_pad_token_id)
                .with_ordinary_target_prefill_control_span_token_count(
                    representative_prompt.ordinary_target_prefill_control_span_token_count,
                )
                .with_thinking_configuration(false, None, Vec::new(), Vec::new())
                .with_performance_attribution(PerformanceAttribution::enabled()),
            )
            .await
            .expect("the recoverable sparse-target request should be admitted");
        qwen3_5_engine
            .force_next_speculative_prefill_failure_for_tests(
                recoverable_memory_pressure_request_id,
                Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetActiveMemoryLimitRejection,
            )
            .await
            .expect("the recoverable sparse-target memory rejection should arm");
        let mut recovered_prefill_progress_event_count = 0_usize;
        loop {
            match qwen3_5_engine
                .decode_next_token(recoverable_memory_pressure_request_id)
                .await
                .expect("recoverable sparse-target memory pressure must remain inside SpecPrefill")
            {
                GeneratedToken::PrefillProgress {
                    completed_prefill_chunk_tokens,
                    ..
                } => {
                    recovered_prefill_progress_event_count =
                        recovered_prefill_progress_event_count.saturating_add(1);
                    if recovered_prefill_progress_event_count == 1
                        || recovered_prefill_progress_event_count
                            .is_multiple_of(RECOVERY_PROGRESS_LOG_INTERVAL)
                    {
                        eprintln!(
                            "[speculative-prefill-memory-recovery] status=prefill-progress progress_event_count={recovered_prefill_progress_event_count} completed_prefill_chunk_tokens={completed_prefill_chunk_tokens}"
                        );
                    }
                }
                GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
                GeneratedToken::GenerationPreparationStarted { .. } => {}
                GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => break,
            }
        }
        assert!(
            recovered_prefill_progress_event_count > 0,
            "the recovered request must publish successful prompt-processing progress"
        );
        let attribution_report_documents =
            crate::serving_acceptance::support::performance_attribution::read_attribution_report_documents(
                &performance_attribution_log_path,
            );
        let recovery_generation_report =
            crate::serving_acceptance::support::performance_attribution::generation_report_for_request(
                &attribution_report_documents,
                recoverable_memory_pressure_request_id.value(),
            );
        let selected_speculative_prefill_token_count =
            crate::serving_acceptance::support::performance_attribution::counter_amount(
                recovery_generation_report,
                "speculative_prefill_selected_token_count",
            );
        let speculative_prefill_retry_count =
            crate::serving_acceptance::support::performance_attribution::counter_amount(
                recovery_generation_report,
                "prefill_capacity_retry_count",
            );
        let sparse_target_chunk_count =
            crate::serving_acceptance::support::performance_attribution::counter_amount(
                recovery_generation_report,
                "speculative_prefill_sparse_target_chunck_count",
            );
        let speculative_prefill_fallback_count =
            crate::serving_acceptance::support::performance_attribution::counter_amount(
                recovery_generation_report,
                "speculative_prefill_fallback_count",
            );
        assert_eq!(
            speculative_prefill_fallback_count, 0,
            "recoverable sparse-target pressure must not fall back to target-only"
        );
        assert!(
            selected_speculative_prefill_token_count >= 1
                || speculative_prefill_retry_count >= 1
                || sparse_target_chunk_count >= 1,
            "recoverable sparse-target pressure must stay on the SpecPrefill path"
        );
        assert_eq!(
            crate::serving_acceptance::support::performance_attribution::counter_amount(
                recovery_generation_report,
                "speculative_prefill_fallback_count",
            ),
            0,
            "recoverable memory pressure must never invoke target-only prefill"
        );
        eprintln!("[speculative-prefill-memory-recovery] status=success");
    })
    .await
    .expect("the sparse-target memory-recovery journey should finish within 115 seconds");
}
