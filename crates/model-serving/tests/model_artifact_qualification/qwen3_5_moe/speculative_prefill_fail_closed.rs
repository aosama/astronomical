use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_ipc_protocol::{RequestId, WorkerSpeculativePrefillConfiguration};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionLog, PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer,
    Qwen3_5SpeculativePrefillFailureStageForTests, Qwen3_5Tokenizer,
};

use super::speculative_prefill::{
    SPECULATIVE_PREFILL_KEEP_PERCENTAGE, prepare_representative_prompt,
};

const FORCED_FAILURE_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const RECOVERY_PROGRESS_LOG_INTERVAL: usize = 64;
const KEEP_EVERY_SPECULATIVE_PREFILL_TOKEN_PERCENTAGE: u32 = 100;

#[tokio::test]
#[ignore = "loads the target and proves an unavailable configured drafter stops model activation"]
async fn should_stop_model_activation_when_the_configured_drafter_is_unavailable() {
    tokio::time::timeout(FORCED_FAILURE_QUALIFICATION_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 1)
            .expect("the unavailable-drafter target artifact should validate");
        let target_model_id = validated_target_artifact.model_id().to_owned();
        let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the unavailable-drafter target tokenizer should load");
        let unavailable_draft_directory = tempfile::tempdir()
            .expect("the unavailable-drafter journey should create an empty draft directory");
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_target_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(32)
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
    tokio::time::timeout(FORCED_FAILURE_QUALIFICATION_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 1)
            .expect("the memory-recovery target artifact should validate");
        let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the memory-recovery target tokenizer should load");
        let target_model_id = validated_target_artifact.model_id().to_owned();
        let target_image_pad_token_id = target_tokenizer.image_pad_token_id();
        let representative_prompt = prepare_representative_prompt(&target_model_directory);
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let attribution_directory = tempfile::tempdir()
            .expect("the memory-recovery qualification should create an attribution directory");
        let performance_attribution_log_path = attribution_directory
            .path()
            .join("performance-attribution.jsonl");
        let persistent_prompt_cache_directory = tempfile::tempdir()
            .expect("the memory-recovery qualification should create an SSD cache directory");
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
            Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(32)
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
                    completed_prefill_chunck_tokens,
                    ..
                } => {
                    recovered_prefill_progress_event_count =
                        recovered_prefill_progress_event_count.saturating_add(1);
                    if recovered_prefill_progress_event_count == 1
                        || recovered_prefill_progress_event_count
                            .is_multiple_of(RECOVERY_PROGRESS_LOG_INTERVAL)
                    {
                        eprintln!(
                            "[speculative-prefill-memory-recovery] status=prefill-progress progress_event_count={recovered_prefill_progress_event_count} completed_prefill_chunck_tokens={completed_prefill_chunck_tokens}"
                        );
                    }
                }
                GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
                GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => break,
            }
        }
        assert!(
            recovered_prefill_progress_event_count > 0,
            "the recovered request must publish successful prompt-processing progress"
        );
        let attribution_report_documents =
            super::performance_attribution::read_attribution_report_documents(
                &performance_attribution_log_path,
            );
        let recovery_generation_report =
            super::performance_attribution::generation_report_for_request(
                &attribution_report_documents,
                recoverable_memory_pressure_request_id.value(),
            );
        for (counter_identifier, expected_counter_description) in [
            (
                "prefill_capacity_rejection_count",
                "recoverable sparse-target allocation rejection",
            ),
            ("prefill_capacity_retry_count", "bounded prefill retry"),
            (
                "speculative_prefill_sparse_target_chunck_count",
                "completed sparse target chunk",
            ),
            (
                "speculative_prefill_selected_token_count",
                "completed selected target position",
            ),
        ] {
            assert!(
                super::performance_attribution::counter_amount(
                    recovery_generation_report,
                    counter_identifier,
                ) >= 1,
                "the recovered request must attribute at least one {expected_counter_description}"
            );
        }
        assert_eq!(
            super::performance_attribution::counter_amount(
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

#[tokio::test]
#[ignore = "loads the configured target and drafter and forces every configured SpecPrefill execution stage to fail closed"]
async fn should_stop_every_forced_speculative_prefill_execution_stage_without_target_only_retry() {
    tokio::time::timeout(FORCED_FAILURE_QUALIFICATION_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 1)
            .expect("the fail-closed target artifact should validate");
        let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the fail-closed target tokenizer should load");
        let target_image_pad_token_id = target_tokenizer.image_pad_token_id();
        let representative_prompt = prepare_representative_prompt(&target_model_directory);
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let attribution_directory = tempfile::tempdir()
            .expect("the fail-closed qualification should create an attribution directory");
        let persistent_prompt_cache_directory = tempfile::tempdir()
            .expect("the fail-closed qualification should create an SSD cache directory");
        let persistent_prompt_cache_config = PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().join("target"),
            persistent_prompt_cache_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );
        let speculative_prefill_configuration = WorkerSpeculativePrefillConfiguration {
            enabled: true,
            target_model_id: Some(validated_target_artifact.model_id().to_owned()),
            draft_model_id: Some(draft_model_id),
            draft_model_directory: Some(draft_model_directory),
            minimum_prompt_tokens: 8_192,
            keep_percentage: KEEP_EVERY_SPECULATIVE_PREFILL_TOKEN_PERCENTAGE,
            selection_chunck_token_count: 32,
            mandatory_trailing_token_count: 512,
            lookahead_token_count: 8,
            importance_pooling_kernel_token_count: 13,
        };
        let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_target_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            Some(persistent_prompt_cache_config),
            Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(32)
                .expect("the fail-closed prefill chunk size should be valid"),
            target_tokenizer.think_end_token_id(),
            target_model_directory,
            crate::common::standard_worker_chunking_configuration(),
            true,
            false,
            speculative_prefill_configuration,
            PerformanceAttribution::enabled(),
            PerformanceAttributionLog::open(
                &attribution_directory.path().join("performance-attribution.jsonl"),
                true,
            )
            .expect("the fail-closed attribution log should open"),
        )
        .expect("the fail-closed engine should construct");
        qwen3_5_engine
            .load()
            .await
            .expect("the configured target and drafter should load before forced failures");

        let baseline_request_id = RequestId::new(95_499);
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(
                    baseline_request_id,
                    representative_prompt.prompt_token_ids.clone(),
                    1,
                )
                .with_image_pad_token_id(target_image_pad_token_id)
                .with_ordinary_target_prefill_control_span_token_count(
                    representative_prompt.ordinary_target_prefill_control_span_token_count,
                ),
            )
            .await
            .expect("the valid baseline request should be admitted");
        loop {
            match qwen3_5_engine
                .decode_next_token(baseline_request_id)
                .await
                .expect("the valid baseline request should complete")
            {
                GeneratedToken::PrefillProgress { .. }
                | GeneratedToken::PromptProcessingPhaseStarted { .. } => continue,
                GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => break,
            }
        }
        let valid_persisted_speculative_prefill_files =
            persisted_speculative_prefill_file_contents(persistent_prompt_cache_directory.path());
        assert!(
            !valid_persisted_speculative_prefill_files.is_empty(),
            "the 100-percent keep baseline must complete and publish selection and target state before forced failures",
        );

        for (failure_number, failure_stage, expected_failure_stage_text) in [
            (
                0_u64,
                Qwen3_5SpeculativePrefillFailureStageForTests::DrafterLoading,
                "drafter loading",
            ),
            (
                1,
                Qwen3_5SpeculativePrefillFailureStageForTests::DraftScoring,
                "draft scoring or selection",
            ),
            (
                2,
                Qwen3_5SpeculativePrefillFailureStageForTests::Selection,
                "draft scoring or selection",
            ),
            (
                3,
                Qwen3_5SpeculativePrefillFailureStageForTests::DrafterPromptStatePersistence,
                "drafter prompt-state persistence",
            ),
            (
                4,
                Qwen3_5SpeculativePrefillFailureStageForTests::SelectionPersistence,
                "selection persistence",
            ),
            (
                5,
                Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetInputAssembly,
                "sparse target input assembly",
            ),
            (
                6,
                Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetExecution,
                "sparse target execution",
            ),
            (
                7,
                Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetStatePersistence,
                "sparse target-state persistence",
            ),
        ] {
            let request_id = RequestId::new(95_500 + failure_number);
            let mut failure_prompt_token_ids = representative_prompt.prompt_token_ids.clone();
            let changed_conversation_token_position = failure_prompt_token_ids.len() - 2;
            failure_prompt_token_ids[changed_conversation_token_position] =
                representative_prompt.prompt_token_ids[0];
            qwen3_5_engine
                .start_generation(
                    Qwen3_5InferenceRequest::new(
                        request_id,
                        failure_prompt_token_ids,
                        1,
                    )
                    .with_image_pad_token_id(target_image_pad_token_id)
                    .with_ordinary_target_prefill_control_span_token_count(
                        representative_prompt.ordinary_target_prefill_control_span_token_count,
                    ),
                )
                .await
                .expect("the forced-failure request should be admitted before its armed stage");
            qwen3_5_engine
                .force_next_speculative_prefill_failure_for_tests(request_id, failure_stage)
                .await
                .expect("the requested SpecPrefill failure stage should arm");

            let generation_error = loop {
                match qwen3_5_engine.decode_next_token(request_id).await {
                    Err(generation_error) => break generation_error,
                    Ok(GeneratedToken::PrefillProgress { .. }) => continue,
                    Ok(successful_generation_event) => panic!(
                        "the armed SpecPrefill stage must stop before successful generation: {successful_generation_event:?}"
                    ),
                }
            };
            assert!(matches!(
                generation_error,
                InferenceEngineError::InvalidRequest { ref reason }
                    if reason.contains(expected_failure_stage_text)
                        && reason.contains("request was stopped")
                        && reason.contains("without a target-only retry")
            ));
            qwen3_5_engine
                .cancel_generation(request_id)
                .await
                .expect("the failed request should release its active engine state");
            for (persisted_file_path, expected_file_contents) in
                &valid_persisted_speculative_prefill_files
            {
                assert_eq!(
                    std::fs::read(persisted_file_path)
                        .expect("previously valid SpecPrefill state should remain readable"),
                    *expected_file_contents,
                    "a forced request failure must not modify previously valid persisted state",
                );
            }
            eprintln!(
                "[speculative-prefill-fail-closed] status=stopped stage={expected_failure_stage_text}"
            );
        }
    })
    .await
    .expect("the complete forced SpecPrefill failure matrix should finish within 115 seconds");
}

fn persisted_speculative_prefill_file_contents(
    prompt_cache_root_directory: &Path,
) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut persisted_file_path_to_contents = BTreeMap::new();
    collect_persisted_speculative_prefill_file_contents(
        prompt_cache_root_directory,
        false,
        &mut persisted_file_path_to_contents,
    );
    persisted_file_path_to_contents
}

fn collect_persisted_speculative_prefill_file_contents(
    current_path: &Path,
    is_inside_speculative_prefill_namespace: bool,
    persisted_file_path_to_contents: &mut BTreeMap<PathBuf, Vec<u8>>,
) {
    if current_path.is_file() {
        if is_inside_speculative_prefill_namespace {
            persisted_file_path_to_contents.insert(
                current_path.to_path_buf(),
                std::fs::read(current_path)
                    .expect("valid persisted SpecPrefill state should remain readable"),
            );
        }
        return;
    }
    let current_path_enters_speculative_prefill_namespace = is_inside_speculative_prefill_namespace
        || current_path.file_name().is_some_and(|directory_name| {
            directory_name == "speculative_prefill_selections"
                || directory_name == "speculative_prefill_target_states"
        });
    for directory_entry in std::fs::read_dir(current_path)
        .expect("the persisted SpecPrefill cache tree should remain readable")
    {
        collect_persisted_speculative_prefill_file_contents(
            &directory_entry
                .expect("the persisted SpecPrefill cache entry should remain readable")
                .path(),
            current_path_enters_speculative_prefill_namespace,
            persisted_file_path_to_contents,
        );
    }
}
