use std::path::Path;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ExpertMemoryMode, RequestId, WorkerPersistentPromptCacheLookupOutcome,
};
use astronomical_model_serving::{
    InferenceEngine, PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheModelContract,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer, qwen3_5_decoder_cache_layout,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::{Instant, MissedTickBehavior, interval, sleep};

use super::large_prefill_prompt::representative_long_generation_prompt_token_ids;

const CACHE_MISS_MEMORY_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const MEMORY_ACCEPTANCE_PREFILL_CHUNK_TOKENS: u32 = 2_048;

#[tokio::test]
#[ignore = "compares expert reclamation for an empty prompt cache with cache-disabled admission"]
async fn should_not_reclaim_more_expert_payload_for_a_cache_miss_than_without_cache() {
    require_cache_miss_memory_acceptance_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the model artifact should validate before preparing the source prompt");
        let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
            .expect("the model tokenizer should load before preparing the source prompt");
        let mut prompt_token_ids = representative_long_generation_prompt_token_ids(
            &prompt_tokenizer,
            validated_artifact.model_id(),
        );
        let maximum_prompt_token_count =
            usize::try_from(validated_artifact.config().maximum_position_count())
                .expect("the model context limit should fit usize")
                .saturating_sub(1);
        repeat_fixture_prompt_to_token_count(&mut prompt_token_ids, maximum_prompt_token_count);
        let mlx_memory_limits =
            crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
        let total_context_token_count = prompt_token_ids
            .len()
            .checked_add(1)
            .expect("the prompt and one output token should fit usize");
        let context_growth_bytes = validated_artifact
            .config()
            .context_memory_reservation_bytes(total_context_token_count)
            .expect("the model context growth should fit usize");
        let cache_specific_workspace_bytes = cache_specific_workspace_bytes(
            &validated_artifact,
            mlx_memory_limits,
            context_growth_bytes,
        );
        let (
            cacheless_reclaimed_expert_payload_bytes,
            shared_test_memory_ceiling_bytes,
            cacheless_expert_memory_mode_after_admission,
        ) = measure_initial_request_expert_reclamation(
            &model_directory,
            None,
            &prompt_token_ids,
            41_101,
            mlx_memory_limits,
            context_growth_bytes,
            cache_specific_workspace_bytes,
            None,
        )
        .await;
        if shared_test_memory_ceiling_bytes.is_some() {
            assert_eq!(
                cacheless_expert_memory_mode_after_admission,
                ExpertMemoryMode::Resident,
                "the dynamically selected cacheless control ceiling should preserve residency"
            );
        }

        let empty_cache_directory =
            tempfile::tempdir().expect("the cache-miss run should create an empty cache directory");
        let (
            cache_miss_reclaimed_expert_payload_bytes,
            cache_miss_memory_ceiling_bytes,
            cache_miss_expert_memory_mode_after_admission,
        ) = measure_initial_request_expert_reclamation(
            &model_directory,
            Some(empty_cache_directory.path()),
            &prompt_token_ids,
            41_102,
            mlx_memory_limits,
            context_growth_bytes,
            cache_specific_workspace_bytes,
            shared_test_memory_ceiling_bytes,
        )
        .await;
        assert_eq!(
            cache_miss_memory_ceiling_bytes, shared_test_memory_ceiling_bytes,
            "the cache-enabled and cache-disabled admissions must use the same ceiling"
        );
        if shared_test_memory_ceiling_bytes.is_some() {
            assert_eq!(
                cache_miss_expert_memory_mode_after_admission,
                ExpertMemoryMode::Resident,
                "a cache miss should not demote experts when the cacheless request fits"
            );
        }

        assert!(
            cache_miss_reclaimed_expert_payload_bytes <= cacheless_reclaimed_expert_payload_bytes,
            "an empty persistent cache must not cause extra expert reclamation: \
             cache_miss={cache_miss_reclaimed_expert_payload_bytes} \
             cacheless={cacheless_reclaimed_expert_payload_bytes}"
        );
    })
    .await;
}

async fn measure_initial_request_expert_reclamation(
    model_directory: &Path,
    persistent_prompt_cache_directory: Option<&Path>,
    prompt_token_ids: &[u32],
    request_id: u64,
    mlx_memory_limits: MlxMemoryLimits,
    context_growth_bytes: usize,
    cache_specific_workspace_bytes: usize,
    requested_test_memory_ceiling_bytes: Option<usize>,
) -> (u64, Option<usize>, ExpertMemoryMode) {
    let mut qwen3_5_engine = load_memory_acceptance_engine(
        model_directory,
        persistent_prompt_cache_directory,
        mlx_memory_limits,
    )
    .await;
    let expert_memory_mode_before_admission = qwen3_5_engine
        .expert_memory_mode_for_tests()
        .await
        .expect("the engine should report expert memory mode")
        .expect("the loaded model should report expert memory mode");
    let minimum_safe_memory_ceiling_bytes =
        if expert_memory_mode_before_admission == ExpertMemoryMode::Resident {
            Some(
                usize::try_from(
                    qwen3_5_engine
                        .update_mlx_memory_limit(
                            u64::try_from(mlx_memory_limits.active_memory_limit_bytes())
                                .expect("the machine ceiling should fit u64"),
                        )
                        .await
                        .expect("the engine should retain the machine memory ceiling")
                        .minimum_mlx_memory_ceiling_bytes(),
                )
                .expect("the minimum safe ceiling should fit usize"),
            )
        } else {
            None
        };
    let active_memory_bytes_before_admission = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the model acceptance should reuse the engine's MLX runtime")
        .memory_snapshot()
        .expect("the model acceptance should sample active memory")
        .active_memory_bytes();
    let target_memory_ceiling_bytes = requested_test_memory_ceiling_bytes.or_else(|| {
        if expert_memory_mode_before_admission != ExpertMemoryMode::Resident
            || cache_specific_workspace_bytes == 0
        {
            return None;
        }
        active_memory_bytes_before_admission
            .checked_add(context_growth_bytes)?
            .checked_add(cache_specific_workspace_bytes / 2)
            .filter(|candidate_ceiling_bytes| {
                *candidate_ceiling_bytes > active_memory_bytes_before_admission
                    && *candidate_ceiling_bytes < mlx_memory_limits.active_memory_limit_bytes()
                    && minimum_safe_memory_ceiling_bytes
                        .is_some_and(|minimum| *candidate_ceiling_bytes >= minimum)
            })
    });
    let mut post_adjustment_memory_limits = None;
    let mut minimum_safe_memory_ceiling_bytes = minimum_safe_memory_ceiling_bytes;
    let applied_memory_ceiling_bytes =
        if let Some(target_memory_ceiling_bytes) = target_memory_ceiling_bytes {
            let memory_limit_adjustment = qwen3_5_engine
                .update_mlx_memory_limit(
                    u64::try_from(target_memory_ceiling_bytes)
                        .expect("the selected acceptance ceiling should fit u64"),
                )
                .await
                .expect("the engine should apply a ceiling above its current active memory");
            minimum_safe_memory_ceiling_bytes = Some(
                usize::try_from(memory_limit_adjustment.minimum_mlx_memory_ceiling_bytes())
                    .expect("the minimum safe ceiling should fit usize"),
            );
            post_adjustment_memory_limits = Some(
                MlxMemoryLimits::new(
                    usize::try_from(memory_limit_adjustment.effective_mlx_memory_ceiling_bytes())
                        .expect("the effective acceptance ceiling should fit usize"),
                    usize::try_from(memory_limit_adjustment.allocator_cache_memory_limit_bytes())
                        .expect("the allocator cache limit should fit usize"),
                )
                .expect("the applied acceptance limits should be valid"),
            );
            Some(
                usize::try_from(memory_limit_adjustment.effective_mlx_memory_ceiling_bytes())
                    .expect("the effective acceptance ceiling should fit usize"),
            )
        } else {
            None
        };
    let expert_memory_mode_before_request = qwen3_5_engine
        .expert_memory_mode_for_tests()
        .await
        .expect("the engine should report expert memory mode before request admission")
        .expect("the loaded model should retain an expert memory mode");
    let expert_payload_bytes_before_start =
        measured_expert_payload_bytes(&qwen3_5_engine, expert_memory_mode_before_request).await;
    let generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(RequestId::new(request_id), prompt_token_ids.to_vec(), 1)
                .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the request should start before prompt processing");
    if persistent_prompt_cache_directory.is_some() {
        let persistent_prompt_cache_diagnostics = generation_start
            .persistent_prompt_cache_diagnostics()
            .expect("the enabled empty prompt cache should report its miss");
        assert_eq!(
            persistent_prompt_cache_diagnostics.lookup_outcome,
            WorkerPersistentPromptCacheLookupOutcome::Miss
        );
    }
    let expert_memory_mode_after_admission = qwen3_5_engine
        .expert_memory_mode_for_tests()
        .await
        .expect("the engine should report expert memory mode after admission")
        .expect("the loaded model should retain an expert memory mode");
    let expert_payload_bytes_after_start =
        measured_expert_payload_bytes(&qwen3_5_engine, expert_memory_mode_after_admission).await;
    let reclaimed_expert_payload_bytes =
        expert_payload_bytes_before_start.saturating_sub(expert_payload_bytes_after_start);
    eprintln!(
        "[prompt-cache-miss-memory] request_id={request_id} cache_enabled={} mode_before_limit={expert_memory_mode_before_admission:?} mode_before_request={expert_memory_mode_before_request:?} mode_after={expert_memory_mode_after_admission:?} active_before_bytes={active_memory_bytes_before_admission} context_growth_bytes={context_growth_bytes} cache_extra_workspace_bytes={cache_specific_workspace_bytes} minimum_safe_ceiling_bytes={minimum_safe_memory_ceiling_bytes:?} ceiling_bytes={applied_memory_ceiling_bytes:?} payload_before_bytes={expert_payload_bytes_before_start} payload_after_bytes={expert_payload_bytes_after_start} reclaimed_bytes={reclaimed_expert_payload_bytes}",
        persistent_prompt_cache_directory.is_some(),
    );
    drop(qwen3_5_engine);
    if let Some(current_memory_limits) = post_adjustment_memory_limits {
        let mut mlx_runtime = MlxRuntime::initialize(current_memory_limits)
            .expect("the runtime should match the applied acceptance memory ceiling");
        mlx_runtime
            .update_memory_limits(mlx_memory_limits)
            .expect("the acceptance should restore the machine memory ceiling after measurement");
    }

    (
        reclaimed_expert_payload_bytes,
        applied_memory_ceiling_bytes,
        expert_memory_mode_after_admission,
    )
}

fn repeat_fixture_prompt_to_token_count(
    prompt_token_ids: &mut Vec<u32>,
    target_token_count: usize,
) {
    let fixture_prompt_token_ids = prompt_token_ids.clone();
    while prompt_token_ids.len() < target_token_count {
        prompt_token_ids.extend_from_slice(&fixture_prompt_token_ids);
    }
    prompt_token_ids.truncate(target_token_count);
}

async fn measured_expert_payload_bytes(
    qwen3_5_engine: &Qwen3_5Engine,
    expert_memory_mode: ExpertMemoryMode,
) -> u64 {
    if expert_memory_mode == ExpertMemoryMode::Resident {
        qwen3_5_engine
            .complete_expert_payload_byte_count_for_tests()
            .await
            .expect("the resident model should report its complete expert payload")
    } else {
        qwen3_5_engine
            .expert_weight_memory_cache_statistics_for_tests()
            .await
            .expect("the engine should report paged expert payload")
            .resident_payload_byte_count
    }
}

async fn load_memory_acceptance_engine(
    model_directory: &Path,
    persistent_prompt_cache_directory: Option<&Path>,
    mlx_memory_limits: MlxMemoryLimits,
) -> Qwen3_5Engine {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before engine loading");
    let prefill_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            MEMORY_ACCEPTANCE_PREFILL_CHUNK_TOKENS,
        )
        .expect("the selected fixed prefill size should be valid");
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.prompt_cache_block_tokens =
        Some(MEMORY_ACCEPTANCE_PREFILL_CHUNK_TOKENS);
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        persistent_prompt_cache_directory.map(|cache_directory| {
            PersistentPromptCacheDiskStoreConfig::new(
                cache_directory.to_path_buf(),
                cache_directory.to_path_buf(),
                crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
            )
        }),
        prefill_chunk_sizer,
        248_069,
        model_directory.to_path_buf(),
        worker_chunking_configuration,
        true,
    )
    .expect("the engine should accept the prompt-cache configuration");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the model");
    qwen3_5_engine
}

fn cache_specific_workspace_bytes(
    validated_artifact: &astronomical_model_serving::ValidatedQwen3_5Artifact,
    mlx_memory_limits: MlxMemoryLimits,
    context_growth_bytes: usize,
) -> usize {
    let worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::resolve(
        validated_artifact.model_id().to_owned(),
        validated_artifact.revision().to_owned(),
        qwen3_5_decoder_cache_layout(
            validated_artifact.config(),
            worker_chunking_configuration.full_attention_key_value_growth_tokens as usize,
            &crate::common::qwen3_5_moe::float32_decoder_layer_cache_dtypes(
                validated_artifact.config(),
            ),
        )
        .expect("the validated artifact should provide a decoder-cache layout"),
        validated_artifact.config().maximum_position_count() as usize,
        mlx_memory_limits.active_memory_limit_bytes() as u64,
        crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        Some(MEMORY_ACCEPTANCE_PREFILL_CHUNK_TOKENS as usize),
        worker_chunking_configuration.prompt_cache_common_prefix_stride_blocks,
    )
    .expect("the model should resolve a persistent storage contract");
    let context_memory_reservation_bytes_per_token = validated_artifact
        .config()
        .context_memory_reservation_bytes(1)
        .expect("the model should reserve context memory per token");
    let restore_workspace_bytes =
        astronomical_model_serving::persistent_context_restore_workspace_bytes(
            context_memory_reservation_bytes_per_token,
            persistent_prompt_cache_model_contract.block_token_count(),
        )
        .expect("the one-block restore workspace should fit usize");
    persistent_prompt_cache_model_contract
        .direct_publication_workspace_bytes()
        .checked_add(restore_workspace_bytes.saturating_sub(context_growth_bytes))
        .expect("cache-specific workspace should fit usize")
}

async fn require_cache_miss_memory_acceptance_completion(
    acceptance_journey: impl std::future::Future<Output = ()>,
) {
    let started_at = Instant::now();
    let deadline = sleep(CACHE_MISS_MEMORY_ACCEPTANCE_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(acceptance_journey);
    tokio::pin!(deadline);
    progress_interval.tick().await;
    eprintln!(
        "[prompt-cache-miss-memory] status=start timeout_seconds={}",
        CACHE_MISS_MEMORY_ACCEPTANCE_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut acceptance_journey => {
                eprintln!(
                    "[prompt-cache-miss-memory] status=success elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut deadline => {
                panic!(
                    "the cache-miss memory acceptance exceeded {} seconds",
                    CACHE_MISS_MEMORY_ACCEPTANCE_TIMEOUT.as_secs()
                );
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = CACHE_MISS_MEMORY_ACCEPTANCE_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[prompt-cache-miss-memory] status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(), remaining.as_secs_f64()
                );
            }
        }
    }
}
