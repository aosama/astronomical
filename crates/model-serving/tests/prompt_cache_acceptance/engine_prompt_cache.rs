use std::{future::Future, path::Path, time::Duration};

use astronomical_ipc_protocol::{RequestId, WorkerEvent};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PersistentPromptCacheDiskStoreConfig,
    PersistentPromptCacheModelContract, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
    qwen3_5_decoder_cache_layout,
};
use tokio::time::{Instant, MissedTickBehavior, interval, sleep};

use super::large_prefill_prompt::{
    LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT, representative_long_generation_prompt_token_ids,
};

const PERSISTENT_PROMPT_CACHE_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];
const MAXIMUM_OUTPUT_TOKEN_COUNT: usize = 10;

struct PersistentPromptCacheParityAcceptanceOutcome {
    cold_generated_token_ids: Vec<u32>,
    cold_completed_prefill_chunk_token_counts: Vec<u32>,
    restored_cached_token_count: u32,
    restored_generated_token_ids: Vec<u32>,
}

#[tokio::test]
#[ignore = "loads and generates with the complete Ornith artifact"]
async fn should_restore_persistent_prompt_cache_blocks_and_report_cached_tokens_on_the_second_run()
{
    require_persistent_prompt_cache_acceptance_completion(
        run_persistent_prompt_cache_restore_acceptance(),
    )
    .await;
}

async fn run_persistent_prompt_cache_restore_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let acceptance_outcome = run_persistent_prompt_cache_parity_acceptance(
        &model_directory,
        persistent_prompt_cache_eligible_prompt_token_ids(
            persistent_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
                &model_directory,
                2,
            )
            .await,
        ),
        1,
        2_048,
    )
    .await;

    assert_eq!(acceptance_outcome.cold_generated_token_ids.len(), 1);
    assert_eq!(
        acceptance_outcome.cold_generated_token_ids,
        acceptance_outcome.restored_generated_token_ids
    );
}

#[tokio::test]
#[ignore = "loads Ornith and compares cold and restored prompt-cache tokens"]
async fn should_preserve_tokens_after_persistent_prompt_cache_restore() {
    require_persistent_prompt_cache_acceptance_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let acceptance_outcome = run_persistent_prompt_cache_parity_acceptance(
            &model_directory,
            persistent_prompt_cache_eligible_prompt_token_ids(
                persistent_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
                    &model_directory,
                    4,
                )
                .await,
            ),
            MAXIMUM_OUTPUT_TOKEN_COUNT,
            2_048,
        )
        .await;

        assert!(
            !acceptance_outcome.cold_generated_token_ids.is_empty(),
            "the cold Ornith request should emit at least one token before terminal EOS"
        );
        assert_eq!(
            acceptance_outcome.cold_generated_token_ids,
            acceptance_outcome.restored_generated_token_ids
        );
    })
    .await;
}

#[tokio::test]
#[ignore = "runs one fixed 2048, 4096, or 8192 cache-safe prefill acceptance cell"]
async fn should_restore_exact_cache_parity_for_one_selected_large_prefill_size() {
    require_persistent_prompt_cache_acceptance_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let configured_prefill_chunk_tokens = std::env::var(
            "ASTRONOMICAL_PROMPT_CACHE_ACCEPTANCE_PREFILL_CHUNCK_TOKENS",
        )
        .map_or(Ok(8_192), |configured_prefill_chunk_tokens| {
            configured_prefill_chunk_tokens.parse::<u32>()
        })
        .expect("the selected prompt-cache prefill acceptance size should be an integer");
        assert!(
            [2_048, 4_096, 8_192].contains(&configured_prefill_chunk_tokens),
            "the selected prompt-cache prefill acceptance size must be 2048, 4096, or 8192"
        );
        let model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let prompt_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the selected acceptance artifact should validate for prompt preparation");
        let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&prompt_artifact)
            .expect("the selected acceptance tokenizer should load");
        let representative_prompt_token_ids = representative_long_generation_prompt_token_ids(
            &prompt_tokenizer,
            prompt_artifact.model_id(),
        );
        let acceptance_outcome = run_persistent_prompt_cache_parity_acceptance(
            &model_directory,
            representative_prompt_token_ids,
            LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT,
            configured_prefill_chunk_tokens,
        )
        .await;

        assert!(
            acceptance_outcome
                .cold_completed_prefill_chunk_token_counts
                .contains(&configured_prefill_chunk_tokens),
            "the selected prefill size must complete as one model forward"
        );
        assert!(
            acceptance_outcome.restored_cached_token_count
                >= configured_prefill_chunk_tokens,
            "the restored request recovered {} tokens but must recover every 2048-token block produced by one selected {configured_prefill_chunk_tokens}-token forward",
            acceptance_outcome.restored_cached_token_count,
        );
        assert!(
            acceptance_outcome.cold_generated_token_ids.len()
                >= LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT * 85 / 100,
            "the representative acceptance produced only {} of approximately {} requested output tokens",
            acceptance_outcome.cold_generated_token_ids.len(),
            LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT,
        );
        assert_eq!(
            acceptance_outcome.cold_generated_token_ids,
            acceptance_outcome.restored_generated_token_ids,
            "cold and restored deterministic generation must remain identical"
        );
    })
    .await;
}

async fn run_persistent_prompt_cache_parity_acceptance(
    model_directory: &Path,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
    fixed_prefill_chunk_tokens: u32,
) -> PersistentPromptCacheParityAcceptanceOutcome {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");

    // First run: cold prefill, should populate the persistent prompt cache.
    // The exact storage contract now depends on bound affine dtypes. Read block
    // geometry from the loaded engine rather than reconstructing a config-only
    // contract that could disagree with production serialization.
    let (mut qwen3_5_engine, _model_id, _model_revision, prompt_cache_block_token_count) =
        load_persistent_prompt_cache_acceptance_engine(
            model_directory,
            persistent_prompt_cache_directory.path(),
            fixed_prefill_chunk_tokens,
        )
        .await;
    let first_request_id = RequestId::new(2_001);
    let first_generation_start = qwen3_5_engine
        .start_generation(locked_prompt_cache_comparison_request(
            first_request_id,
            prompt_token_ids.clone(),
            generated_token_count,
        ))
        .await
        .expect("the first engine should accept the request");
    assert_eq!(first_generation_start.cached_token_count(), 0);
    let (cold_generated_token_ids, cold_completed_prefill_chunk_token_counts) =
        generate_token_ids(&mut qwen3_5_engine, first_request_id, generated_token_count).await;
    let minimum_expected_sequence_state_block_count = (usize::try_from(fixed_prefill_chunk_tokens)
        .unwrap_or(usize::MAX)
        / prompt_cache_block_token_count)
        .max(1);
    wait_for_persistent_prompt_cache_blocks(
        &qwen3_5_engine,
        minimum_expected_sequence_state_block_count,
    )
    .await;
    // Second run: same prompt, same loaded engine, same prompt-cache directory. The
    // prompt is longer than one persistent prompt-cache block, so the second start
    // must report a hit without paying another full model load in this proof.
    let second_request_id = RequestId::new(2_002);
    let second_generation_start = qwen3_5_engine
        .start_generation(locked_prompt_cache_comparison_request(
            second_request_id,
            prompt_token_ids,
            generated_token_count,
        ))
        .await
        .expect("the second engine should accept the request");
    assert!(
        second_generation_start.cached_token_count() >= prompt_cache_block_token_count as u32,
        "the second run should report at least one restored prompt-cache block"
    );
    let restored_cached_token_count = second_generation_start.cached_token_count();
    let (restored_generated_token_ids, _restored_completed_prefill_chunk_token_counts) =
        generate_token_ids(
            &mut qwen3_5_engine,
            second_request_id,
            generated_token_count,
        )
        .await;
    PersistentPromptCacheParityAcceptanceOutcome {
        cold_generated_token_ids,
        cold_completed_prefill_chunk_token_counts,
        restored_cached_token_count,
        restored_generated_token_ids,
    }
}

pub(super) async fn load_persistent_prompt_cache_acceptance_engine(
    model_directory: &Path,
    persistent_prompt_cache_directory: &Path,
    fixed_prefill_chunk_tokens: u32,
) -> (Qwen3_5Engine, String, String, usize) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before engine loading");
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    let prefill_chunk_sizer =
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            fixed_prefill_chunk_tokens,
        )
        .expect("the selected fixed prefill size should be valid");
    let mlx_memory_limits =
        crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.prompt_cache_block_tokens = Some(fixed_prefill_chunk_tokens);
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.to_path_buf(),
            persistent_prompt_cache_directory.to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        )),
        prefill_chunk_sizer,
        248_069,
        model_directory.to_path_buf(),
        worker_chunking_configuration,
        true,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the engine should accept the prompt-cache directory");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the model");
    let prompt_cache_block_token_count = prompt_cache_block_token_count(&qwen3_5_engine).await;
    (
        qwen3_5_engine,
        model_id,
        model_revision,
        prompt_cache_block_token_count,
    )
}

async fn persistent_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
    model_directory: &Path,
    block_multiplier: usize,
) -> usize {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before sizing the prompt");
    let mlx_memory_limits =
        crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    // This pre-load helper needs only a prompt comfortably beyond a boundary.
    // Use the widest supported state dtype and a caller-supplied multiplier for
    // sizing; hit assertions later use the exact block count reported after load.
    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::resolve(
        validated_artifact.model_id().to_owned(),
        validated_artifact.revision().to_owned(),
        qwen3_5_decoder_cache_layout(
            validated_artifact.config(),
            crate::common::standard_worker_chunking_configuration()
                .full_attention_key_value_growth_tokens as usize,
            &crate::common::qwen3_5_moe::float32_decoder_layer_cache_dtypes(
                validated_artifact.config(),
            ),
        )
        .expect("the validated artifact should provide a decoder-cache layout"),
        validated_artifact.config().maximum_position_count() as usize,
        mlx_memory_limits.active_memory_limit_bytes() as u64,
        crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        crate::common::standard_worker_chunking_configuration()
            .prompt_cache_block_tokens
            .map(|block_token_count| block_token_count as usize),
        crate::common::standard_worker_chunking_configuration()
            .prompt_cache_common_prefix_stride_blocks,
    )
    .expect("the model should resolve a persistent storage contract");
    persistent_prompt_cache_eligible_prompt_token_ids(
        persistent_prompt_cache_model_contract
            .block_token_count()
            .saturating_mul(block_multiplier)
            .saturating_add(16),
    )
    .len()
}

async fn prompt_cache_block_token_count(qwen3_5_engine: &Qwen3_5Engine) -> usize {
    // Stats are emitted from the engine-owned, load-derived cache contract and
    // therefore share the exact geometry used by lookup and publication.
    let cache_stats = qwen3_5_engine
        .collect_persistent_prompt_cache_stats()
        .await
        .expect("the engine should report persistent prompt-cache stats")
        .expect("the acceptance engine should have persistent prompt caching enabled");
    let WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_block_token_count,
        ..
    } = cache_stats
    else {
        panic!("the engine returned an unexpected prompt-cache stats event")
    };
    usize::try_from(persistent_prompt_cache_block_token_count)
        .expect("the prompt-cache block token count should fit usize")
}
pub(super) async fn wait_for_persistent_prompt_cache_blocks(
    qwen3_5_engine: &Qwen3_5Engine,
    expected_sequence_state_block_count: usize,
) {
    let publication_started_at = Instant::now();
    loop {
        let cache_stats = qwen3_5_engine
            .collect_persistent_prompt_cache_stats()
            .await
            .expect("the engine should report persistent prompt-cache stats")
            .expect("the acceptance engine should have persistent prompt caching enabled");
        let WorkerEvent::PersistentPromptCacheStats {
            persistent_prompt_cache_block_token_count,
            persistent_prompt_cache_sequence_state_block_count,
            persistent_prompt_cache_boundary_state_snapshot_count,
            ..
        } = cache_stats
        else {
            panic!("the engine returned an unexpected prompt-cache stats event")
        };
        if persistent_prompt_cache_sequence_state_block_count
            >= u64::try_from(expected_sequence_state_block_count).unwrap_or(u64::MAX)
        {
            eprintln!(
                "[persistent-prompt-cache-acceptance] status=cache-ready sequence_blocks={persistent_prompt_cache_sequence_state_block_count} boundary_snapshots={persistent_prompt_cache_boundary_state_snapshot_count}"
            );
            return;
        }
        assert!(
            publication_started_at.elapsed() < Duration::from_secs(10),
            "only {persistent_prompt_cache_sequence_state_block_count} of {expected_sequence_state_block_count} expected prompt-cache blocks were published"
        );
        eprintln!(
            "[persistent-prompt-cache-acceptance] status=waiting-for-cache block_tokens={persistent_prompt_cache_block_token_count} published_blocks={persistent_prompt_cache_sequence_state_block_count} expected_blocks={expected_sequence_state_block_count}"
        );
        sleep(Duration::from_millis(250)).await;
    }
}

pub(super) async fn generate_token_ids(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    generated_token_count: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut generated_token_ids = Vec::new();
    let mut completed_prefill_chunk_token_counts = Vec::new();
    loop {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the engine should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => {
                generated_token_ids.push(token_id);
                if generation_finalization.is_some()
                    || generated_token_ids.len() == generated_token_count
                {
                    return (generated_token_ids, completed_prefill_chunk_token_counts);
                }
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunk_tokens,
                ..
            } => completed_prefill_chunk_token_counts.push(completed_prefill_chunk_tokens),
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                return (generated_token_ids, completed_prefill_chunk_token_counts);
            }
        }
    }
}

pub(super) fn persistent_prompt_cache_eligible_prompt_token_ids(
    target_token_count: usize,
) -> Vec<u32> {
    let mut prompt_token_ids = Vec::with_capacity(target_token_count);
    while prompt_token_ids.len() < target_token_count {
        prompt_token_ids.extend_from_slice(&SAY_HI_PROMPT_TOKEN_IDS);
    }
    prompt_token_ids.truncate(target_token_count);
    prompt_token_ids
}

pub(super) async fn require_persistent_prompt_cache_acceptance_completion(
    test_future: impl Future<Output = ()>,
) {
    // The pinned artifact is intentionally large, but a local acceptance
    // must never leave the laptop compiling or evaluating indefinitely.
    let started_at = Instant::now();
    let timeout_deadline = sleep(PERSISTENT_PROMPT_CACHE_ACCEPTANCE_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    eprintln!(
        "[persistent-prompt-cache-acceptance] status=start timeout_seconds={}",
        PERSISTENT_PROMPT_CACHE_ACCEPTANCE_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut test_future => {
                eprintln!(
                    "[persistent-prompt-cache-acceptance] status=success elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!(
                    "the persistent prompt-cache acceptance exceeded {} seconds",
                    PERSISTENT_PROMPT_CACHE_ACCEPTANCE_TIMEOUT.as_secs()
                );
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = PERSISTENT_PROMPT_CACHE_ACCEPTANCE_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[persistent-prompt-cache-acceptance] status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}

fn locked_prompt_cache_comparison_request(
    request_id: RequestId,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
) -> Qwen3_5InferenceRequest {
    Qwen3_5InferenceRequest::new_sampling(
        request_id,
        prompt_token_ids,
        u16::try_from(generated_token_count)
            .expect("the generated token count should fit the engine request"),
        0,
        1_000,
        None,
    )
    .with_image_pad_token_id(248_069)
    .with_thinking_configuration(false, None, Vec::new(), Vec::new())
}
