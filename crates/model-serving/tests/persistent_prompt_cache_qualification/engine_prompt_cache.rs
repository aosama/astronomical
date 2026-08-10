use std::{future::Future, path::Path, time::Duration};

use astronomical_ipc_protocol::{RequestId, WorkerEvent};
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PersistentPromptCacheDiskStoreConfig, PersistentPromptCacheModelContract,
    PersistentPromptCachePrefixLookup, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
    qwen3_5_decoder_cache_layout,
};
use tokio::time::{Instant, MissedTickBehavior, interval, sleep};

use super::large_prefill_prompt::{
    LARGE_PREFILL_QUALIFICATION_OUTPUT_TOKEN_COUNT, representative_long_generation_prompt_token_ids,
};

const PERSISTENT_PROMPT_CACHE_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];
const MAXIMUM_GREEDY_OUTPUT_TOKEN_COUNT: usize = 10;

struct PersistentPromptCacheParityQualificationOutcome {
    cold_generated_token_ids: Vec<u32>,
    cold_completed_prefill_chunck_token_counts: Vec<u32>,
    restored_cached_token_count: u32,
    restored_generated_token_ids: Vec<u32>,
}

#[tokio::test]
#[ignore = "loads and generates with the complete Ornith artifact"]
async fn should_generate_identical_output_with_prompt_cache_disabled_and_cold_prefill() {
    require_persistent_prompt_cache_qualification_completion(
        run_prompt_cache_disabled_cold_prefill_qualification(),
    )
    .await;
}

async fn run_prompt_cache_disabled_cold_prefill_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate before engine loading");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(16)
            .expect("the test prefill_chunck_tokens should be valid"),
        248_069,
        model_directory.to_path_buf(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
    )
    .expect("the bounded Ornith engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the complete Ornith model");
    let request_id = RequestId::new(2_000);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(request_id, SAY_HI_PROMPT_TOKEN_IDS.to_vec(), 10)
                .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the engine should accept one greedy generation request");

    let mut generated_token_ids = Vec::new();
    while generated_token_ids.len() < 10 {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each engine boundary should advance the request")
        {
            GeneratedToken::TokenId { token_id, .. } => generated_token_ids.push(token_id),
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }

    assert_eq!(
        generated_token_ids,
        vec![12_675, 0, 2_500, 628, 353, 1_438, 488, 3_242, 30, 248_046]
    );
}

#[tokio::test]
#[ignore = "loads and generates with the complete Ornith artifact"]
async fn should_restore_persistent_prompt_cache_blocks_and_report_cached_tokens_on_the_second_run()
{
    require_persistent_prompt_cache_qualification_completion(
        run_persistent_prompt_cache_restore_qualification(),
    )
    .await;
}

async fn run_persistent_prompt_cache_restore_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let qualification_outcome = run_persistent_prompt_cache_greedy_parity_qualification(
        &model_directory,
        persistent_prompt_cache_eligible_prompt_token_ids(
            persistent_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
                &model_directory,
                1,
            )
            .await,
        ),
        1,
        None,
    )
    .await;

    assert_eq!(qualification_outcome.cold_generated_token_ids.len(), 1);
    assert_eq!(
        qualification_outcome.cold_generated_token_ids,
        qualification_outcome.restored_generated_token_ids
    );
}

#[tokio::test]
#[ignore = "loads Ornith and compares cold and restored prompt-cache greedy tokens"]
async fn should_preserve_ornith_greedy_tokens_after_persistent_prompt_cache_restore() {
    require_persistent_prompt_cache_qualification_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_ornith_model_artifact_directory();
        let qualification_outcome = run_persistent_prompt_cache_greedy_parity_qualification(
            &model_directory,
            persistent_prompt_cache_eligible_prompt_token_ids(
                persistent_prompt_cache_eligible_prompt_token_count_for_block_multiplier(
                    &model_directory,
                    4,
                )
                .await,
            ),
            MAXIMUM_GREEDY_OUTPUT_TOKEN_COUNT,
            None,
        )
        .await;

        assert!(
            !qualification_outcome.cold_generated_token_ids.is_empty(),
            "the cold Ornith request should emit at least one greedy token before terminal EOS"
        );
        assert_eq!(
            qualification_outcome.cold_generated_token_ids,
            qualification_outcome.restored_generated_token_ids
        );
    })
    .await;
}

#[tokio::test]
#[ignore = "runs one fixed 2048, 4096, or 8192 cache-safe prefill qualification cell"]
async fn should_qualify_one_selected_large_prefill_size_with_exact_cache_restore_parity() {
    require_persistent_prompt_cache_qualification_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let configured_prefill_chunck_tokens = std::env::var(
            "ASTRONOMICAL_PROMPT_CACHE_QUALIFICATION_PREFILL_CHUNCK_TOKENS",
        )
        .map_or(Ok(8_192), |configured_prefill_chunck_tokens| {
            configured_prefill_chunck_tokens.parse::<u32>()
        })
        .expect("the selected prompt-cache prefill qualification size should be an integer");
        assert!(
            [2_048, 4_096, 8_192].contains(&configured_prefill_chunck_tokens),
            "the selected prompt-cache prefill qualification size must be 2048, 4096, or 8192"
        );
        let model_directory = std::env::var(
            "ASTRONOMICAL_PROMPT_CACHE_QUALIFICATION_MODEL_ID",
        )
        .map_or_else(
            |_| crate::common::configured_ornith_model_artifact_directory(),
            |configured_model_id| {
                crate::common::configured_model_artifact_directory_by_id(&configured_model_id)
            },
        );
        let prompt_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the selected qualification artifact should validate for prompt preparation");
        let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&prompt_artifact)
            .expect("the selected qualification tokenizer should load");
        let representative_prompt_token_ids = representative_long_generation_prompt_token_ids(
            &prompt_tokenizer,
            prompt_artifact.model_id(),
        );
        let qualification_outcome = run_persistent_prompt_cache_greedy_parity_qualification(
            &model_directory,
            representative_prompt_token_ids,
            LARGE_PREFILL_QUALIFICATION_OUTPUT_TOKEN_COUNT,
            Some(configured_prefill_chunck_tokens),
        )
        .await;

        assert!(
            qualification_outcome
                .cold_completed_prefill_chunck_token_counts
                .contains(&configured_prefill_chunck_tokens),
            "the selected prefill size must complete as one model forward"
        );
        assert!(
            qualification_outcome.restored_cached_token_count
                >= configured_prefill_chunck_tokens,
            "the restored request recovered {} tokens but must recover every 2048-token block produced by one selected {configured_prefill_chunck_tokens}-token forward",
            qualification_outcome.restored_cached_token_count,
        );
        assert!(
            qualification_outcome.cold_generated_token_ids.len()
                >= LARGE_PREFILL_QUALIFICATION_OUTPUT_TOKEN_COUNT * 85 / 100,
            "the representative qualification produced only {} of approximately {} requested output tokens",
            qualification_outcome.cold_generated_token_ids.len(),
            LARGE_PREFILL_QUALIFICATION_OUTPUT_TOKEN_COUNT,
        );
        assert_eq!(
            qualification_outcome.cold_generated_token_ids,
            qualification_outcome.restored_generated_token_ids,
            "cold and restored deterministic generation must remain identical"
        );
    })
    .await;
}

async fn run_persistent_prompt_cache_greedy_parity_qualification(
    model_directory: &Path,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
    fixed_prefill_chunck_tokens: Option<u32>,
) -> PersistentPromptCacheParityQualificationOutcome {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");

    // First run: cold prefill, should populate the persistent prompt cache.
    let (mut qwen3_5_engine, _model_id, _model_revision, persistent_prompt_cache_model_contract) =
        load_persistent_prompt_cache_qualification_engine(
            model_directory,
            persistent_prompt_cache_directory.path(),
            fixed_prefill_chunck_tokens,
        )
        .await;
    let first_request_id = RequestId::new(2_001);
    let first_generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                first_request_id,
                prompt_token_ids.clone(),
                u16::try_from(generated_token_count)
                    .expect("the generated token count should fit the engine request"),
            )
            .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the first engine should accept the request");
    assert_eq!(first_generation_start.cached_token_count(), 0);
    let (cold_generated_token_ids, cold_completed_prefill_chunck_token_counts) =
        generate_token_ids(&mut qwen3_5_engine, first_request_id, generated_token_count).await;
    let minimum_expected_sequence_state_block_count = fixed_prefill_chunck_tokens
        .map(|fixed_prefill_chunck_tokens| {
            usize::try_from(fixed_prefill_chunck_tokens).unwrap_or(usize::MAX)
                / persistent_prompt_cache_model_contract.block_token_count()
        })
        .unwrap_or(1);
    wait_for_persistent_prompt_cache_blocks(
        &qwen3_5_engine,
        minimum_expected_sequence_state_block_count,
    )
    .await;
    let direct_prefix_lookup = PersistentPromptCachePrefixLookup::for_prompt(
        &persistent_prompt_cache_model_contract,
        &prompt_token_ids,
        |persistent_prompt_cache_block_hash| {
            persistent_prompt_cache_file_exists(
                persistent_prompt_cache_directory.path(),
                "kv_blocks",
                persistent_prompt_cache_block_hash,
            )
        },
        |persistent_prompt_cache_block_hash| {
            persistent_prompt_cache_file_exists(
                persistent_prompt_cache_directory.path(),
                "recurrent_snapshots",
                persistent_prompt_cache_block_hash,
            )
        },
    );
    eprintln!(
        "[persistent-prompt-cache-qualification] direct_lookup_restored_tokens={} diagnostics={:?}",
        direct_prefix_lookup.restored_token_count(),
        direct_prefix_lookup.diagnostics(),
    );

    // Second run: same prompt, same loaded engine, same prompt-cache directory. The
    // prompt is longer than one persistent prompt-cache block, so the second start
    // must report a hit without paying another full model load in this proof.
    let second_request_id = RequestId::new(2_002);
    let second_generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                second_request_id,
                prompt_token_ids,
                u16::try_from(generated_token_count)
                    .expect("the generated token count should fit the engine request"),
            )
            .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the second engine should accept the request");
    assert!(
        second_generation_start.cached_token_count()
            >= persistent_prompt_cache_model_contract.block_token_count() as u32,
        "the second run should report at least one restored prompt-cache block"
    );
    let restored_cached_token_count = second_generation_start.cached_token_count();
    let (restored_generated_token_ids, _restored_completed_prefill_chunck_token_counts) =
        generate_token_ids(
            &mut qwen3_5_engine,
            second_request_id,
            generated_token_count,
        )
        .await;
    PersistentPromptCacheParityQualificationOutcome {
        cold_generated_token_ids,
        cold_completed_prefill_chunck_token_counts,
        restored_cached_token_count,
        restored_generated_token_ids,
    }
}

pub(super) async fn load_persistent_prompt_cache_qualification_engine(
    model_directory: &Path,
    persistent_prompt_cache_directory: &Path,
    fixed_prefill_chunck_tokens: Option<u32>,
) -> (
    Qwen3_5Engine,
    String,
    String,
    PersistentPromptCacheModelContract,
) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before engine loading");
    let model_id = validated_artifact.model_id().to_owned();
    let model_revision = validated_artifact.revision().to_owned();
    let maximum_prefill_chunck_tokens = validated_artifact.config().maximum_position_count();
    let prefill_chunck_sizer = match fixed_prefill_chunck_tokens {
        Some(fixed_prefill_chunck_tokens) => {
            Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(fixed_prefill_chunck_tokens)
                .expect("the selected fixed prefill size should be valid")
        }
        None => Qwen3_5PrefillChunckSizer::production(
            maximum_prefill_chunck_tokens,
            vec![1_024, 2_048, 4_096, 8_192],
        )
        .expect("the configured candidates should configure the optimizer"),
    };
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::resolve(
        model_id.clone(),
        model_revision.clone(),
        qwen3_5_decoder_cache_layout(validated_artifact.config())
            .expect("the validated artifact should provide a decoder-cache layout"),
        validated_artifact.config().maximum_position_count() as usize,
        mlx_memory_limits.active_memory_limit_bytes() as u64,
        crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
    )
    .expect("the qualification model should resolve a persistent storage contract");
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.to_path_buf(),
            persistent_prompt_cache_directory.to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        )),
        prefill_chunck_sizer,
        248_069,
        model_directory.to_path_buf(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
    )
    .expect("the engine should accept the prompt-cache directory");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the model");
    (
        qwen3_5_engine,
        model_id,
        model_revision,
        persistent_prompt_cache_model_contract,
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
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let persistent_prompt_cache_model_contract = PersistentPromptCacheModelContract::resolve(
        validated_artifact.model_id().to_owned(),
        validated_artifact.revision().to_owned(),
        qwen3_5_decoder_cache_layout(validated_artifact.config())
            .expect("the validated artifact should provide a decoder-cache layout"),
        validated_artifact.config().maximum_position_count() as usize,
        mlx_memory_limits.active_memory_limit_bytes() as u64,
        crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
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

fn persistent_prompt_cache_file_exists(
    persistent_prompt_cache_directory: &Path,
    cache_file_kind_directory: &str,
    persistent_prompt_cache_block_hash: &[u8; 32],
) -> bool {
    let persistent_prompt_cache_block_hash_hex = persistent_prompt_cache_block_hash
        .iter()
        .map(|block_hash_byte| format!("{block_hash_byte:02x}"))
        .collect::<String>();
    persistent_prompt_cache_directory
        .join(cache_file_kind_directory)
        .join(format!(
            "{persistent_prompt_cache_block_hash_hex}.safetensors"
        ))
        .is_file()
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
            .expect("the qualification engine should have persistent prompt caching enabled");
        let WorkerEvent::PersistentPromptCacheStats {
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
                "[persistent-prompt-cache-qualification] status=cache-ready sequence_blocks={persistent_prompt_cache_sequence_state_block_count} boundary_snapshots={persistent_prompt_cache_boundary_state_snapshot_count}"
            );
            return;
        }
        assert!(
            publication_started_at.elapsed() < Duration::from_secs(10),
            "only {persistent_prompt_cache_sequence_state_block_count} of {expected_sequence_state_block_count} expected prompt-cache blocks were published"
        );
        eprintln!(
            "[persistent-prompt-cache-qualification] status=waiting-for-cache published_blocks={persistent_prompt_cache_sequence_state_block_count} expected_blocks={expected_sequence_state_block_count}"
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
    let mut completed_prefill_chunck_token_counts = Vec::new();
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
                    return (generated_token_ids, completed_prefill_chunck_token_counts);
                }
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunck_tokens,
                ..
            } => completed_prefill_chunck_token_counts.push(completed_prefill_chunck_tokens),
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                return (generated_token_ids, completed_prefill_chunck_token_counts);
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

async fn require_persistent_prompt_cache_qualification_completion(
    test_future: impl Future<Output = ()>,
) {
    // The pinned artifact is intentionally large, but a local qualification
    // must never leave the laptop compiling or evaluating indefinitely.
    let started_at = Instant::now();
    let timeout_deadline = sleep(PERSISTENT_PROMPT_CACHE_QUALIFICATION_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    eprintln!(
        "[persistent-prompt-cache-qualification] status=start timeout_seconds={}",
        PERSISTENT_PROMPT_CACHE_QUALIFICATION_TIMEOUT.as_secs()
    );

    loop {
        tokio::select! {
            () = &mut test_future => {
                eprintln!(
                    "[persistent-prompt-cache-qualification] status=success elapsed_seconds={:.1}",
                    started_at.elapsed().as_secs_f64()
                );
                return;
            }
            () = &mut timeout_deadline => {
                panic!(
                    "the persistent prompt-cache qualification exceeded {} seconds",
                    PERSISTENT_PROMPT_CACHE_QUALIFICATION_TIMEOUT.as_secs()
                );
            }
            _ = progress_interval.tick() => {
                let elapsed = started_at.elapsed();
                let remaining = PERSISTENT_PROMPT_CACHE_QUALIFICATION_TIMEOUT.saturating_sub(elapsed);
                eprintln!(
                    "[persistent-prompt-cache-qualification] status=running elapsed_seconds={:.0} ETA<={:.0}",
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64()
                );
            }
        }
    }
}
