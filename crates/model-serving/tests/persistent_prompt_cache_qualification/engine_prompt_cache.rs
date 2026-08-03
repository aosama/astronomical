use std::{future::Future, path::Path, time::Duration};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT, PersistentPromptCacheDiskStoreConfig,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer,
};
use tokio::time::{Instant, MissedTickBehavior, interval, sleep};

const PERSISTENT_PROMPT_CACHE_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const SAY_HI_PROMPT_TOKEN_IDS: [u32; 15] = [
    248_045, 846, 198, 44_240, 15_131, 13, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];
const QWEN3_6_35B_A3B_EIGHT_BIT_MODEL_ID: &str = "Qwen3.6-35B-A3B-8bit";
const MAXIMUM_GREEDY_OUTPUT_TOKEN_COUNT: usize = 10;

#[tokio::test]
#[ignore = "loads and generates with the complete pinned 22 GB Ornith artifact"]
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
        .expect("the pinned Ornith artifact should validate before engine loading");
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
        false,
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
            GeneratedToken::EndOfSequence => break,
        }
    }

    assert_eq!(
        generated_token_ids,
        vec![12_675, 0, 2_500, 628, 353, 1_438, 488, 3_242, 30, 248_046]
    );
}

#[tokio::test]
#[ignore = "loads and generates with the complete pinned 22 GB Ornith artifact"]
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
    let (first_generated_token_ids, second_generated_token_ids) =
        run_persistent_prompt_cache_greedy_parity_qualification(
            &model_directory,
            persistent_prompt_cache_eligible_prompt_token_ids(
                PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT + 16,
            ),
            1,
        )
        .await;

    assert_eq!(first_generated_token_ids.len(), 1);
    assert_eq!(first_generated_token_ids, second_generated_token_ids);
}

#[tokio::test]
#[ignore = "loads Qwen3.6-35B-A3B-8bit and compares cold and restored prompt-cache greedy tokens"]
async fn should_preserve_qwen3_6_greedy_tokens_after_persistent_prompt_cache_restore() {
    require_persistent_prompt_cache_qualification_completion(async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_model_artifact_directory_by_id(
            QWEN3_6_35B_A3B_EIGHT_BIT_MODEL_ID,
        );
        let (first_generated_token_ids, second_generated_token_ids) =
            run_persistent_prompt_cache_greedy_parity_qualification(
                &model_directory,
                persistent_prompt_cache_eligible_prompt_token_ids(
                    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT * 4 + 16,
                ),
                MAXIMUM_GREEDY_OUTPUT_TOKEN_COUNT,
            )
            .await;

        assert!(
            !first_generated_token_ids.is_empty(),
            "the cold Qwen3.6 request should emit at least one greedy token before terminal EOS"
        );
        assert_eq!(first_generated_token_ids, second_generated_token_ids);
    })
    .await;
}

async fn run_persistent_prompt_cache_greedy_parity_qualification(
    model_directory: &Path,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
) -> (Vec<u32>, Vec<u32>) {
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");

    // First run: cold prefill, should populate the persistent prompt cache.
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model-artifact checkpoint should validate before engine loading");
    let maximum_prefill_chunck_tokens = validated_artifact.config().maximum_position_count();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().to_path_buf(),
            persistent_prompt_cache_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        )),
        Qwen3_5PrefillChunckSizer::production(maximum_prefill_chunck_tokens)
            .expect("the validated model context maximum should configure the optimizer"),
        248_069,
        model_directory.to_path_buf(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        false,
    )
    .expect("the engine should accept the prompt-cache directory");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the model");
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
    let first_generated_token_ids =
        generate_token_ids(&mut qwen3_5_engine, first_request_id, generated_token_count).await;

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
            >= PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT as u32,
        "the second run should report at least one restored prompt-cache block"
    );
    let second_generated_token_ids = generate_token_ids(
        &mut qwen3_5_engine,
        second_request_id,
        generated_token_count,
    )
    .await;
    (first_generated_token_ids, second_generated_token_ids)
}

async fn generate_token_ids(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    generated_token_count: usize,
) -> Vec<u32> {
    let mut generated_token_ids = Vec::new();
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
                    return generated_token_ids;
                }
            }
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::EndOfSequence => {
                return generated_token_ids;
            }
        }
    }
}

fn persistent_prompt_cache_eligible_prompt_token_ids(target_token_count: usize) -> Vec<u32> {
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
