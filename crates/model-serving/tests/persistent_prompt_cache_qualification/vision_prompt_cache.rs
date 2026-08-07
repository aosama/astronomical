use std::io::Cursor;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId, WorkerEvent,
};
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use tokio::time::{Instant, sleep, timeout};

const VISUAL_QUALIFICATION_PREFILL_CHUNCK_TOKENS: u32 = 8_192;
const VISUAL_QUALIFICATION_MINIMUM_PROMPT_TOKENS: usize = 8_256;
const VISUAL_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "loads Ornith and qualifies checkpoint-aware visual prefill with prompt-cache restore"]
async fn should_preserve_visual_prefill_output_across_an_8192_token_cache_restore() {
    timeout(
        VISUAL_QUALIFICATION_TIMEOUT,
        run_visual_prompt_cache_qualification(false, false),
    )
    .await
    .expect("the visual prompt-cache qualification must finish within 115 seconds");
}

#[tokio::test]
#[ignore = "forces one checkpoint-capable visual prefill rejection and verifies retry rollback"]
async fn should_restore_visual_and_decoder_state_after_a_checkpoint_prefill_retry() {
    timeout(
        VISUAL_QUALIFICATION_TIMEOUT,
        run_visual_prompt_cache_qualification(true, false),
    )
    .await
    .expect("the visual prefill rollback qualification must finish within 115 seconds");
}

#[tokio::test]
#[ignore = "removes the optional cache writer and verifies the visual forward remains 8192 tokens"]
async fn should_keep_large_visual_prefill_when_the_cache_writer_is_unavailable() {
    timeout(
        VISUAL_QUALIFICATION_TIMEOUT,
        run_visual_prompt_cache_qualification(false, true),
    )
    .await
    .expect("the read-only visual cache qualification must finish within 115 seconds");
}

async fn run_visual_prompt_cache_qualification(
    force_prefill_retry: bool,
    disable_prompt_cache_writes: bool,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate for visual cache qualification");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Ornith tokenizer should load");
    let prompt_cache_directory = tempfile::tempdir()
        .expect("the visual qualification should create a prompt-cache directory");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        Some(PersistentPromptCacheDiskStoreConfig::new(
            prompt_cache_directory.path().to_path_buf(),
            prompt_cache_directory.path().to_path_buf(),
            10_000_000_000,
        )),
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(
            VISUAL_QUALIFICATION_PREFILL_CHUNCK_TOKENS,
        )
        .expect("the visual qualification prefill size should be valid"),
        tokenizer.think_end_token_id(),
        model_directory,
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
    )
    .expect("the visual qualification engine should construct");
    eprintln!("[visual-prompt-cache-qualification] status=loading-model");
    qwen3_5_engine
        .load()
        .await
        .expect("the visual qualification engine should load");
    if disable_prompt_cache_writes {
        qwen3_5_engine
            .disable_persistent_prompt_cache_writes_for_tests()
            .await
            .expect("the visual qualification should remove the optional cache writer");
    }

    let cold_request = representative_visual_request(&tokenizer, RequestId::new(9_100));
    assert!(cold_request.input_token_ids().len() >= VISUAL_QUALIFICATION_MINIMUM_PROMPT_TOKENS);
    qwen3_5_engine
        .start_generation(cold_request)
        .await
        .expect("the cold visual request should start");
    if force_prefill_retry {
        qwen3_5_engine
            .force_next_prefill_capacity_rejection_for_tests(RequestId::new(9_100))
            .await
            .expect("the visual qualification should arm one prefill rejection");
    }
    let (cold_token_id, cold_prefill_chunck_token_counts) =
        generate_one_token(&mut qwen3_5_engine, RequestId::new(9_100)).await;
    assert!(
        cold_prefill_chunck_token_counts.contains(&VISUAL_QUALIFICATION_PREFILL_CHUNCK_TOKENS),
        "the visual request must complete one 8192-token forward"
    );
    if disable_prompt_cache_writes {
        assert_eq!(
            prompt_cache_sequence_state_block_count(&qwen3_5_engine).await,
            0
        );
    } else {
        wait_for_four_prompt_cache_blocks(&qwen3_5_engine).await;
    }

    let restored_request = representative_visual_request(&tokenizer, RequestId::new(9_101));
    let restored_generation_start = qwen3_5_engine
        .start_generation(restored_request)
        .await
        .expect("the restored visual request should start");
    if disable_prompt_cache_writes {
        assert_eq!(restored_generation_start.cached_token_count(), 0);
    } else {
        assert!(
            restored_generation_start.cached_token_count()
                >= VISUAL_QUALIFICATION_PREFILL_CHUNCK_TOKENS
        );
    }
    let (restored_token_id, _restored_prefill_chunck_token_counts) =
        generate_one_token(&mut qwen3_5_engine, RequestId::new(9_101)).await;
    assert_eq!(cold_token_id, restored_token_id);
    eprintln!("[visual-prompt-cache-qualification] status=success");
}

fn representative_visual_request(
    tokenizer: &Qwen3_5Tokenizer,
    request_id: RequestId,
) -> astronomical_model_serving::Qwen3_5InferenceRequest {
    let context_sentence =
        "The image observation is compared with a written astronomical measurement. ";
    for context_sentence_repetition_count in (512..=2_048).step_by(128) {
        let request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id,
                    model: "Ornith-1.0-35B-OptiQ-4bit".to_owned(),
                    messages: vec![ChatMessage::User {
                        content: format!(
                            "Describe the image, then use these notes:\n\n{}",
                            context_sentence.repeat(context_sentence_repetition_count),
                        ),
                        images: vec![ChatImageInput {
                            mime_type: "image/png".to_owned(),
                            decoded_bytes: one_pixel_png(),
                        }],
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: 1,
                        temperature_thousandths: Some(0),
                        top_p_thousandths: Some(950),
                        seed: Some(1),
                        thinking_budget: None,
                    },
                },
                false,
            )
            .expect("the representative visual request should prepare");
        if request.input_token_ids().len() >= VISUAL_QUALIFICATION_MINIMUM_PROMPT_TOKENS {
            return request;
        }
    }
    panic!("the representative visual request did not reach 8,256 prompt tokens")
}

async fn generate_one_token(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
) -> (u32, Vec<u32>) {
    let mut completed_prefill_chunck_token_counts = Vec::new();
    loop {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the visual request should advance")
        {
            GeneratedToken::TokenId { token_id, .. } => {
                return (token_id, completed_prefill_chunck_token_counts);
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunck_tokens,
                ..
            } => {
                eprintln!(
                    "[visual-prompt-cache-qualification] completed_prefill_chunck_tokens={completed_prefill_chunck_tokens}"
                );
                completed_prefill_chunck_token_counts.push(completed_prefill_chunck_tokens);
            }
            GeneratedToken::EndOfSequence => {
                panic!("the visual request ended before producing one token");
            }
        }
    }
}

async fn wait_for_four_prompt_cache_blocks(qwen3_5_engine: &Qwen3_5Engine) {
    let wait_started_at = Instant::now();
    loop {
        let persistent_prompt_cache_sequence_state_block_count =
            prompt_cache_sequence_state_block_count(qwen3_5_engine).await;
        if persistent_prompt_cache_sequence_state_block_count >= 4 {
            return;
        }
        assert!(
            wait_started_at.elapsed() < Duration::from_secs(10),
            "the visual qualification did not publish four prompt-cache blocks"
        );
        eprintln!(
            "[visual-prompt-cache-qualification] status=waiting-for-cache published_blocks={persistent_prompt_cache_sequence_state_block_count}"
        );
        sleep(Duration::from_millis(250)).await;
    }
}

async fn prompt_cache_sequence_state_block_count(qwen3_5_engine: &Qwen3_5Engine) -> u64 {
    let cache_stats = qwen3_5_engine
        .collect_persistent_prompt_cache_stats()
        .await
        .expect("the visual qualification should collect cache stats")
        .expect("the visual qualification should have a prompt cache");
    let WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_sequence_state_block_count,
        ..
    } = cache_stats
    else {
        panic!("the visual qualification received unexpected cache stats")
    };
    persistent_prompt_cache_sequence_state_block_count
}

fn one_pixel_png() -> Vec<u8> {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([128, 64, 32]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the one-pixel image should encode");
    encoded_image_bytes.into_inner()
}
