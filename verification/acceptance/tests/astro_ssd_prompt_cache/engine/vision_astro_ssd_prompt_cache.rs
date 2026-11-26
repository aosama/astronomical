use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId, WorkerEvent,
};
use model_serving::{
    AstroSsdPromptCacheDiskStoreConfig, GeneratedToken, InferenceEngine,
    PromptProcessingChunkSizer, Qwen3_5ArtifactValidator, Qwen3_5StreamingEngine, Qwen3_5Tokenizer,
    TextMlxEngineConstructors,
};
use tokio::time::{Instant, sleep, timeout};

const VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS: u32 = 2_048;
const VISUAL_ACCEPTANCE_MINIMUM_PROMPT_TOKENS: usize = 8_256;
const VISUAL_ACCEPTANCE_SAMPLING_SEED: u64 = 9_100;
const VISUAL_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "loads Ornith and proves checkpoint-aware visual prefill with astro SSD prompt cache restore"]
async fn should_preserve_visual_prefill_output_across_a_full_prefill_chunk_astro_ssd_prompt_cache_restore()
 {
    timeout(
        VISUAL_ACCEPTANCE_TIMEOUT,
        run_visual_astro_ssd_prompt_cache_acceptance(),
    )
    .await
    .expect("the visual astro-ssd-prompt-cache acceptance must finish within 115 seconds");
}

#[tokio::test]
#[ignore = "loads Ornith and proves causal prefix reuse when a later image is appended"]
async fn should_restore_the_causal_prefix_before_an_appended_image() {
    timeout(
        VISUAL_ACCEPTANCE_TIMEOUT,
        run_appended_visual_astro_ssd_prompt_cache_acceptance(),
    )
    .await
    .expect("the appended-image astro-ssd-prompt-cache acceptance must finish within 115 seconds");
}

#[tokio::test]
#[ignore = "loads Ornith and proves causal prefix reuse for a visual prompt"]
async fn should_restore_the_causal_prefix_of_a_visual_prompt() {
    timeout(
        VISUAL_ACCEPTANCE_TIMEOUT,
        run_visual_astro_ssd_prompt_cache_acceptance(),
    )
    .await
    .expect("the visual astro-ssd-prompt-cache acceptance must finish within 115 seconds");
}

async fn run_visual_astro_ssd_prompt_cache_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let astro_ssd_prompt_cache_directory = tempfile::tempdir()
        .expect("the visual acceptance should create an astro SSD prompt cache directory");
    let (mut qwen3_5_engine, tokenizer) =
        load_visual_acceptance_engine(astro_ssd_prompt_cache_directory.path()).await;
    let cold_request = representative_visual_request(&tokenizer, RequestId::new(9_100));
    assert!(cold_request.input_token_ids().len() >= VISUAL_ACCEPTANCE_MINIMUM_PROMPT_TOKENS);
    qwen3_5_engine
        .start_generation(cold_request)
        .await
        .expect("the cold visual request should start");
    let (cold_token_id, cold_prefill_chunk_token_counts) =
        generate_one_token(&mut qwen3_5_engine, RequestId::new(9_100)).await;
    assert!(
        cold_prefill_chunk_token_counts.iter().sum::<u32>()
            >= VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS,
        "the cold visual request must read at least one full prompt chunk"
    );
    wait_for_visual_astro_ssd_prompt_cache_block(&qwen3_5_engine).await;

    let restored_request = representative_visual_request(&tokenizer, RequestId::new(9_101));
    let restored_generation_start = qwen3_5_engine
        .start_generation(restored_request)
        .await
        .expect("the restored visual request should start");
    assert!(
        restored_generation_start.astro_ssd_prompt_cache_restored_token_count()
            >= VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS
    );
    let (restored_token_id, _restored_prefill_chunk_token_counts) =
        generate_one_token(&mut qwen3_5_engine, RequestId::new(9_101)).await;
    eprintln!(
        "[visual-astro-ssd-prompt-cache-acceptance] status=parity cold_token_id={cold_token_id} restored_token_id={restored_token_id}"
    );
    assert_eq!(cold_token_id, restored_token_id);
    eprintln!("[visual-astro-ssd-prompt-cache-acceptance] status=success");
}

async fn run_appended_visual_astro_ssd_prompt_cache_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let astro_ssd_prompt_cache_directory = tempfile::tempdir()
        .expect("the appended-image acceptance should create an astro SSD prompt cache directory");
    let (mut qwen3_5_engine, tokenizer) =
        load_visual_acceptance_engine(astro_ssd_prompt_cache_directory.path()).await;

    let cold_appended_request =
        representative_appended_visual_request(&tokenizer, RequestId::new(9_200));
    qwen3_5_engine
        .start_generation(cold_appended_request)
        .await
        .expect("the cold appended-image request should start");
    let (cold_appended_token_id, _) =
        generate_one_token(&mut qwen3_5_engine, RequestId::new(9_200)).await;
    qwen3_5_engine
        .clear_astro_ssd_prompt_cache(None)
        .await
        .expect("the acceptance should clear its private astro SSD prompt cache")
        .expect("the loaded engine should own an astro SSD prompt cache");

    let causal_prefix_request = representative_visual_request(&tokenizer, RequestId::new(9_201));
    let causal_prefix_token_count = causal_prefix_request.input_token_ids().len();
    qwen3_5_engine
        .start_generation(causal_prefix_request)
        .await
        .expect("the causal-prefix request should start");
    generate_one_token(&mut qwen3_5_engine, RequestId::new(9_201)).await;
    wait_for_visual_astro_ssd_prompt_cache_block(&qwen3_5_engine).await;

    let restored_appended_request =
        representative_appended_visual_request(&tokenizer, RequestId::new(9_202));
    let restored_generation_start = qwen3_5_engine
        .start_generation(restored_appended_request)
        .await
        .expect("the restored appended-image request should start");
    // The appended request may reuse every complete published block of the
    // causal prefix and nothing past the second image boundary. Comparing
    // against the causal prefix length rather than one block keeps that intent
    // true as the block size changes: a prompt that is an exact multiple of the
    // block size keeps its last block for a forward pass
    // (see astro_ssd_prompt_cache::prefix_lookup).
    let reusable_causal_prefix_token_count = causal_prefix_token_count
        / VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS as usize
        * VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS as usize;
    assert_eq!(
        restored_generation_start.astro_ssd_prompt_cache_restored_token_count() as usize,
        reusable_causal_prefix_token_count,
        "the restored appended-image request must reuse exactly the complete published blocks of the causal prefix"
    );
    let (restored_appended_token_id, _) =
        generate_one_token(&mut qwen3_5_engine, RequestId::new(9_202)).await;
    assert_eq!(cold_appended_token_id, restored_appended_token_id);
    eprintln!(
        "[appended-image-astro-ssd-prompt-cache-acceptance] status=success restored_tokens={} token_id={restored_appended_token_id}",
        restored_generation_start.astro_ssd_prompt_cache_restored_token_count()
    );
}

async fn load_visual_acceptance_engine(
    astro_ssd_prompt_cache_directory: &Path,
) -> (Qwen3_5StreamingEngine, Qwen3_5Tokenizer) {
    let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate for visual astro SSD prompt cache acceptance");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Ornith tokenizer should load");
    let mlx_memory_limits =
        crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let mut worker_chunking_configuration = crate::common::standard_worker_chunking_configuration();
    worker_chunking_configuration.ssd_prompt_cache_block_tokens =
        Some(VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS);
    let mut qwen3_5_engine = Qwen3_5StreamingEngine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.mlx_buffer_cache_limit_bytes(),
        Some(AstroSsdPromptCacheDiskStoreConfig::new(
            astro_ssd_prompt_cache_directory.to_path_buf(),
            astro_ssd_prompt_cache_directory.to_path_buf(),
            // The journey shares the configured global `astro_ssd_prompt_cache` quota with every
            // other `astro_ssd_prompt_cache` journey. A literal quota smaller than the
            // full-context chain is not a tighter journey: at 2,048-token
            // blocks this model needs a 13,611,687,296-byte chain, so any
            // smaller literal fails contract resolution at engine load before
            // the journey writes a single block.
            crate::common::configured_model_artifact_astro_ssd_prompt_cache_maximum_size_bytes(),
        )),
        PromptProcessingChunkSizer::for_chunk_size_tokens(VISUAL_ACCEPTANCE_PREFILL_CHUNK_TOKENS)
            .expect("the visual acceptance prefill size should be valid"),
        tokenizer.think_end_token_id(),
        model_directory,
        worker_chunking_configuration,
    )
    .expect("the visual acceptance engine should construct");
    eprintln!("[visual-astro-ssd-prompt-cache-acceptance] status=loading-model");
    qwen3_5_engine
        .load()
        .await
        .expect("the visual acceptance engine should load");
    super::keep_the_hosted_drafter_silent(&qwen3_5_engine).await;
    (qwen3_5_engine, tokenizer)
}

fn representative_visual_request(
    tokenizer: &Qwen3_5Tokenizer,
    request_id: RequestId,
) -> model_serving::TextInferenceRequest {
    representative_visual_request_with_optional_later_image(tokenizer, request_id, false)
}

fn representative_appended_visual_request(
    tokenizer: &Qwen3_5Tokenizer,
    request_id: RequestId,
) -> model_serving::TextInferenceRequest {
    representative_visual_request_with_optional_later_image(tokenizer, request_id, true)
}

fn representative_visual_request_with_optional_later_image(
    tokenizer: &Qwen3_5Tokenizer,
    request_id: RequestId,
    should_append_later_image: bool,
) -> model_serving::TextInferenceRequest {
    let context_sentence =
        "Two households, both alike in dignity, in fair Verona, where we lay our scene. ";
    for context_sentence_repetition_count in (512..=2_048).step_by(128) {
        let mut messages = vec![ChatMessage::User {
            content: format!(
                "Describe the image, then use these notes:\n\n{}",
                context_sentence.repeat(context_sentence_repetition_count),
            ),
            images: vec![ChatImageInput {
                mime_type: "image/png".to_owned(),
                decoded_bytes: one_pixel_png(),
            }],
        }];
        if should_append_later_image {
            messages.push(ChatMessage::Assistant {
                content: Some("The first scene has been observed.".to_owned()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            });
            messages.push(ChatMessage::User {
                content: "Compare it with this later scene.".to_owned(),
                images: vec![ChatImageInput {
                    mime_type: "image/png".to_owned(),
                    decoded_bytes: second_one_pixel_png(),
                }],
            });
        }
        let request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id,
                    model: crate::common::resident_sparse_moe_model_id().to_owned(),
                    messages,
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: 1,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: Some(VISUAL_ACCEPTANCE_SAMPLING_SEED),
                        thinking_budget: None,
                        repetition_penalty_thousandths: None,
                    },
                    structured_generation: None,
                },
                false,
            )
            .expect("the representative visual request should prepare");
        if request.input_token_ids().len() >= VISUAL_ACCEPTANCE_MINIMUM_PROMPT_TOKENS {
            return request;
        }
    }
    panic!("the representative visual request did not reach 8,256 prompt tokens")
}

async fn generate_one_token(
    qwen3_5_engine: &mut Qwen3_5StreamingEngine,
    request_id: RequestId,
) -> (u32, Vec<u32>) {
    let mut completed_prefill_chunk_token_counts = Vec::new();
    loop {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the visual request should advance")
        {
            GeneratedToken::TokenId { token_id, .. } => {
                return (token_id, completed_prefill_chunk_token_counts);
            }
            GeneratedToken::PrefillProgress {
                completed_prefill_chunk_tokens,
                mlx_memory_telemetry,
                ..
            } => {
                eprintln!(
                    "[visual-astro-ssd-prompt-cache-acceptance] completed_prefill_chunk_tokens={completed_prefill_chunk_tokens} active_memory_bytes={:?} peak_memory_bytes={:?}",
                    mlx_memory_telemetry
                        .as_ref()
                        .map(|telemetry| telemetry.active_memory_bytes),
                    mlx_memory_telemetry
                        .as_ref()
                        .map(|telemetry| telemetry.peak_memory_bytes),
                );
                completed_prefill_chunk_token_counts.push(completed_prefill_chunk_tokens);
            }
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the visual request ended before producing one token");
            }
        }
    }
}

async fn wait_for_visual_astro_ssd_prompt_cache_block(qwen3_5_engine: &Qwen3_5StreamingEngine) {
    let wait_started_at = Instant::now();
    loop {
        let astro_ssd_prompt_cache_sequence_state_block_count =
            astro_ssd_prompt_cache_sequence_state_block_count(qwen3_5_engine).await;
        if astro_ssd_prompt_cache_sequence_state_block_count >= 1 {
            return;
        }
        assert!(
            wait_started_at.elapsed() < Duration::from_secs(10),
            "the visual acceptance did not publish its first astro-ssd-prompt-cache block"
        );
        eprintln!(
            "[visual-astro-ssd-prompt-cache-acceptance] status=waiting-for-astro-ssd-prompt-cache published_blocks={astro_ssd_prompt_cache_sequence_state_block_count}"
        );
        sleep(Duration::from_millis(250)).await;
    }
}

async fn astro_ssd_prompt_cache_sequence_state_block_count(
    qwen3_5_engine: &Qwen3_5StreamingEngine,
) -> u64 {
    let astro_ssd_prompt_cache_stats = qwen3_5_engine
        .collect_astro_ssd_prompt_cache_stats()
        .await
        .expect("the visual acceptance should collect astro SSD prompt cache stats")
        .expect("the visual acceptance should have an astro SSD prompt cache");
    let WorkerEvent::AstroSsdPromptCacheStats {
        astro_ssd_prompt_cache_sequence_state_block_count,
        ..
    } = astro_ssd_prompt_cache_stats
    else {
        panic!("the visual acceptance received unexpected astro SSD prompt cache stats")
    };
    astro_ssd_prompt_cache_sequence_state_block_count
}

fn one_pixel_png() -> Vec<u8> {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([128, 64, 32]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the one-pixel image should encode");
    encoded_image_bytes.into_inner()
}

fn second_one_pixel_png() -> Vec<u8> {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([32, 96, 160]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the second one-pixel image should encode");
    encoded_image_bytes.into_inner()
}
