use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use tokio::time::timeout;

use crate::serving_acceptance::support::IMAGE_PAD_TOKEN_ID;

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

const LOCAL_AI_PROMPT_TOKEN_IDS: [u32; 20] = [
    248_045, 846, 198, 657, 799, 14_542, 8_495, 314, 2_136, 14_791, 13, 248_046, 198, 248_045,
    74_455, 198, 248_068, 271, 248_069, 271,
];

const ROMEO_CONTINUATION_MAX_TOKENS: usize = 20;

#[tokio::test]
#[ignore = "loads a configured dense model and continues a Romeo and Juliet prompt"]
async fn should_generate_continuation_through_the_engine_trait() {
    timeout(Duration::from_secs(120), run_romeo_continuation())
        .await
        .expect("Romeo continuation must finish within 120 seconds");
}

async fn run_romeo_continuation() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_id = crate::common::small_dense_model_id();
    let model_directory = crate::common::configured_installed_model_directory_by_id(model_id);
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the configured dense artifact should validate before engine loading");
    let model_vocabulary_size = validated_artifact.config().vocabulary_size();
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the configured tokenizer should load");
    let image_pad_token_id = tokenizer.image_pad_token_id();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(16)
            .expect("the test prefill chunk size should be valid"),
        image_pad_token_id,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the dense engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the configured dense model");
    let request_id = RequestId::new(1_000);
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(400)
        .collect::<String>();
    let romeo_request = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id,
                model: model_id.to_owned(),
                messages: vec![ChatMessage::User {
                    content: format!(
                        "Name one household from this Romeo and Juliet excerpt: {source_excerpt}"
                    ),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: ROMEO_CONTINUATION_MAX_TOKENS as u16,
                    temperature_thousandths: Some(1_000),
                    top_p_thousandths: None,
                    seed: None,
                    thinking_budget: None,
                },
                qwen_thinking_channel_seed: None,
            },
            false,
        )
        .expect("the Romeo prompt should prepare with thinking off");
    qwen3_5_engine
        .start_generation(romeo_request)
        .await
        .expect("the engine should accept one Romeo request");

    let mut generated_token_count = 0usize;
    while generated_token_count < ROMEO_CONTINUATION_MAX_TOKENS {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each engine boundary should advance the request")
        {
            GeneratedToken::TokenId { token_id, .. } => {
                assert!(
                    token_id < model_vocabulary_size,
                    "generated token id {token_id} must be within vocabulary size {model_vocabulary_size}"
                );
                generated_token_count += 1;
            }
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }
    assert!(
        generated_token_count > 0,
        "the engine must produce at least one token"
    );
}

#[tokio::test]
#[ignore = "loads and samples with the complete Ornith artifact"]
async fn should_generate_a_sampled_continuation_through_the_engine_trait() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate before engine loading");
    let model_vocabulary_size = validated_artifact.config().vocabulary_size();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(16)
            .expect("the test prefill_chunck_tokens should be valid"),
        IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the bounded Ornith engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the complete Ornith model");
    let request_id = RequestId::new(1_001);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                LOCAL_AI_PROMPT_TOKEN_IDS.to_vec(),
                16,
                1_000,
                1_000,
                Some(1_234),
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID),
        )
        .await
        .expect("the engine should accept one deterministically sampled request");

    // Structural validity: the engine must produce at least one token and
    // finish cleanly, without asserting exact token values that couple the
    // test to one specific quantization artifact.
    let mut generated_token_count = 0usize;
    while generated_token_count < 16 {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each engine boundary should advance the sampled request")
        {
            GeneratedToken::TokenId { token_id, .. } => {
                assert!(
                    token_id < model_vocabulary_size,
                    "sampled token id {token_id} must be within vocabulary size {model_vocabulary_size}"
                );
                generated_token_count += 1;
            }
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }
    assert!(
        generated_token_count > 0,
        "the engine must produce at least one sampled token"
    );
}
