use std::time::Duration;

use ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use model_serving::{
    InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5Tokenizer, TextInferenceRequest,
};
use tokio::time::timeout;

use super::super::small_chunk_text_engine::{
    construct_small_chunk_text_engine, count_tokens_within_vocabulary,
};
use crate::serving_acceptance::support::{IMAGE_PAD_TOKEN_ID, LOCAL_AI_PROMPT_TOKEN_IDS};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../../verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

const ROMEO_CONTINUATION_MAX_TOKENS: usize = 20;

#[tokio::test]
#[ignore = "STALE CONTRACT: the engine is the SSD-paging sparse-MoE engine and its startup residency selection requires a sparse expert payload (qwen3_5_core/artifacts/memory_geometry_measurements.rs rejects dense artifacts by design); dense serving is covered end to end by the daemon e2e and public REST journeys"]
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
    let mut qwen3_5_engine = construct_small_chunk_text_engine(
        validated_artifact,
        &mlx_memory_limits,
        &model_directory,
        image_pad_token_id,
    );
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
                    repetition_penalty_thousandths: None,
                },
                structured_generation: None,
            },
            false,
        )
        .expect("the Romeo prompt should prepare with thinking off");
    qwen3_5_engine
        .start_generation(romeo_request)
        .await
        .expect("the engine should accept one Romeo request");

    let generated_token_count = count_tokens_within_vocabulary(
        &mut qwen3_5_engine,
        request_id,
        ROMEO_CONTINUATION_MAX_TOKENS,
        model_vocabulary_size,
        "generated",
    )
    .await;
    assert!(
        generated_token_count > 0,
        "the engine must produce at least one token"
    );
}

#[tokio::test]
#[ignore = "loads and samples with the complete Ornith artifact"]
async fn should_generate_a_sampled_continuation_through_the_engine_trait() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_large_sparse_moe_streaming_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the Ornith artifact should validate before engine loading");
    let model_vocabulary_size = validated_artifact.config().vocabulary_size();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let mut qwen3_5_engine = construct_small_chunk_text_engine(
        validated_artifact,
        &mlx_memory_limits,
        &model_directory,
        IMAGE_PAD_TOKEN_ID,
    );
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the complete Ornith model");
    let request_id = RequestId::new(1_001);
    qwen3_5_engine
        .start_generation(
            TextInferenceRequest::new_sampling(
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
    let generated_token_count = count_tokens_within_vocabulary(
        &mut qwen3_5_engine,
        request_id,
        16,
        model_vocabulary_size,
        "sampled",
    )
    .await;
    assert!(
        generated_token_count > 0,
        "the engine must produce at least one sampled token"
    );
}
