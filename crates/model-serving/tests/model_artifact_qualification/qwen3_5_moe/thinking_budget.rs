//! Model-backed qualification for the hard reasoning-budget decoder boundary.

use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use tokio::time::timeout;

use super::ORNITH_IMAGE_PAD_TOKEN_ID;

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 128;

#[tokio::test]
#[ignore = "loads Ornith and commits the complete bounded-reasoning transition"]
async fn should_condition_visible_answer_generation_on_the_complete_forced_transition() {
    timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_ornith_model_artifact_directory();
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the Ornith artifact should validate before engine loading");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
            .expect("the Ornith tokenizer should load");
        let forced_transition_token_ids = tokenizer.forced_thinking_transition_token_ids().to_vec();
        let request_id = RequestId::new(1_004);
        let inference_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id,
                    model: validated_artifact.model_id().to_owned(),
                    messages: vec![ChatMessage::User {
                        content: format!(
                            "Summarize this Romeo and Juliet excerpt in one sentence.\n\n{}",
                            ROMEO_AND_JULIET_SOURCE
                                .chars()
                                .take(512)
                                .collect::<String>()
                        ),
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: MAXIMUM_OUTPUT_TOKEN_COUNT,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(1),
                    },
                },
                true,
            )
            .expect("the bounded Romeo and Juliet request should prepare");
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let mut engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
            validated_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(128)
                .expect("the bounded prompt chunk should be valid"),
            ORNITH_IMAGE_PAD_TOKEN_ID,
            model_directory.to_path_buf(),
            crate::common::standard_worker_chunking_configuration(),
            false,
            crate::common::disabled_worker_speculative_prefill_configuration(),
        )
        .expect("the bounded Ornith engine settings should be valid");
        eprintln!("[thinking-budget] status=progress phase=model_load");
        engine
            .load()
            .await
            .expect("the engine should load the qualified model");
        engine
            .start_generation(inference_request)
            .await
            .expect("the engine should accept the bounded request");

        let mut emitted_token_ids = Vec::new();
        let mut emitted_reasoning_flags = Vec::new();
        while emitted_token_ids.len() < forced_transition_token_ids.len().saturating_add(2) {
            eprintln!(
                "[thinking-budget] status=progress emitted_tokens={}",
                emitted_token_ids.len()
            );
            match engine
                .decode_next_token(request_id)
                .await
                .expect("the bounded request should advance")
            {
                GeneratedToken::TokenId {
                    token_id,
                    is_reasoning_token,
                    ..
                } => {
                    emitted_token_ids.push(token_id);
                    emitted_reasoning_flags.push(is_reasoning_token);
                }
                GeneratedToken::EndOfSequence => {
                    panic!("the model ended before producing the bounded transition and answer")
                }
                GeneratedToken::PrefillProgress { .. }
                | GeneratedToken::PromptProcessingPhaseStarted { .. }
                | GeneratedToken::GenerationPreparationStarted { .. } => {}
            }
        }

        assert_eq!(
            &emitted_token_ids[1..forced_transition_token_ids.len().saturating_add(1)],
            forced_transition_token_ids.as_slice()
        );
        assert!(emitted_reasoning_flags[0]);
        assert_eq!(emitted_reasoning_flags.last(), Some(&false));
    })
    .await
    .expect("the bounded-reasoning qualification must finish within 115 seconds");
}
