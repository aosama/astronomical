//! User journey: a two-turn conversation whose second turn extends the first
//! turn's prompt by fewer tokens than one cache block. The first turn's final
//! partial block ("tail") must be restored on the second turn so the engine
//! prefills only the newly added suffix instead of the tail tokens again.

use std::path::Path;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerEvent,
};
use astronomical_model_serving::{
    InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5InferenceRequest, Qwen3_5Tokenizer,
};

use super::engine_prompt_cache::{
    generate_token_ids, load_persistent_prompt_cache_acceptance_engine,
    require_persistent_prompt_cache_acceptance_completion, wait_for_persistent_prompt_cache_blocks,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
const TURN_ONE_TAIL_TOKEN_COUNT: usize = 16;
const TURN_TWO_EXTENSION_TOKEN_COUNT: usize = 16;
const TURN_ONE_REQUEST_ID: u64 = 3_001;
const TURN_TWO_REQUEST_ID: u64 = 3_002;

#[tokio::test]
#[ignore = "loads and generates with the complete Ornith artifact"]
async fn should_restore_the_previous_turns_partial_tail_block_on_an_extended_prompt() {
    require_persistent_prompt_cache_acceptance_completion(run_partial_tail_reuse_acceptance())
        .await;
}

async fn run_partial_tail_reuse_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    // The resident 4bit Ornith-35B artifact is the installed sparse-MoE variant on
    // development machines; the tail-reuse path is artifact-size independent.
    let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the test should create a prompt-cache directory");
    let (mut qwen3_5_engine, _model_id, _model_revision, block_token_count) =
        load_persistent_prompt_cache_acceptance_engine(
            &model_directory,
            persistent_prompt_cache_directory.path(),
            2_048,
        )
        .await;

    let conversation_token_ids = romeo_and_juliet_conversation_token_ids(
        &model_directory,
        2 * block_token_count + TURN_ONE_TAIL_TOKEN_COUNT + TURN_TWO_EXTENSION_TOKEN_COUNT,
    );
    let turn_one_prompt_token_ids =
        conversation_token_ids[..2 * block_token_count + TURN_ONE_TAIL_TOKEN_COUNT].to_vec();
    let turn_two_prompt_token_ids = conversation_token_ids
        [..2 * block_token_count + TURN_ONE_TAIL_TOKEN_COUNT + TURN_TWO_EXTENSION_TOKEN_COUNT]
        .to_vec();
    assert_eq!(
        turn_two_prompt_token_ids[..turn_one_prompt_token_ids.len()],
        turn_one_prompt_token_ids[..],
        "the second turn must extend the first turn's prompt without changing its prefix"
    );

    // Turn 1: two complete blocks plus a trailing partial block. The cold run
    // publishes both full blocks and captures the trailing tail.
    let turn_one_request_id = RequestId::new(TURN_ONE_REQUEST_ID);
    let turn_one_start = qwen3_5_engine
        .start_generation(extended_conversation_request(
            turn_one_request_id.clone(),
            turn_one_prompt_token_ids,
        ))
        .await
        .expect("the engine should accept the first turn");
    assert_eq!(
        turn_one_start.cached_token_count(),
        0,
        "the first turn on a fresh cache directory must prefill cold"
    );
    let (turn_one_generated_token_ids, _) =
        generate_token_ids(&mut qwen3_5_engine, turn_one_request_id, 1).await;
    assert_eq!(turn_one_generated_token_ids.len(), 1);
    // Two full blocks plus the captured tail each carry sequence state.
    wait_for_persistent_prompt_cache_blocks(&qwen3_5_engine, 3).await;

    // Turn 2: extend the same conversation by fewer tokens than one block. The
    // lookup must restore both complete blocks AND the first turn's tail, so
    // the second turn prefills only the newly added suffix.
    let turn_two_request_id = RequestId::new(TURN_TWO_REQUEST_ID);
    let turn_two_start = qwen3_5_engine
        .start_generation(extended_conversation_request(
            turn_two_request_id.clone(),
            turn_two_prompt_token_ids,
        ))
        .await
        .expect("the engine should accept the second turn");
    let expected_restored_token_count = (2 * block_token_count + TURN_ONE_TAIL_TOKEN_COUNT) as u32;
    assert_eq!(
        turn_two_start.cached_token_count(),
        expected_restored_token_count,
        "the second turn should restore both complete blocks and the first turn's partial tail"
    );
    let (turn_two_generated_token_ids, _) =
        generate_token_ids(&mut qwen3_5_engine, turn_two_request_id, 1).await;
    assert_eq!(turn_two_generated_token_ids.len(), 1);

    let cache_stats = qwen3_5_engine
        .collect_persistent_prompt_cache_stats()
        .await
        .expect("the engine should report persistent prompt-cache stats")
        .expect("the acceptance engine should have persistent prompt caching enabled");
    let WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_partial_tail_hits,
        ..
    } = cache_stats
    else {
        panic!("the engine returned an unexpected prompt-cache stats event")
    };
    assert!(
        persistent_prompt_cache_partial_tail_hits >= 1,
        "the second turn's tail restore should be counted as a partial tail hit"
    );
}

fn romeo_and_juliet_conversation_token_ids(
    model_directory: &Path,
    minimum_token_count: usize,
) -> Vec<u32> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the acceptance artifact should validate for prompt preparation");
    let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the acceptance tokenizer should load");
    for source_repetition_count in 1..=4 {
        let prompt_content = format!(
            "Romeo and Juliet source material:\n\n{}\n\nContinue the story in your own words.",
            ROMEO_AND_JULIET_SOURCE.repeat(source_repetition_count),
        );
        let prepared_request = prompt_tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(3_000),
                    model: validated_artifact.model_id().to_owned(),
                    messages: vec![ChatMessage::User {
                        content: prompt_content,
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: 16,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                    structured_generation: None,
                },
                false,
            )
            .expect("the Romeo and Juliet acceptance prompt should prepare");
        if prepared_request.input_token_ids().len() >= minimum_token_count {
            return prepared_request.input_token_ids()[..minimum_token_count].to_vec();
        }
    }
    panic!("the Romeo and Juliet acceptance prompt did not reach the requested token count");
}

fn extended_conversation_request(
    request_id: RequestId,
    prompt_token_ids: Vec<u32>,
) -> Qwen3_5InferenceRequest {
    Qwen3_5InferenceRequest::new_sampling(request_id, prompt_token_ids, 1, 1_000, 1_000, None)
        .with_image_pad_token_id(248_069)
        .with_thinking_configuration(false, None, Vec::new(), Vec::new())
}
