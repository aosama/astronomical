use std::fs;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngineError, LagunaServingSettings, MlxInferenceExecution,
    initialize_laguna_execution, initialize_laguna_execution_with_serving_settings,
};

use super::page_artifact::write_sparse_artifact;
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_start_a_laguna_engine_from_a_validated_artifact_and_generate_tokens() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("Laguna startup directory");
    write_sparse_artifact(model_directory.path(), false);
    let (generation_processor, mut engine) = initialize_laguna_execution(
        model_directory.path(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        true,
    )
    .expect("a synthetic Laguna artifact should start through the family startup module");
    let load_result = engine
        .load()
        .expect("the started Laguna engine should load");
    assert!(load_result.expert_memory_mode().is_some());
    let minimum_mlx_memory_ceiling_bytes = load_result.minimum_mlx_memory_ceiling_bytes();
    assert!(minimum_mlx_memory_ceiling_bytes > 1);
    assert!(matches!(
        engine.update_mlx_memory_limit(minimum_mlx_memory_ceiling_bytes - 1),
        Err(InferenceEngineError::MlxMemoryLimitRejected {
            minimum_mlx_memory_ceiling_bytes: rejected_minimum,
            ..
        }) if rejected_minimum == minimum_mlx_memory_ceiling_bytes
    ));

    let started_model_id = model_directory
        .path()
        .file_name()
        .expect("the startup directory should have a name")
        .to_string_lossy()
        .into_owned();
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(106),
        model: started_model_id,
        messages: vec![ChatMessage::User {
            content: "Use the supplied play as the only source for literary analysis. Wherefore art thou Romeo?".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 4_u16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(106),
            thinking_budget: Some(0),
        },
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the Romeo and Juliet chat should prepare");
    assert!(!prepared_generation.prompt_token_ids().is_empty());
    engine
        .start_generation(prepared_generation.into_inference_request())
        .expect("Laguna prompt processing should start");
    let mut observed_generated_boundary = false;
    for _advance_attempt in 0..16 {
        match engine
            .decode_next_token(chat_command.request_id)
            .expect("Laguna should advance prompt processing or generation")
        {
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::TokenId { token_id, .. } => {
                assert!(token_id < 8, "the synthetic vocabulary has eight tokens");
                observed_generated_boundary = true;
                break;
            }
            GeneratedToken::EndOfSequence => {
                observed_generated_boundary = true;
                break;
            }
            other => panic!("Laguna emitted an unexpected generation boundary: {other:?}"),
        }
    }
    assert!(
        observed_generated_boundary,
        "bounded prompt progress must eventually reach token generation"
    );
    engine
        .cancel_generation(chat_command.request_id)
        .expect("cancelling the Laguna request should leave the engine reusable");
}

#[tokio::test]
async fn should_fail_model_loading_when_required_prompt_cache_cannot_initialize() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("Laguna startup directory");
    write_sparse_artifact(model_directory.path(), false);
    let cache_parent = tempfile::tempdir().expect("prompt-cache parent directory");
    let invalid_cache_root = cache_parent.path().join("regular-file-cache-root");
    fs::write(&invalid_cache_root, b"not a directory")
        .expect("the invalid cache root fixture should write");
    let mut serving_settings = LagunaServingSettings::default_fixed();
    serving_settings.persistent_prompt_cache_enabled = true;
    serving_settings.prompt_cache_config =
        Some(PromptCacheConfig::new(invalid_cache_root, 50_000_000_000));
    let (_, mut execution) = initialize_laguna_execution_with_serving_settings(
        model_directory.path(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        false,
        serving_settings,
    )
    .expect("the descriptor-only startup phase should complete");

    assert!(matches!(
        execution.load(),
        Err(InferenceEngineError::Fatal { reason })
            if reason == "required Laguna prompt cache initialization failed"
    ));
}
