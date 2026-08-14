use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer,
};

use super::engine_prompt_cache::{
    persistent_prompt_cache_eligible_prompt_token_ids,
    require_persistent_prompt_cache_qualification_completion,
};

#[tokio::test]
#[ignore = "loads and generates with the complete Ornith artifact"]
async fn should_generate_without_prompt_cache_storage_contract_work_when_cache_is_disabled() {
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
        crate::common::sample_machine_model_artifact_qualification_mlx_memory_limits().await;
    let mut cache_disabled_chunking_configuration =
        crate::common::standard_worker_chunking_configuration();
    // This value deliberately violates the model's persistent-state alignment. Disabled cache
    // execution must never resolve or validate a storage contract that cannot be used.
    cache_disabled_chunking_configuration.prompt_cache_block_tokens = Some(1);
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(16)
            .expect("the test prefill_chunck_tokens should be valid"),
        248_069,
        model_directory.to_path_buf(),
        cache_disabled_chunking_configuration,
        true,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the bounded Ornith engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the complete Ornith model");
    let request_id = RequestId::new(2_000);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                persistent_prompt_cache_eligible_prompt_token_ids(15),
                10,
            )
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
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }

    assert_eq!(
        generated_token_ids,
        vec![12_675, 0, 2_500, 628, 353, 1_438, 488, 3_242, 30, 248_046]
    );
}
