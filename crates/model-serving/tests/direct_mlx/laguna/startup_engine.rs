use std::fs;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ExpertMemoryMode,
    RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngineError, LagunaServingSettings, MlxInferenceExecution,
    initialize_laguna_execution_with_serving_settings,
};

use super::page_artifact::{write_sparse_artifact, write_sparse_artifact_with_maximum_position};
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_start_a_laguna_engine_from_a_validated_artifact_and_generate_tokens() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("Laguna startup directory");
    write_sparse_artifact(model_directory.path(), false);
    let attribution_directory = tempfile::tempdir().expect("Laguna attribution directory");
    let attribution_log_path = attribution_directory
        .path()
        .join("performance-attribution.jsonl");
    let mut serving_settings = LagunaServingSettings::default_fixed();
    serving_settings.performance_attribution_log_path = Some(attribution_log_path.clone());
    let (generation_processor, mut engine) = initialize_laguna_execution_with_serving_settings(
        model_directory.path(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        true,
        serving_settings,
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
        qwen_thinking_channel_seed: None,
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the Romeo and Juliet chat should prepare");
    assert!(!prepared_generation.prompt_token_ids().is_empty());
    engine
        .start_generation(prepared_generation.into_inference_request())
        .expect("Laguna prompt processing should start");
    engine.inject_two_prefill_capacity_failures_for_test();
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
    let lowered_limit = engine
        .update_mlx_memory_limit(minimum_mlx_memory_ceiling_bytes)
        .expect("the advertised minimum Laguna ceiling should remain executable");
    assert_eq!(
        lowered_limit.effective_mlx_memory_ceiling_bytes(),
        minimum_mlx_memory_ceiling_bytes
    );
    let raised_limit = engine
        .update_mlx_memory_limit(DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES as u64)
        .expect("raising the Laguna ceiling should publish capacity without eager reads");
    assert_eq!(
        raised_limit.effective_mlx_memory_ceiling_bytes(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES as u64
    );
    assert_laguna_attribution_matrix(&attribution_log_path);
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

#[tokio::test]
async fn should_publish_then_restore_an_admitted_romeo_and_juliet_prompt_prefix() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = tempfile::tempdir().expect("Laguna prompt-cache model directory");
    write_sparse_artifact_with_maximum_position(model_directory.path(), false, 1_024);
    let cache_directory = tempfile::tempdir().expect("Laguna prompt-cache directory");
    let mut chunking = crate::common::standard_worker_chunking_configuration();
    chunking.fixed_prompt_processing_chunk_size_tokens = 256;
    chunking.prompt_cache_block_tokens = Some(256);
    chunking.prompt_cache_common_prefix_stride_blocks = 1;
    let mut serving_settings = LagunaServingSettings::default_fixed();
    serving_settings.chunking = Some(chunking);
    serving_settings.persistent_prompt_cache_enabled = true;
    serving_settings.prompt_cache_config = Some(PromptCacheConfig::new(
        cache_directory.path().to_path_buf(),
        10_000_000,
    ));
    let (generation_processor, mut engine) = initialize_laguna_execution_with_serving_settings(
        model_directory.path(),
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        true,
        serving_settings,
    )
    .expect("the cache-enabled Laguna startup should prepare");
    engine
        .load()
        .expect("the cache-enabled Laguna model should load");
    let source_material = include_str!(
        "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
    )
    .split_whitespace()
    .take(400)
    .collect::<Vec<_>>()
    .join(" ");
    let first_request_id = RequestId::new(107);
    let first_command =
        romeo_and_juliet_cache_command(first_request_id, model_directory.path(), &source_material);
    let first_prepared_generation = generation_processor
        .prepare_chat(&first_command)
        .expect("the cold Romeo and Juliet cache prompt should prepare");
    assert!(first_prepared_generation.prompt_token_ids().len() > 256);
    engine
        .start_generation(first_prepared_generation.into_inference_request())
        .expect("the cold Romeo and Juliet cache request should start");
    complete_generation(&mut engine, first_request_id);

    let second_request_id = RequestId::new(108);
    let second_command =
        romeo_and_juliet_cache_command(second_request_id, model_directory.path(), &source_material);
    let second_prepared_generation = generation_processor
        .prepare_chat(&second_command)
        .expect("the warm Romeo and Juliet cache prompt should prepare");
    let warm_start = engine
        .start_generation(second_prepared_generation.into_inference_request())
        .expect("the warm Romeo and Juliet cache request should restore");
    assert!(warm_start.restored_prompt_prefix_token_count() >= 256);
    assert_eq!(
        warm_start.expert_memory_mode(),
        Some(ExpertMemoryMode::Resident),
        "bounded cache restoration must preserve complete expert residency"
    );
    complete_generation(&mut engine, second_request_id);
}

fn romeo_and_juliet_cache_command(
    request_id: RequestId,
    model_directory: &std::path::Path,
    source_material: &str,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id,
        model: model_directory
            .file_name()
            .expect("the synthetic model directory should have a name")
            .to_string_lossy()
            .into_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Summarize this Romeo and Juliet source while preserving the tragic outcome.\n\n{source_material}"
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(107),
            thinking_budget: Some(0),
        },
        qwen_thinking_channel_seed: None,
    }
}

fn complete_generation(
    engine: &mut astronomical_model_serving::LagunaInferenceExecution,
    request_id: RequestId,
) {
    for _advance_attempt in 0..16 {
        match engine
            .decode_next_token(request_id)
            .expect("the Laguna cache request should advance")
        {
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => {
                engine
                    .cancel_generation(request_id)
                    .expect("the cold Laguna cache request should finalize");
                return;
            }
            other => panic!("Laguna emitted an unexpected cache boundary: {other:?}"),
        }
    }
    panic!("the cold Laguna cache request did not finish within bounded advances");
}

fn assert_laguna_attribution_matrix(attribution_log_path: &std::path::Path) {
    let reports = fs::read_to_string(attribution_log_path)
        .expect("Laguna attribution reports should be readable")
        .lines()
        .map(serde_json::from_str::<serde_json::Value>)
        .collect::<Result<Vec<_>, _>>()
        .expect("every Laguna attribution report should be valid JSON");
    let model_loading_report = reports
        .iter()
        .find(|report| report["report_kind"] == "model_loading")
        .expect("Laguna should publish model-loading attribution");
    let generation_report = reports
        .iter()
        .find(|report| report["report_kind"] == "generation")
        .expect("Laguna should publish generation attribution");

    assert_report_operations(
        model_loading_report,
        &[
            "artifact_validation",
            "mlx_runtime_initialization",
            "model_safetensors_mapping",
            "model_tensor_binding",
        ],
    );
    assert_report_operations(
        generation_report,
        &[
            "chat_command_validation",
            "prompt_rendering",
            "prompt_tokenization",
            "prompt_prefill_advance_span",
            "decode_advance_span",
            "attention_forward_span",
            "router_score_selection",
            "generated_token_item_synchronization_wait",
            "generation_finalization",
        ],
    );
    assert_eq!(
        attribution_counter_amount(generation_report, "prefill_capacity_rejection_count",),
        2
    );
    assert_eq!(
        attribution_counter_amount(generation_report, "prefill_capacity_retry_count"),
        1
    );
}

fn attribution_counter_amount(report: &serde_json::Value, counter_identifier: &str) -> u64 {
    report["counters"]
        .as_array()
        .and_then(|counter_reports| {
            counter_reports.iter().find_map(|counter_report| {
                (counter_report["counter"] == counter_identifier)
                    .then(|| counter_report["amount"].as_u64())
                    .flatten()
            })
        })
        .unwrap_or(0)
}

fn assert_report_operations(report: &serde_json::Value, required_operations: &[&str]) {
    let recorded_operations = report["operations"]
        .as_array()
        .expect("attribution operations should be an array");
    for required_operation in required_operations {
        assert!(
            recorded_operations.iter().any(|operation| {
                operation["operation"].as_str() == Some(required_operation)
                    && operation["occurrence_count"].as_u64().unwrap_or(0) > 0
                    && operation["last_ended_offset_nanoseconds"]
                        .as_u64()
                        .is_some()
            }),
            "Laguna attribution did not record start/end timing for {required_operation}"
        );
    }
}
