use async_openai::{Client, config::OpenAIConfig};
use serde_json::Value;
use tokio::time::timeout;

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, launch_model_artifact_rest_server_for_model_with_memory_limit,
    stop_model_artifact_rest_server,
};
use super::persistent_prompt_cache_rest_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, PINNED_MODEL_ID, get_json_endpoint,
    prepare_cacheable_romeo_and_juliet_prompt, required_u64, send_streaming_chat_request,
    user_message, write_cache_pressure_worker_config,
};

// This suite is the user-facing acceptance boundary for cache-pressure behavior: a client sends
// a long OpenAI Chat request, the real worker publishes reusable state, and the next identical
// request restores it. Direct-MLX coverage below the worker is retained separately for allocator
// arithmetic; this test protects the complete server-to-client journey.
const CACHEABLE_PROMPT_TOKEN_COUNT: usize = 10_001;

macro_rules! persistent_prompt_cache_memory_rest_qualification {
    ($test_name:ident, $maximum_mlx_memory_bytes:expr, $expected_expert_memory_mode:expr) => {
        #[tokio::test(flavor = "multi_thread")]
        #[ignore = "launches the production REST server and pinned model for a cold-and-warm 10K Romeo and Juliet cache journey"]
        async fn $test_name() {
            // Every memory cell owns a real model process and GPU. Keep its deadline inside the
            // test so an interrupted external runner cannot leave unbounded qualification work.
            timeout(
                E2E_TIMEOUT,
                run_persistent_prompt_cache_memory_rest_journey(
                    $maximum_mlx_memory_bytes,
                    $expected_expert_memory_mode,
                ),
            )
            .await
            .expect("the persistent prompt-cache REST journey must finish within 115 seconds");
        }
    };
}

persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_eleven_gb_through_rest,
    11_000_000_000,
    Some("paged")
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_eleven_point_five_gb_through_rest,
    11_500_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_twelve_gb_through_rest,
    12_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_twelve_point_five_gb_through_rest,
    12_500_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_thirteen_gb_through_rest,
    13_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_thirteen_point_five_gb_through_rest,
    13_500_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_pinned_romeo_and_juliet_cache_at_fourteen_gb_through_rest,
    14_000_000_000,
    Some("resident")
);

async fn run_persistent_prompt_cache_memory_rest_journey(
    maximum_mlx_memory_bytes: u64,
    expected_expert_memory_mode: Option<&str>,
) {
    let qualification_log_prefix = format!(
        "[persistent-prompt-cache-rest:{}gb]",
        maximum_mlx_memory_bytes as f64 / 1_000_000_000.0
    );
    let model_directory = crate::common::configured_model_artifact_directory_by_id(PINNED_MODEL_ID);
    let configured_worker_home = tempfile::tempdir()
        .expect("the cache-pressure REST journey should create an isolated home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the cache-pressure REST journey should create a performance log directory");
    write_cache_pressure_worker_config(
        configured_worker_home.path(),
        &model_directory,
        maximum_mlx_memory_bytes,
    );
    // Build a textual request instead of injecting token IDs: only the public REST tokenizer
    // may define the actual user journey. The local tokenizer is used solely to size a stable
    // Romeo-and-Juliet fixture near the cache boundary before it is sent as ordinary text.
    let prepared_romeo_and_juliet_prompt =
        prepare_cacheable_romeo_and_juliet_prompt(&model_directory, CACHEABLE_PROMPT_TOKEN_COUNT);
    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model_with_memory_limit(
        PINNED_MODEL_ID,
        model_directory,
        Some(configured_worker_home.path()),
        Some(performance_log_directory.path()),
        Some(maximum_mlx_memory_bytes),
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-qualification-client"),
    );

    // Cold request: publication is a synchronous requirement of successful prompt processing.
    eprintln!(
        "{qualification_log_prefix} status=progress phase=cold_request prompt_tokens={} maximum_output_tokens={MAXIMUM_OUTPUT_TOKEN_COUNT}",
        prepared_romeo_and_juliet_prompt.prompt_token_count,
    );
    let cold_response = send_streaming_chat_request(
        &openai_client,
        vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
        &prepared_romeo_and_juliet_prompt,
        &qualification_log_prefix,
        "cold",
    )
    .await;
    assert!(
        cold_response.streamed_output_character_count > 0,
        "the cold request must stream model-generated text through the public OpenAI-compatible client"
    );
    let cold_status_document = get_json_endpoint(server_address, "/v1/status").await;
    assert_expert_memory_mode(
        &cold_status_document,
        expected_expert_memory_mode,
        &qualification_log_prefix,
    );
    let cold_cache_stats_document = get_json_endpoint(server_address, "/v1/cache/stats").await;
    let block_token_count = required_u64(
        &cold_cache_stats_document,
        "persistent_prompt_cache_block_token_count",
    );
    let expected_published_block_count = cold_response
        .prompt_token_count
        .saturating_sub(1)
        .saturating_div(block_token_count);
    assert!(expected_published_block_count > 0);
    assert!(
        required_u64(
            &cold_cache_stats_document,
            "persistent_prompt_cache_sequence_state_block_count",
        ) >= expected_published_block_count,
        "synchronous publication must expose every completed cold boundary before the stream completes: {cold_cache_stats_document}"
    );

    // Warm request: the same messages and model identity must recover the cold request prefix.
    // Cache counters, rather than timing, are the durable proof because machines differ widely.
    eprintln!("{qualification_log_prefix} status=progress phase=warm_request");
    let warm_response = send_streaming_chat_request(
        &openai_client,
        vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
        &prepared_romeo_and_juliet_prompt,
        &qualification_log_prefix,
        "warm",
    )
    .await;
    assert!(
        warm_response.streamed_output_character_count > 0,
        "the warm request must stream model-generated text through the public OpenAI-compatible client"
    );
    let warm_cache_stats_document = get_json_endpoint(server_address, "/v1/cache/stats").await;
    let restored_prompt_token_count = required_u64(
        &warm_cache_stats_document,
        "persistent_prompt_cache_tokens_saved",
    );
    assert!(
        required_u64(&warm_cache_stats_document, "persistent_prompt_cache_hits") > 0,
        "the warm request must restore the cold request's persistent prompt-cache state: {warm_cache_stats_document}"
    );
    assert!(
        restored_prompt_token_count > 0,
        "the warm request must report restored prompt tokens: {warm_cache_stats_document}"
    );
    stop_model_artifact_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{qualification_log_prefix} status=success cold_output_characters={} warm_output_characters={} restored_prompt_tokens={restored_prompt_token_count}",
        cold_response.streamed_output_character_count,
        warm_response.streamed_output_character_count,
    );
}

fn assert_expert_memory_mode(
    status_document: &Value,
    expected_expert_memory_mode: Option<&str>,
    qualification_log_prefix: &str,
) {
    let observed_expert_memory_mode = status_document["expert_memory_mode"].as_str();
    eprintln!(
        "{qualification_log_prefix} status=progress phase=cold_status expert_memory_mode={observed_expert_memory_mode:?}"
    );
    if let Some(expected_expert_memory_mode) = expected_expert_memory_mode {
        assert_eq!(
            observed_expert_memory_mode,
            Some(expected_expert_memory_mode),
            "the configured memory cell must report its expected expert-memory mode: {status_document}"
        );
    }
}
