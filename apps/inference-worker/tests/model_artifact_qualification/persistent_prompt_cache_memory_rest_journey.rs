use async_openai::{Client, config::OpenAIConfig};
use serde_json::Value;
use tokio::time::{Duration, Instant, sleep, timeout};

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, launch_model_artifact_rest_server_for_model_with_memory_limit,
    stop_model_artifact_rest_server,
};
use super::persistent_prompt_cache_rest_support::{
    CACHE_PRESSURE_MODEL_ID, MAXIMUM_OUTPUT_TOKEN_COUNT, THINKING_BUDGET_TOKEN_COUNT,
    get_json_endpoint, prepare_cacheable_romeo_and_juliet_prompt, read_performance_records,
    required_u64, send_streaming_chat_request, user_message, write_cache_pressure_worker_config,
};

// This suite is the user-facing acceptance boundary for cache-pressure behavior: a client sends
// a long OpenAI Chat request, the real worker publishes reusable state, and the next identical
// request restores it. Direct-MLX coverage below the worker is retained separately for allocator
// arithmetic; this test protects the complete server-to-client journey.
const CACHEABLE_PROMPT_TOKEN_COUNT: usize = 10_001;
const RESIDENT_RESTORE_PRESSURE_PROMPT_TOKEN_COUNT: usize = 90_001;
const RESIDENT_RESTORE_PRESSURE_MLX_MEMORY_BYTES: u64 = 38_000_000_000;
const RESIDENT_RESTORE_PRESSURE_MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 16;
const RESIDENT_RESTORE_PRESSURE_THINKING_BUDGET_TOKEN_COUNT: u16 = 8;

macro_rules! persistent_prompt_cache_memory_rest_qualification {
    ($test_name:ident, $maximum_mlx_memory_bytes:expr, $expected_expert_memory_mode:expr) => {
        #[tokio::test(flavor = "multi_thread")]
        #[ignore = "launches the production REST server and reference model for a cold-and-warm 10K Romeo and Juliet cache journey"]
        async fn $test_name() {
            // Every memory cell owns a real model process and GPU. Keep its deadline inside the
            // test so an interrupted external runner cannot leave unbounded qualification work.
            timeout(
                E2E_TIMEOUT,
                run_persistent_prompt_cache_memory_rest_journey(
                    $maximum_mlx_memory_bytes,
                    $expected_expert_memory_mode,
                    CACHEABLE_PROMPT_TOKEN_COUNT,
                    None,
                    MAXIMUM_OUTPUT_TOKEN_COUNT,
                    THINKING_BUDGET_TOKEN_COUNT,
                ),
            )
            .await
            .expect("the persistent prompt-cache REST journey must finish within 115 seconds");
        }
    };
}

persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_twenty_three_gb_through_rest,
    23_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_twenty_five_gb_through_rest,
    25_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_twenty_eight_gb_through_rest,
    28_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_thirty_gb_through_rest,
    30_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_thirty_two_gb_through_rest,
    32_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_thirty_five_gb_through_rest,
    35_000_000_000,
    None
);
persistent_prompt_cache_memory_rest_qualification!(
    should_publish_and_restore_the_romeo_and_juliet_cache_at_thirty_eight_gb_through_rest,
    38_000_000_000,
    Some("resident")
);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and proves a long cache restore demotes resident experts before allocation"]
async fn should_demote_resident_experts_before_restoring_a_long_cached_prompt_at_thirty_eight_gb_through_rest()
 {
    timeout(
        E2E_TIMEOUT,
        run_persistent_prompt_cache_memory_rest_journey(
            RESIDENT_RESTORE_PRESSURE_MLX_MEMORY_BYTES,
            Some("resident"),
            RESIDENT_RESTORE_PRESSURE_PROMPT_TOKEN_COUNT,
            Some("paged"),
            RESIDENT_RESTORE_PRESSURE_MAXIMUM_OUTPUT_TOKEN_COUNT,
            RESIDENT_RESTORE_PRESSURE_THINKING_BUDGET_TOKEN_COUNT,
        ),
    )
    .await
    .expect("the resident-to-paged cache-restore journey must finish within 115 seconds");
}

async fn run_persistent_prompt_cache_memory_rest_journey(
    maximum_mlx_memory_bytes: u64,
    expected_expert_memory_mode: Option<&str>,
    cacheable_prompt_token_count: usize,
    expected_warm_request_expert_memory_mode: Option<&str>,
    maximum_output_token_count: u16,
    thinking_budget_token_count: u16,
) {
    let qualification_log_prefix = format!(
        "[persistent-prompt-cache-rest:{}gb]",
        maximum_mlx_memory_bytes as f64 / 1_000_000_000.0
    );
    let model_directory =
        crate::common::configured_model_artifact_directory_by_id(CACHE_PRESSURE_MODEL_ID);
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
        prepare_cacheable_romeo_and_juliet_prompt(&model_directory, cacheable_prompt_token_count);
    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model_with_memory_limit(
        CACHE_PRESSURE_MODEL_ID,
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
        "{qualification_log_prefix} status=progress phase=cold_request prompt_tokens={} maximum_output_tokens={maximum_output_token_count}",
        prepared_romeo_and_juliet_prompt.prompt_token_count,
    );
    let cold_response = send_streaming_chat_request(
        &openai_client,
        vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
        &prepared_romeo_and_juliet_prompt,
        &qualification_log_prefix,
        "cold",
        maximum_output_token_count,
        thinking_budget_token_count,
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
        "cold_status",
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
    let warm_response = if let Some(expected_warm_request_expert_memory_mode) =
        expected_warm_request_expert_memory_mode
    {
        let (warm_response, ()) = tokio::join!(
            send_streaming_chat_request(
                &openai_client,
                vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
                &prepared_romeo_and_juliet_prompt,
                &qualification_log_prefix,
                "warm",
                maximum_output_token_count,
                thinking_budget_token_count,
            ),
            wait_for_expert_memory_mode(
                server_address,
                expected_warm_request_expert_memory_mode,
                &qualification_log_prefix,
            ),
        );
        warm_response
    } else {
        send_streaming_chat_request(
            &openai_client,
            vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
            &prepared_romeo_and_juliet_prompt,
            &qualification_log_prefix,
            "warm",
            maximum_output_token_count,
            thinking_budget_token_count,
        )
        .await
    };
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
    if expected_warm_request_expert_memory_mode.is_some() {
        let performance_records = read_performance_records(performance_log_directory.path());
        assert_eq!(
            performance_records.len(),
            2,
            "the cold and warm requests should each produce one performance record"
        );
        let warm_cache_diagnostics = &performance_records[1]["persistent_prompt_cache_diagnostics"];
        assert_eq!(
            warm_cache_diagnostics["expert_bytes_reclaimed_for_publication"], 0,
            "the long warm request must remain resident through ordinary initial admission so this qualification isolates restore pressure: {warm_cache_diagnostics}"
        );
        assert!(
            warm_cache_diagnostics["expert_bytes_reclaimed_for_restore"]
                .as_u64()
                .is_some_and(|reclaimed_expert_bytes| reclaimed_expert_bytes > 0),
            "cache reconstruction must reclaim resident experts before loading the long prefix: {warm_cache_diagnostics}"
        );
    }
    let final_status_document = get_json_endpoint(server_address, "/v1/status").await;
    assert_expert_memory_mode(
        &final_status_document,
        expected_expert_memory_mode,
        &qualification_log_prefix,
        "final_status",
    );
    stop_model_artifact_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{qualification_log_prefix} status=success cold_output_characters={} warm_output_characters={} restored_prompt_tokens={restored_prompt_token_count}",
        cold_response.streamed_output_character_count,
        warm_response.streamed_output_character_count,
    );
}

async fn wait_for_expert_memory_mode(
    server_address: std::net::SocketAddr,
    expected_expert_memory_mode: &str,
    qualification_log_prefix: &str,
) {
    let observation_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let observed_expert_memory_mode = status_document["expert_memory_mode"].as_str();
        if observed_expert_memory_mode == Some(expected_expert_memory_mode) {
            eprintln!(
                "{qualification_log_prefix} status=progress phase=warm_request_mode expert_memory_mode={expected_expert_memory_mode}"
            );
            return;
        }
        assert!(
            Instant::now() < observation_deadline,
            "the active warm request did not transition to {expected_expert_memory_mode}: {status_document}"
        );
        sleep(Duration::from_millis(20)).await;
    }
}

fn assert_expert_memory_mode(
    status_document: &Value,
    expected_expert_memory_mode: Option<&str>,
    qualification_log_prefix: &str,
    status_phase_name: &str,
) {
    let observed_expert_memory_mode = status_document["expert_memory_mode"].as_str();
    eprintln!(
        "{qualification_log_prefix} status=progress phase={status_phase_name} expert_memory_mode={observed_expert_memory_mode:?}"
    );
    if let Some(expected_expert_memory_mode) = expected_expert_memory_mode {
        assert_eq!(
            observed_expert_memory_mode,
            Some(expected_expert_memory_mode),
            "the configured memory cell must report its expected expert-memory mode: {status_document}"
        );
    }
}
