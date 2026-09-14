use async_openai::{Client, config::OpenAIConfig};
use tokio::time::timeout;

use super::rest_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, THINKING_BUDGET_TOKEN_COUNT, cache_pressure_model_id,
    get_json_endpoint, prepare_cacheable_romeo_and_juliet_prompt, read_performance_records,
    required_u64, send_streaming_chat_request, user_message, write_cache_pressure_worker_config,
};
use crate::openai_rest::{E2E_TIMEOUT, stop_serving_rest_server};
use crate::support::serving_rest::launch_serving_rest_server_for_model_with_memory_limit;

// Issue #657 guard for the dense-only reporting path: the usage surface must keep
// reporting exactly the block-aligned dense restore when SpecPrefill is not
// configured. The SpecPrefill reuse-report journey pins the combined dense plus
// sparse total; this journey pins the dense-only value so a future reporting
// change cannot silently double-count or drop the dense component.
const CACHEABLE_PROMPT_TOKEN_COUNT: usize = 8_192;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and pins the dense-only cached-token report"]
async fn should_report_the_exact_dense_restored_prefix_in_warm_request_usage() {
    timeout(E2E_TIMEOUT, run_dense_cached_token_report_journey())
        .await
        .expect("the dense cached-token report journey must finish within 115 seconds");
}

async fn run_dense_cached_token_report_journey() {
    let acceptance_log_prefix = "[dense-cached-token-report]";
    let model_directory =
        crate::support::configured_installed_model_directory_by_id(cache_pressure_model_id());
    let configured_worker_home =
        tempfile::tempdir().expect("the dense-report journey should create an isolated home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the dense-report journey should create a performance log directory");
    write_cache_pressure_worker_config(configured_worker_home.path(), &model_directory, None);
    let prepared_romeo_and_juliet_prompt =
        prepare_cacheable_romeo_and_juliet_prompt(&model_directory, CACHEABLE_PROMPT_TOKEN_COUNT);
    let model_artifact_rest_server = launch_serving_rest_server_for_model_with_memory_limit(
        cache_pressure_model_id(),
        model_directory,
        Some(configured_worker_home.path()),
        Some(performance_log_directory.path()),
        None,
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-acceptance-client"),
    );

    eprintln!("{acceptance_log_prefix} status=progress phase=cold_request");
    let cold_response = send_streaming_chat_request(
        &openai_client,
        vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
        &prepared_romeo_and_juliet_prompt,
        acceptance_log_prefix,
        "cold",
        MAXIMUM_OUTPUT_TOKEN_COUNT,
        THINKING_BUDGET_TOKEN_COUNT,
    )
    .await;
    assert!(
        cold_response.streamed_output_character_count > 0,
        "the cold request must stream model-generated text"
    );

    let cold_cache_stats_document = get_json_endpoint(server_address, "/v1/cache/stats").await;
    let block_token_count = required_u64(
        &cold_cache_stats_document,
        "persistent_prompt_cache_block_token_count",
    );
    let expected_restored_token_count = cold_response
        .prompt_token_count
        .saturating_sub(1)
        .saturating_div(block_token_count)
        .saturating_mul(block_token_count);
    assert!(
        expected_restored_token_count > 0,
        "the cold prompt must complete at least one cache block: block_token_count={block_token_count} prompt_tokens={}",
        cold_response.prompt_token_count,
    );

    eprintln!("{acceptance_log_prefix} status=progress phase=warm_request");
    let warm_response = send_streaming_chat_request(
        &openai_client,
        vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
        &prepared_romeo_and_juliet_prompt,
        acceptance_log_prefix,
        "warm",
        MAXIMUM_OUTPUT_TOKEN_COUNT,
        THINKING_BUDGET_TOKEN_COUNT,
    )
    .await;
    assert!(
        warm_response.streamed_output_character_count > 0,
        "the warm request must stream model-generated text"
    );

    // The performance record is the authoritative supervisor-side observation of
    // the worker Completed event, which is the same value the OpenAI usage surface
    // forwards as prompt_tokens_details.cached_tokens.
    let performance_records = read_performance_records(performance_log_directory.path());
    assert_eq!(
        performance_records.len(),
        2,
        "the cold and warm requests should each produce one performance record"
    );
    let warm_cached_token_count = performance_records[1]["cached_token_count"]
        .as_u64()
        .expect("the warm performance record should report cached_token_count");
    assert_eq!(
        warm_cached_token_count, expected_restored_token_count,
        "issue #657 dense-path guard: the warm request must report exactly the block-aligned \
         dense restored prefix (expected {expected_restored_token_count})",
    );

    let final_status_document = get_json_endpoint(server_address, "/v1/status").await;
    crate::support::memory_utilization_parity::assert_status_memory_ceiling_utilization_closes(
        &final_status_document,
    );

    stop_serving_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{acceptance_log_prefix} status=success prompt_tokens={} cached_tokens={warm_cached_token_count}",
        warm_response.prompt_token_count,
    );
}
