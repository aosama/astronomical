use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use tokio::time::timeout;

use super::rest_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, PreparedRomeoAndJulietPrompt, StreamedAssistantResponse,
    THINKING_BUDGET_TOKEN_COUNT, astro_ssd_prompt_cache_pressure_model_id, get_json_endpoint,
    prepare_astro_ssd_prompt_cache_romeo_and_juliet_prompt, read_performance_records, required_u64,
    send_streaming_chat_request, user_message, write_astro_ssd_prompt_cache_pressure_worker_config,
};
use crate::openai_rest;
use crate::openai_rest::E2E_TIMEOUT;
use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{self, ServingRestServer};

// Issue #657 guard: report the block-aligned dense restore without double-counting
// or dropping its contribution from the public usage surface.
const ASTRO_SSD_PROMPT_CACHE_PROMPT_TOKEN_COUNT: usize = 8_192;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and pins the dense-only report of tokens restored from the astro SSD prompt cache"]
async fn should_report_the_exact_dense_restored_prefix_in_warm_request_usage() {
    timeout(E2E_TIMEOUT, run_dense_astro_ssd_prompt_cache_restored_token_report_journey())
        .await
        .expect("the dense astro SSD prompt cache restored-token report journey must finish within 115 seconds");
}

async fn run_dense_astro_ssd_prompt_cache_restored_token_report_journey() {
    let acceptance_log_prefix = "[dense-astro-ssd-prompt-cache-restored-token-report]";
    let model_directory = crate::support::configured_installed_model_directory_by_id(
        astro_ssd_prompt_cache_pressure_model_id(),
    );
    let configured_worker_home =
        tempfile::tempdir().expect("the dense-report journey should create an isolated home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the dense-report journey should create a performance log directory");
    let (prepared_romeo_and_juliet_prompt, model_artifact_rest_server) =
        launch_dense_report_rest_server(
            model_directory,
            configured_worker_home.path(),
            performance_log_directory.path(),
        )
        .await;
    let server_address = model_artifact_rest_server.server_address;
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");

    let cold_response = run_streaming_request_phase(
        &openai_client,
        &prepared_romeo_and_juliet_prompt,
        acceptance_log_prefix,
        "cold",
    )
    .await;
    let expected_restored_token_count =
        expected_block_aligned_restored_token_count(server_address, &cold_response).await;
    let warm_response = run_streaming_request_phase(
        &openai_client,
        &prepared_romeo_and_juliet_prompt,
        acceptance_log_prefix,
        "warm",
    )
    .await;
    let warm_astro_ssd_prompt_cache_restored_token_count =
        assert_warm_request_reported_exact_restored_prefix(
            performance_log_directory.path(),
            expected_restored_token_count,
        );

    let final_status_document = get_json_endpoint(server_address, "/v1/status").await;
    crate::support::memory_utilization_parity::assert_status_memory_ceiling_utilization_closes(
        &final_status_document,
    );

    openai_rest::stop_serving_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{acceptance_log_prefix} status=success prompt_tokens={} astro_ssd_prompt_cache_restored_tokens={warm_astro_ssd_prompt_cache_restored_token_count}",
        warm_response.prompt_token_count,
    );
}

async fn launch_dense_report_rest_server(
    model_directory: PathBuf,
    configured_worker_home: &Path,
    performance_log_directory: &Path,
) -> (PreparedRomeoAndJulietPrompt, ServingRestServer) {
    write_astro_ssd_prompt_cache_pressure_worker_config(
        configured_worker_home,
        &model_directory,
        None,
    );
    let prepared_romeo_and_juliet_prompt = prepare_astro_ssd_prompt_cache_romeo_and_juliet_prompt(
        &model_directory,
        ASTRO_SSD_PROMPT_CACHE_PROMPT_TOKEN_COUNT,
    );
    let model_artifact_rest_server =
        serving_rest::launch_serving_rest_server_for_model_with_memory_limit(
            astro_ssd_prompt_cache_pressure_model_id(),
            model_directory,
            Some(configured_worker_home),
            Some(performance_log_directory),
            None,
        )
        .await;
    (prepared_romeo_and_juliet_prompt, model_artifact_rest_server)
}

async fn run_streaming_request_phase(
    openai_client: &LocalOpenAiClient,
    prepared_romeo_and_juliet_prompt: &PreparedRomeoAndJulietPrompt,
    acceptance_log_prefix: &str,
    request_phase_name: &str,
) -> StreamedAssistantResponse {
    eprintln!("{acceptance_log_prefix} status=progress phase={request_phase_name}_request");
    let streamed_response = send_streaming_chat_request(
        openai_client,
        vec![user_message(&prepared_romeo_and_juliet_prompt.user_message)],
        prepared_romeo_and_juliet_prompt,
        acceptance_log_prefix,
        request_phase_name,
        MAXIMUM_OUTPUT_TOKEN_COUNT,
        THINKING_BUDGET_TOKEN_COUNT,
    )
    .await;
    assert!(
        streamed_response.streamed_output_character_count > 0,
        "the {request_phase_name} request must stream model-generated text"
    );
    streamed_response
}

async fn expected_block_aligned_restored_token_count(
    server_address: SocketAddr,
    cold_response: &StreamedAssistantResponse,
) -> u64 {
    let cold_astro_ssd_prompt_cache_stats_document =
        get_json_endpoint(server_address, "/v1/ssd-prompt-cache/stats").await;
    let block_token_count = required_u64(
        &cold_astro_ssd_prompt_cache_stats_document,
        "astro_ssd_prompt_cache_block_token_count",
    );
    let expected_restored_token_count = cold_response
        .prompt_token_count
        .saturating_sub(1)
        .saturating_div(block_token_count)
        .saturating_mul(block_token_count);
    assert!(
        expected_restored_token_count > 0,
        "the cold prompt must complete at least one astro SSD prompt cache block: block_token_count={block_token_count} prompt_tokens={}",
        cold_response.prompt_token_count,
    );
    expected_restored_token_count
}

fn assert_warm_request_reported_exact_restored_prefix(
    performance_log_directory: &Path,
    expected_restored_token_count: u64,
) -> u64 {
    // The performance record is the authoritative daemon-side observation of
    // the worker Completed event, which is the same value the OpenAI usage surface
    // forwards as prompt_tokens_details.cached_tokens.
    let performance_records = read_performance_records(performance_log_directory);
    assert_eq!(
        performance_records.len(),
        2,
        "the cold and warm requests should each produce one performance record"
    );
    let warm_astro_ssd_prompt_cache_restored_token_count =
        performance_records[1]["astro_ssd_prompt_cache_restored_token_count"]
            .as_u64()
            .expect("the warm performance record should report astro_ssd_prompt_cache_restored_token_count");
    assert_eq!(
        warm_astro_ssd_prompt_cache_restored_token_count, expected_restored_token_count,
        "issue #657 dense-path guard: the warm request must report exactly the block-aligned \
         dense restored prefix (expected {expected_restored_token_count})",
    );
    warm_astro_ssd_prompt_cache_restored_token_count
}
