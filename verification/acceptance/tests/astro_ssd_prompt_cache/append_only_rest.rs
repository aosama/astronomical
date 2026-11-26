use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tokio::time::timeout;

use super::rest_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, PreparedRomeoAndJulietPrompt, StreamedAssistantResponse,
    THINKING_BUDGET_TOKEN_COUNT, assistant_message, astro_ssd_prompt_cache_pressure_model_id,
    get_json_endpoint, prepare_astro_ssd_prompt_cache_romeo_and_juliet_prompt,
    read_performance_records, required_u64, send_streaming_chat_request, user_message,
    write_astro_ssd_prompt_cache_pressure_worker_config,
};
use crate::openai_rest;
use crate::openai_rest::E2E_TIMEOUT;
use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{self, ServingRestServer};

const ASTRO_SSD_PROMPT_CACHE_PROMPT_TOKEN_COUNT: usize = 8_192;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and restores completed cold astro SSD prompt cache blocks on an appended Romeo chat"]
async fn should_restore_every_completed_cold_block_for_an_appended_chat_through_rest() {
    timeout(E2E_TIMEOUT, run_append_only_rest_journey())
        .await
        .expect("the append-only persistent astro-ssd-prompt-cache journey must finish within 115 seconds");
}

async fn run_append_only_rest_journey() {
    let acceptance_log_prefix = "[astro-ssd-prompt-cache-append-only-rest]";
    let model_directory = crate::support::configured_installed_model_directory_by_id(
        astro_ssd_prompt_cache_pressure_model_id(),
    );
    let configured_worker_home = tempfile::tempdir()
        .expect("the append-only REST journey should create an isolated worker home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the append-only REST journey should create a performance log directory");
    let (prepared_prompt, model_artifact_rest_server) = launch_append_only_rest_server(
        model_directory,
        configured_worker_home.path(),
        performance_log_directory.path(),
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");

    let cold_turn = run_cold_turn(
        &openai_client,
        &prepared_prompt,
        server_address,
        acceptance_log_prefix,
    )
    .await;

    eprintln!("{acceptance_log_prefix} status=progress phase=appended_warm_request");
    let warm_response = send_appended_warm_request(
        &openai_client,
        &prepared_prompt,
        &cold_turn.response,
        acceptance_log_prefix,
    )
    .await;
    assert!(
        warm_response.streamed_output_character_count > 0,
        "the appended warm turn must stream a response"
    );

    assert_warm_turn_restored_cold_prefix(
        server_address,
        performance_log_directory.path(),
        &cold_turn,
    )
    .await;
    let final_status = get_json_endpoint(server_address, "/v1/status").await;
    assert_worker_health_and_memory(&final_status);

    openai_rest::stop_serving_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{acceptance_log_prefix} status=success cold_prompt_tokens={} restored_tokens={} restored_blocks={} warm_prompt_tokens={}",
        cold_turn.response.prompt_token_count,
        cold_turn.expected_restored_token_count,
        cold_turn.expected_restored_block_count,
        warm_response.prompt_token_count,
    );
}

/// What the cold turn proves the `astro_ssd_prompt_cache` published, so the
/// appended warm turn can be held to restoring exactly that block-aligned prefix.
struct ColdTurnOutcome {
    response: StreamedAssistantResponse,
    block_token_count: u64,
    expected_restored_token_count: u64,
    expected_restored_block_count: u64,
}

async fn launch_append_only_rest_server(
    model_directory: PathBuf,
    configured_worker_home: &Path,
    performance_log_directory: &Path,
) -> (PreparedRomeoAndJulietPrompt, ServingRestServer) {
    write_astro_ssd_prompt_cache_pressure_worker_config(
        configured_worker_home,
        &model_directory,
        None,
    );
    let prepared_prompt = prepare_astro_ssd_prompt_cache_romeo_and_juliet_prompt(
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
    (prepared_prompt, model_artifact_rest_server)
}

async fn run_cold_turn(
    openai_client: &LocalOpenAiClient,
    prepared_prompt: &PreparedRomeoAndJulietPrompt,
    server_address: SocketAddr,
    acceptance_log_prefix: &str,
) -> ColdTurnOutcome {
    eprintln!(
        "{acceptance_log_prefix} status=progress phase=cold_request prompt_tokens={} maximum_output_tokens={MAXIMUM_OUTPUT_TOKEN_COUNT}",
        prepared_prompt.prompt_token_count,
    );
    let cold_response = send_streaming_chat_request(
        openai_client,
        vec![user_message(&prepared_prompt.user_message)],
        prepared_prompt,
        acceptance_log_prefix,
        "cold",
        MAXIMUM_OUTPUT_TOKEN_COUNT,
        THINKING_BUDGET_TOKEN_COUNT,
    )
    .await;
    assert!(
        !cold_response.assistant_content.is_empty(),
        "the cold turn must return real assistant content for the appended conversation"
    );
    let cold_astro_ssd_prompt_cache_stats =
        get_json_endpoint(server_address, "/v1/ssd-prompt-cache/stats").await;
    let block_token_count = required_u64(
        &cold_astro_ssd_prompt_cache_stats,
        "astro_ssd_prompt_cache_block_token_count",
    );
    assert!(
        block_token_count > 0,
        "the active model must expose its block geometry"
    );
    let expected_restored_token_count = cold_response
        .prompt_token_count
        .saturating_sub(1)
        .saturating_div(block_token_count)
        .saturating_mul(block_token_count);
    let expected_restored_block_count = expected_restored_token_count / block_token_count;
    assert!(expected_restored_block_count > 0);
    assert_eq!(
        required_u64(
            &cold_astro_ssd_prompt_cache_stats,
            "astro_ssd_prompt_cache_sequence_state_block_count",
        ),
        expected_restored_block_count,
        "the cold turn must synchronously publish block zero and every completed boundary: {cold_astro_ssd_prompt_cache_stats}"
    );
    ColdTurnOutcome {
        response: cold_response,
        block_token_count,
        expected_restored_token_count,
        expected_restored_block_count,
    }
}

async fn send_appended_warm_request(
    openai_client: &LocalOpenAiClient,
    prepared_prompt: &PreparedRomeoAndJulietPrompt,
    cold_response: &StreamedAssistantResponse,
    acceptance_log_prefix: &str,
) -> StreamedAssistantResponse {
    send_streaming_chat_request(
        openai_client,
        vec![
            user_message(&prepared_prompt.user_message),
            assistant_message(&cold_response.assistant_content),
            user_message(
                "Which decision by Friar Laurence most directly contributes to the tragic outcome? Answer in two concise sentences.",
            ),
        ],
        prepared_prompt,
        acceptance_log_prefix,
        "appended_warm",
        MAXIMUM_OUTPUT_TOKEN_COUNT,
        THINKING_BUDGET_TOKEN_COUNT,
    )
    .await
}

async fn assert_warm_turn_restored_cold_prefix(
    server_address: SocketAddr,
    performance_log_directory: &Path,
    cold_turn: &ColdTurnOutcome,
) {
    let warm_astro_ssd_prompt_cache_stats =
        get_json_endpoint(server_address, "/v1/ssd-prompt-cache/stats").await;
    assert_eq!(
        required_u64(
            &warm_astro_ssd_prompt_cache_stats,
            "astro_ssd_prompt_cache_tokens_saved",
        ),
        cold_turn.expected_restored_token_count,
        "the appended turn must restore the complete block-aligned cold prefix: {warm_astro_ssd_prompt_cache_stats}"
    );
    let performance_records = read_performance_records(performance_log_directory);
    assert_eq!(
        performance_records.len(),
        2,
        "the cold and appended turns should each produce one performance record"
    );
    assert_cold_publication_diagnostics(
        &performance_records[0],
        cold_turn.block_token_count,
        cold_turn.expected_restored_block_count,
    );
    assert_warm_restore_diagnostics(
        &performance_records[1],
        cold_turn.block_token_count,
        cold_turn.expected_restored_block_count,
    );
}

fn assert_cold_publication_diagnostics(
    performance_record: &Value,
    block_token_count: u64,
    expected_published_block_count: u64,
) {
    let astro_ssd_prompt_cache_diagnostics =
        &performance_record["astro_ssd_prompt_cache_diagnostics"];
    assert_eq!(
        astro_ssd_prompt_cache_diagnostics["block_token_count"],
        block_token_count
    );
    assert_eq!(
        astro_ssd_prompt_cache_diagnostics["published_block_count"],
        expected_published_block_count
    );
}

fn assert_warm_restore_diagnostics(
    performance_record: &Value,
    block_token_count: u64,
    expected_restored_block_count: u64,
) {
    let astro_ssd_prompt_cache_diagnostics =
        &performance_record["astro_ssd_prompt_cache_diagnostics"];
    assert_eq!(astro_ssd_prompt_cache_diagnostics["lookup_outcome"], "hit");
    assert_eq!(
        astro_ssd_prompt_cache_diagnostics["block_token_count"],
        block_token_count
    );
    assert_eq!(
        astro_ssd_prompt_cache_diagnostics["matched_sequence_state_block_count"],
        expected_restored_block_count
    );
    assert_eq!(
        astro_ssd_prompt_cache_diagnostics["restored_block_count"],
        expected_restored_block_count
    );
    let first_missing_block_index =
        astro_ssd_prompt_cache_diagnostics["first_missing_sequence_state_block_index"].as_u64();
    assert!(
        first_missing_block_index
            .is_none_or(|missing_block_index| missing_block_index >= expected_restored_block_count),
        "the first missing block must not fall within the restorable cold prefix: {astro_ssd_prompt_cache_diagnostics}"
    );
}

fn assert_worker_health_and_memory(status_document: &Value) {
    assert_eq!(status_document["status"], "ready");
    let mlx_memory_ceiling_bytes = required_u64(status_document, "mlx_memory_ceiling_bytes");
    let memory_snapshot = &status_document["mlx_memory_snapshot"];
    for memory_field_name in ["active_memory_bytes", "peak_memory_bytes"] {
        assert!(
            required_u64(memory_snapshot, memory_field_name) <= mlx_memory_ceiling_bytes,
            "the worker must remain within the reported MLX ceiling: {status_document}"
        );
    }
}
