use async_openai::{Client, config::OpenAIConfig};
use serde_json::Value;
use tokio::time::timeout;

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, launch_model_artifact_rest_server_for_model_with_memory_limit,
    stop_model_artifact_rest_server,
};
use super::persistent_prompt_cache_rest_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, PINNED_MODEL_ID, assistant_message, get_json_endpoint,
    prepare_cacheable_romeo_and_juliet_prompt, read_performance_records, required_u64,
    send_streaming_chat_request, user_message, write_cache_pressure_worker_config,
};

const MAXIMUM_MLX_MEMORY_BYTES: u64 = 11_000_000_000;
const CACHEABLE_PROMPT_TOKEN_COUNT: usize = 40_001;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and pinned model for an append-only 40K Romeo and Juliet cache journey"]
async fn should_restore_every_completed_cold_block_for_an_appended_chat_at_eleven_gb_through_rest()
{
    timeout(E2E_TIMEOUT, run_append_only_rest_journey())
        .await
        .expect("the append-only persistent prompt-cache journey must finish within 115 seconds");
}

async fn run_append_only_rest_journey() {
    let qualification_log_prefix = "[persistent-prompt-cache-append-only-rest:11gb]";
    let model_directory = crate::common::configured_model_artifact_directory_by_id(PINNED_MODEL_ID);
    let configured_worker_home = tempfile::tempdir()
        .expect("the append-only REST journey should create an isolated worker home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the append-only REST journey should create a performance log directory");
    write_cache_pressure_worker_config(
        configured_worker_home.path(),
        &model_directory,
        MAXIMUM_MLX_MEMORY_BYTES,
    );
    let prepared_prompt =
        prepare_cacheable_romeo_and_juliet_prompt(&model_directory, CACHEABLE_PROMPT_TOKEN_COUNT);
    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model_with_memory_limit(
        PINNED_MODEL_ID,
        model_directory,
        Some(configured_worker_home.path()),
        Some(performance_log_directory.path()),
        Some(MAXIMUM_MLX_MEMORY_BYTES),
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-qualification-client"),
    );

    eprintln!(
        "{qualification_log_prefix} status=progress phase=cold_request prompt_tokens={} maximum_output_tokens={MAXIMUM_OUTPUT_TOKEN_COUNT}",
        prepared_prompt.prompt_token_count,
    );
    let cold_response = send_streaming_chat_request(
        &openai_client,
        vec![user_message(&prepared_prompt.user_message)],
        &prepared_prompt,
        qualification_log_prefix,
        "cold",
    )
    .await;
    assert!(
        !cold_response.assistant_content.is_empty(),
        "the cold turn must return real assistant content for the appended conversation"
    );
    let cold_cache_stats = get_json_endpoint(server_address, "/v1/cache/stats").await;
    let block_token_count = required_u64(
        &cold_cache_stats,
        "persistent_prompt_cache_block_token_count",
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
            &cold_cache_stats,
            "persistent_prompt_cache_sequence_state_block_count",
        ),
        expected_restored_block_count,
        "the cold turn must synchronously publish block zero and every completed boundary: {cold_cache_stats}"
    );

    eprintln!("{qualification_log_prefix} status=progress phase=appended_warm_request");
    let warm_response = send_streaming_chat_request(
        &openai_client,
        vec![
            user_message(&prepared_prompt.user_message),
            assistant_message(&cold_response.assistant_content),
            user_message(
                "Which decision by Friar Laurence most directly contributes to the tragic outcome? Answer in two concise sentences.",
            ),
        ],
        &prepared_prompt,
        qualification_log_prefix,
        "appended_warm",
    )
    .await;
    assert!(
        warm_response.streamed_output_character_count > 0,
        "the appended warm turn must stream a response"
    );

    let warm_cache_stats = get_json_endpoint(server_address, "/v1/cache/stats").await;
    assert_eq!(
        required_u64(&warm_cache_stats, "persistent_prompt_cache_tokens_saved",),
        expected_restored_token_count,
        "the appended turn must restore the complete block-aligned cold prefix: {warm_cache_stats}"
    );
    let performance_records = read_performance_records(performance_log_directory.path());
    assert_eq!(
        performance_records.len(),
        2,
        "the cold and appended turns should each produce one performance record"
    );
    assert_cold_publication_diagnostics(
        &performance_records[0],
        block_token_count,
        expected_restored_block_count,
    );
    assert_warm_restore_diagnostics(
        &performance_records[1],
        block_token_count,
        expected_restored_block_count,
    );
    let final_status = get_json_endpoint(server_address, "/v1/status").await;
    assert_worker_health_and_memory(&final_status);

    stop_model_artifact_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{qualification_log_prefix} status=success cold_prompt_tokens={} restored_tokens={expected_restored_token_count} restored_blocks={expected_restored_block_count} warm_prompt_tokens={}",
        cold_response.prompt_token_count, warm_response.prompt_token_count,
    );
}

fn assert_cold_publication_diagnostics(
    performance_record: &Value,
    block_token_count: u64,
    expected_published_block_count: u64,
) {
    let cache_diagnostics = &performance_record["persistent_prompt_cache_diagnostics"];
    assert_eq!(cache_diagnostics["block_token_count"], block_token_count);
    assert_eq!(
        cache_diagnostics["published_block_count"],
        expected_published_block_count
    );
}

fn assert_warm_restore_diagnostics(
    performance_record: &Value,
    block_token_count: u64,
    expected_restored_block_count: u64,
) {
    let cache_diagnostics = &performance_record["persistent_prompt_cache_diagnostics"];
    assert_eq!(cache_diagnostics["lookup_outcome"], "hit");
    assert_eq!(cache_diagnostics["block_token_count"], block_token_count);
    assert_eq!(
        cache_diagnostics["matched_sequence_state_block_count"],
        expected_restored_block_count
    );
    assert_eq!(
        cache_diagnostics["restored_block_count"],
        expected_restored_block_count
    );
    let first_missing_block_index =
        cache_diagnostics["first_missing_sequence_state_block_index"].as_u64();
    assert!(
        first_missing_block_index
            .is_none_or(|missing_block_index| missing_block_index >= expected_restored_block_count),
        "the first missing block must not fall within the restorable cold prefix: {cache_diagnostics}"
    );
}

fn assert_worker_health_and_memory(status_document: &Value) {
    assert_eq!(status_document["status"], "ready");
    assert_eq!(
        required_u64(status_document, "mlx_memory_ceiling_bytes"),
        MAXIMUM_MLX_MEMORY_BYTES,
    );
    let memory_snapshot = &status_document["mlx_memory_snapshot"];
    for memory_field_name in ["active_memory_bytes", "peak_memory_bytes"] {
        assert!(
            required_u64(memory_snapshot, memory_field_name) <= MAXIMUM_MLX_MEMORY_BYTES,
            "the worker must remain within the exact 11 GB decimal-SI ceiling: {status_document}"
        );
    }
}
