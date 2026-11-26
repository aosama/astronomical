use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tokio::time::{Duration, Instant, sleep, timeout};

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

// This suite is the user-facing acceptance boundary for `astro_ssd_prompt_cache` pressure behavior: a client sends
// a long OpenAI Chat request, the real worker publishes reusable state, and the next identical
// request restores it. Direct-MLX coverage below the worker is retained separately for allocator
// arithmetic; this test protects the complete server-to-client journey.
const ASTRO_SSD_PROMPT_CACHE_PROMPT_TOKEN_COUNT: usize = 8_192;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and publishes then restores a Romeo and Juliet astro SSD prompt cache"]
async fn should_publish_and_restore_the_romeo_and_juliet_astro_ssd_prompt_cache_through_rest() {
    timeout(
        E2E_TIMEOUT,
        run_astro_ssd_prompt_cache_memory_rest_journey(
            None,
            None,
            ASTRO_SSD_PROMPT_CACHE_PROMPT_TOKEN_COUNT,
            None,
            MAXIMUM_OUTPUT_TOKEN_COUNT,
            THINKING_BUDGET_TOKEN_COUNT,
        ),
    )
    .await
    .expect("the persistent astro-ssd-prompt-cache REST journey must finish within 115 seconds");
}

async fn run_astro_ssd_prompt_cache_memory_rest_journey(
    maximum_mlx_memory_bytes: Option<u64>,
    expected_expert_memory_mode: Option<&str>,
    astro_ssd_prompt_cache_restored_token_count: usize,
    expected_warm_request_expert_memory_mode: Option<&str>,
    maximum_output_token_count: u16,
    thinking_budget_token_count: u16,
) {
    let acceptance_log_prefix = "[astro-ssd-prompt-cache-rest]";
    let model_directory = crate::support::configured_installed_model_directory_by_id(
        astro_ssd_prompt_cache_pressure_model_id(),
    );
    let configured_worker_home = tempfile::tempdir()
        .expect("the astro SSD prompt cache pressure REST journey should create an isolated home");
    let performance_log_directory = tempfile::tempdir()
        .expect("the astro SSD prompt cache pressure REST journey should create a performance log directory");
    let (prepared_romeo_and_juliet_prompt, model_artifact_rest_server) =
        launch_restore_rest_server(
            model_directory,
            configured_worker_home.path(),
            performance_log_directory.path(),
            maximum_mlx_memory_bytes,
            astro_ssd_prompt_cache_restored_token_count,
        )
        .await;
    let server_address = model_artifact_rest_server.server_address;
    let rest_session = RestJourneySession {
        openai_client: LocalOpenAiClient::new(server_address, "local-acceptance-client"),
        server_address,
        prepared_romeo_and_juliet_prompt,
        acceptance_log_prefix,
        maximum_output_token_count,
        thinking_budget_token_count,
        expected_expert_memory_mode,
    };

    let cold_response = rest_session.run_cold_request().await;
    let warm_response = rest_session
        .run_warm_request(expected_warm_request_expert_memory_mode)
        .await;
    let restored_prompt_token_count = rest_session.assert_warm_request_restored_prompt().await;
    if expected_warm_request_expert_memory_mode.is_some() {
        assert_warm_request_reclaimed_experts_for_restore(performance_log_directory.path());
    }
    let final_status_document = get_json_endpoint(server_address, "/v1/status").await;
    crate::support::memory_utilization_parity::assert_and_preserve_status_memory_ceiling_utilization(
        "astro-ssd-prompt-cache-restore",
        configured_worker_home.path(),
        &final_status_document,
    );
    assert_expert_memory_mode(
        &final_status_document,
        expected_expert_memory_mode,
        acceptance_log_prefix,
        "final_status",
    );
    openai_rest::stop_serving_rest_server(model_artifact_rest_server).await;
    eprintln!(
        "{acceptance_log_prefix} status=success cold_output_characters={} warm_output_characters={} restored_prompt_tokens={restored_prompt_token_count}",
        cold_response.streamed_output_character_count,
        warm_response.streamed_output_character_count,
    );
}

/// The running server plus everything the cold and warm requests of one REST
/// journey have in common.
struct RestJourneySession<'expectation> {
    openai_client: LocalOpenAiClient,
    server_address: SocketAddr,
    prepared_romeo_and_juliet_prompt: PreparedRomeoAndJulietPrompt,
    acceptance_log_prefix: &'static str,
    maximum_output_token_count: u16,
    thinking_budget_token_count: u16,
    expected_expert_memory_mode: Option<&'expectation str>,
}

impl RestJourneySession<'_> {
    async fn send_prompt_request(&self, request_phase_name: &str) -> StreamedAssistantResponse {
        send_streaming_chat_request(
            &self.openai_client,
            vec![user_message(
                &self.prepared_romeo_and_juliet_prompt.user_message,
            )],
            &self.prepared_romeo_and_juliet_prompt,
            self.acceptance_log_prefix,
            request_phase_name,
            self.maximum_output_token_count,
            self.thinking_budget_token_count,
        )
        .await
    }

    /// Cold request: publication is a synchronous requirement of successful prompt processing.
    async fn run_cold_request(&self) -> StreamedAssistantResponse {
        eprintln!(
            "{} status=progress phase=cold_request prompt_tokens={} maximum_output_tokens={}",
            self.acceptance_log_prefix,
            self.prepared_romeo_and_juliet_prompt.prompt_token_count,
            self.maximum_output_token_count,
        );
        let cold_response = self.send_prompt_request("cold").await;
        assert!(
            cold_response.streamed_output_character_count > 0,
            "the cold request must stream model-generated text through the public OpenAI-compatible client"
        );
        let cold_status_document = get_json_endpoint(self.server_address, "/v1/status").await;
        crate::support::memory_utilization_parity::assert_status_memory_ceiling_utilization_closes(
            &cold_status_document,
        );
        assert_expert_memory_mode(
            &cold_status_document,
            self.expected_expert_memory_mode,
            self.acceptance_log_prefix,
            "cold_status",
        );
        let cold_astro_ssd_prompt_cache_stats_document =
            get_json_endpoint(self.server_address, "/v1/ssd-prompt-cache/stats").await;
        let block_token_count = required_u64(
            &cold_astro_ssd_prompt_cache_stats_document,
            "astro_ssd_prompt_cache_block_token_count",
        );
        let expected_published_block_count = cold_response
            .prompt_token_count
            .saturating_sub(1)
            .saturating_div(block_token_count);
        assert!(expected_published_block_count > 0);
        assert!(
            required_u64(
                &cold_astro_ssd_prompt_cache_stats_document,
                "astro_ssd_prompt_cache_sequence_state_block_count",
            ) >= expected_published_block_count,
            "synchronous publication must expose every completed cold boundary before the stream completes: {cold_astro_ssd_prompt_cache_stats_document}"
        );
        cold_response
    }

    /// Warm request: the same messages and model identity must recover the cold request prefix.
    /// `astro_ssd_prompt_cache` counters, rather than timing, are the durable proof because machines differ widely.
    async fn run_warm_request(
        &self,
        expected_warm_request_expert_memory_mode: Option<&str>,
    ) -> StreamedAssistantResponse {
        eprintln!(
            "{} status=progress phase=warm_request",
            self.acceptance_log_prefix
        );
        let warm_response = if let Some(expected_warm_request_expert_memory_mode) =
            expected_warm_request_expert_memory_mode
        {
            let (warm_response, ()) = tokio::join!(
                self.send_prompt_request("warm"),
                wait_for_expert_memory_mode(
                    self.server_address,
                    expected_warm_request_expert_memory_mode,
                    self.acceptance_log_prefix,
                ),
            );
            warm_response
        } else {
            self.send_prompt_request("warm").await
        };
        assert!(
            warm_response.streamed_output_character_count > 0,
            "the warm request must stream model-generated text through the public OpenAI-compatible client"
        );
        warm_response
    }

    async fn assert_warm_request_restored_prompt(&self) -> u64 {
        let warm_astro_ssd_prompt_cache_stats_document =
            get_json_endpoint(self.server_address, "/v1/ssd-prompt-cache/stats").await;
        let restored_prompt_token_count = required_u64(
            &warm_astro_ssd_prompt_cache_stats_document,
            "astro_ssd_prompt_cache_tokens_saved",
        );
        assert!(
            required_u64(
                &warm_astro_ssd_prompt_cache_stats_document,
                "astro_ssd_prompt_cache_hits"
            ) > 0,
            "the warm request must restore the cold request's persistent astro-ssd-prompt-cache state: {warm_astro_ssd_prompt_cache_stats_document}"
        );
        assert!(
            restored_prompt_token_count > 0,
            "the warm request must report restored prompt tokens: {warm_astro_ssd_prompt_cache_stats_document}"
        );
        restored_prompt_token_count
    }
}

async fn launch_restore_rest_server(
    model_directory: PathBuf,
    configured_worker_home: &Path,
    performance_log_directory: &Path,
    maximum_mlx_memory_bytes: Option<u64>,
    astro_ssd_prompt_cache_restored_token_count: usize,
) -> (PreparedRomeoAndJulietPrompt, ServingRestServer) {
    write_astro_ssd_prompt_cache_pressure_worker_config(
        configured_worker_home,
        &model_directory,
        maximum_mlx_memory_bytes,
    );
    // Build a textual request instead of injecting token IDs: only the public REST tokenizer
    // may define the actual user journey. The local tokenizer is used solely to size a stable
    // Romeo-and-Juliet fixture near the `astro_ssd_prompt_cache` boundary before it is sent as ordinary text.
    let prepared_romeo_and_juliet_prompt = prepare_astro_ssd_prompt_cache_romeo_and_juliet_prompt(
        &model_directory,
        astro_ssd_prompt_cache_restored_token_count,
    );
    let model_artifact_rest_server =
        serving_rest::launch_serving_rest_server_for_model_with_memory_limit(
            astro_ssd_prompt_cache_pressure_model_id(),
            model_directory,
            Some(configured_worker_home),
            Some(performance_log_directory),
            maximum_mlx_memory_bytes,
        )
        .await;
    (prepared_romeo_and_juliet_prompt, model_artifact_rest_server)
}

fn assert_warm_request_reclaimed_experts_for_restore(performance_log_directory: &Path) {
    let performance_records = read_performance_records(performance_log_directory);
    assert_eq!(
        performance_records.len(),
        2,
        "the cold and warm requests should each produce one performance record"
    );
    let warm_astro_ssd_prompt_cache_diagnostics =
        &performance_records[1]["astro_ssd_prompt_cache_diagnostics"];
    assert_eq!(
        warm_astro_ssd_prompt_cache_diagnostics["expert_bytes_reclaimed_for_publication"], 0,
        "the long warm request must remain resident through ordinary initial admission so this acceptance isolates restore pressure: {warm_astro_ssd_prompt_cache_diagnostics}"
    );
    assert!(
        warm_astro_ssd_prompt_cache_diagnostics["expert_bytes_reclaimed_for_restore"]
            .as_u64()
            .is_some_and(|reclaimed_expert_bytes| reclaimed_expert_bytes > 0),
        "astro SSD prompt cache reconstruction must reclaim resident experts before loading the long prefix: {warm_astro_ssd_prompt_cache_diagnostics}"
    );
}

async fn wait_for_expert_memory_mode(
    server_address: std::net::SocketAddr,
    expected_expert_memory_mode: &str,
    acceptance_log_prefix: &str,
) {
    let observation_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let observed_expert_memory_mode = status_document["expert_memory_mode"].as_str();
        if observed_expert_memory_mode == Some(expected_expert_memory_mode) {
            eprintln!(
                "{acceptance_log_prefix} status=progress phase=warm_request_mode expert_memory_mode={expected_expert_memory_mode}"
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
    acceptance_log_prefix: &str,
    status_phase_name: &str,
) {
    let observed_expert_memory_mode = status_document["expert_memory_mode"].as_str();
    eprintln!(
        "{acceptance_log_prefix} status=progress phase={status_phase_name} expert_memory_mode={observed_expert_memory_mode:?}"
    );
    if let Some(expected_expert_memory_mode) = expected_expert_memory_mode {
        assert_eq!(
            observed_expert_memory_mode,
            Some(expected_expert_memory_mode),
            "the configured memory cell must report its expected expert-memory mode: {status_document}"
        );
    }
}
