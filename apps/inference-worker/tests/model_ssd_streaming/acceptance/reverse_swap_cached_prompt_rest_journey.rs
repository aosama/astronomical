//! Public cached-prompt journey guarding physical expert residency across an
//! intervening request on Ornith 1.5 oQ6e.
//!
//! A user first serves a long prompt, then a short unrelated request, then
//! returns to the original prompt with a short append. Success requires valid
//! streamed output within the interactive latency budget without replacing
//! hidden physical page-ins with more logical expert-source traffic.

use std::{fs, path::Path};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, launch_real_model_rest_server_for_models, stop_real_model_rest_server,
};

const OQ6E_MODEL_ID: &str = "Ornith-1.5-35B-A3B-oQ6e-mtp";
const CONFIGURED_MLX_MEMORY_CEILING_BYTES: u64 = 30_000_000_000;
// Five thousand tokens remains a meaningful persistent-cache and model-swap workload while
// leaving enough of the journey deadline for two demand-loaded model transitions.
const LONG_PROMPT_TOKEN_COUNT: usize = 5_000;
const REVERSE_SWAP_TIMEOUT: Duration = Duration::from_secs(35);
const MAXIMUM_LOGICAL_EXPERT_SOURCE_READ_BYTES: u64 = 36_500_000_000;
const MAXIMUM_UNOWNED_PHYSICAL_READ_BYTES: u64 = 5_000_000_000;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches two production model artifacts and proves cached reverse-swap latency and physical residency"]
async fn should_complete_cached_reverse_swap_without_hidden_model_page_ins() {
    timeout(JOURNEY_TIMEOUT, run_cached_reverse_swap_journey())
        .await
        .expect("the cached reverse-swap REST journey must finish within 115 seconds");
}

async fn run_cached_reverse_swap_journey() {
    let oq6e_model_directory =
        crate::common::configured_model_artifact_directory_by_id(OQ6E_MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the reverse-swap worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), [&oq6e_model_directory]);
    let long_prompt = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &oq6e_model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Read this Romeo and Juliet excerpt and reply with exactly one word.",
        LONG_PROMPT_TOKEN_COUNT,
    );
    let model_artifacts = [(OQ6E_MODEL_ID.to_owned(), oq6e_model_directory)];
    let real_model_rest_server = launch_real_model_rest_server_for_models(
        &model_artifacts,
        isolated_worker_home.path(),
        CONFIGURED_MLX_MEMORY_CEILING_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-acceptance-client"),
    );

    complete_request(
        &openai_client,
        OQ6E_MODEL_ID,
        &long_prompt,
        Duration::from_secs(75),
        "oq6_cache_seed",
    )
    .await;
    complete_request(
        &openai_client,
        OQ6E_MODEL_ID,
        "Quote one word from Romeo and Juliet.",
        Duration::from_secs(25),
        "q6_intervening_swap",
    )
    .await;
    let appended_prompt =
        format!("{long_prompt}\n\nNow answer with one different word from Romeo and Juliet.");
    let reverse_swap_elapsed = complete_request(
        &openai_client,
        OQ6E_MODEL_ID,
        &appended_prompt,
        REVERSE_SWAP_TIMEOUT,
        "oq6_cached_append",
    )
    .await;

    stop_real_model_rest_server(real_model_rest_server).await;
    let reverse_swap_report = reverse_swap_generation_report(isolated_worker_home.path());
    let prompt_token_count = counter_amount(&reverse_swap_report, "prompt_token_count");
    let restored_token_count = counter_amount(
        &reverse_swap_report,
        "restored_persistent_prompt_cache_token_count",
    );
    let logical_expert_source_read_bytes =
        counter_amount(&reverse_swap_report, "positional_file_read_byte_count");
    let process_physical_read_bytes = reverse_swap_report["process_physical_disk_read_bytes"]
        .as_u64()
        .expect("reverse-swap attribution should include process physical reads");
    let unowned_physical_read_bytes =
        process_physical_read_bytes.saturating_sub(logical_expert_source_read_bytes);

    assert!(
        reverse_swap_elapsed <= REVERSE_SWAP_TIMEOUT,
        "the cached reverse swap must remain interactive: elapsed={reverse_swap_elapsed:?}"
    );
    assert!(
        restored_token_count > 0 && restored_token_count < prompt_token_count,
        "the reverse swap must restore the cached prefix and process only its append"
    );
    assert!(
        logical_expert_source_read_bytes <= MAXIMUM_LOGICAL_EXPERT_SOURCE_READ_BYTES,
        "the fix must not shift hidden page-ins into excessive explicit expert streaming: logical_bytes={logical_expert_source_read_bytes}"
    );
    assert!(
        unowned_physical_read_bytes <= MAXIMUM_UNOWNED_PHYSICAL_READ_BYTES,
        "physical reads not explained by logical expert streaming must remain bounded: unowned_bytes={unowned_physical_read_bytes}"
    );
    eprintln!(
        "[cached-reverse-swap] status=success elapsed_seconds={:.2} prompt_tokens={prompt_token_count} restored_tokens={restored_token_count} logical_expert_source_read_bytes={logical_expert_source_read_bytes} process_physical_read_bytes={process_physical_read_bytes} unowned_physical_read_bytes={unowned_physical_read_bytes}",
        reverse_swap_elapsed.as_secs_f64(),
    );
}

async fn complete_request(
    openai_client: &Client<OpenAIConfig>,
    model_id: &str,
    user_message: &str,
    request_timeout: Duration,
    phase: &str,
) -> Duration {
    let request_started_at = Instant::now();
    let completion_request = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 1,
        "thinking_budget": 0,
        "max_tokens": 1,
    });
    let request_completion = async {
        let mut streamed_completion: StreamResponse<Value> = openai_client
            .chat()
            .create_stream_byot(completion_request)
            .await
            .unwrap_or_else(|request_error| panic!("{phase} should start: {request_error}"));
        let mut streamed_model_text = String::new();
        let mut finish_reason = None;
        while let Some(stream_item) = streamed_completion.next().await {
            let stream_chunk = stream_item
                .unwrap_or_else(|stream_error| panic!("{phase} should stream: {stream_error}"));
            for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
                if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                    streamed_model_text.push_str(content_fragment);
                }
                if let Some(reason) = choice["finish_reason"].as_str() {
                    finish_reason = Some(reason.to_owned());
                }
            }
        }
        assert!(
            !streamed_model_text.trim().is_empty(),
            "{phase} should return model text"
        );
        assert!(matches!(finish_reason.as_deref(), Some("stop" | "length")));
    };
    timeout(request_timeout, request_completion)
        .await
        .unwrap_or_else(|_| panic!("{phase} must finish within {request_timeout:?}"));
    let request_elapsed = request_started_at.elapsed();
    eprintln!(
        "[cached-reverse-swap] status=progress phase={phase} elapsed_seconds={:.2}",
        request_elapsed.as_secs_f64(),
    );
    request_elapsed
}

fn write_acceptance_config<'a>(
    isolated_worker_home: &Path,
    model_directories: impl IntoIterator<Item = &'a std::path::PathBuf>,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the reverse-swap configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": model_directories.into_iter().collect::<Vec<_>>(),
            "maximum_mlx_memory_gb": 30,
        },
        "prompt_cache": {"enabled": true, "maximum_size_gb": 50},
        "diagnostics": {"performance_attribution_enabled": true},
        "chunking": {"fixed_prompt_processing_chunk_size_tokens": 2_048},
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the reverse-swap configuration should serialize"),
    )
    .expect("the reverse-swap configuration should be written");
}

fn reverse_swap_generation_report(isolated_worker_home: &Path) -> Value {
    latest_attribution_report(isolated_worker_home, "generation")
}

fn latest_attribution_report(isolated_worker_home: &Path, report_kind: &str) -> Value {
    let attribution_log_path =
        isolated_worker_home.join(".astronomical-dev/logs/performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the reverse-swap journey should write performance attribution");
    let matching_attribution_reports = attribution_log
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("each reverse-swap attribution row should be valid JSON")
        })
        .filter(|report| report["report_kind"] == report_kind)
        .collect::<Vec<_>>();
    assert_eq!(matching_attribution_reports.len(), 3);
    matching_attribution_reports
        .into_iter()
        .last()
        .expect("the reverse-swap generation report should exist")
}

fn counter_amount(generation_report: &Value, counter_identifier: &str) -> u64 {
    generation_report["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|counter| counter["counter"] == counter_identifier)
        .and_then(|counter| counter["amount"].as_u64())
        .unwrap_or(0)
}
