//! Production-worker journeys for artifact-derived Qwen3.5 engine selection.

use std::{fs, path::Path, time::Duration};

use astronomical_model_serving::{
    CompleteResidencyHeadroomBoundary, Qwen3_5ArtifactValidator,
    mlx_ram_budget_model_geometry_from_validated_artifact,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, put_json_endpoint,
    stop_real_model_rest_server,
};
use crate::support::{
    artifact_bytes::acceptance_model_scan_root, configured_installed_model_directory_by_id,
    exact_model_prompt, large_sparse_moe_model_id, resident_sparse_moe_model_id,
};

const MAXIMUM_REAL_MODEL_MEMORY_BYTES: u64 = 36_000_000_000;
const FIXED_PREFILL_CHUNK_SIZE_TOKENS: u32 = 2_048;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 8;
const SMALL_PROMPT_TOKEN_COUNT: usize = 256;
const CACHE_SEED_PROMPT_TOKEN_COUNT: usize = FIXED_PREFILL_CHUNK_SIZE_TOKENS as usize;
const CACHE_APPEND_PROMPT_TOKEN_COUNT: usize = 512;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const RESIDENT_SELECTION_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "starts the production worker and loads the configured sparse artifact as resident"]
#[serial]
async fn should_select_resident_engine_when_artifact_geometry_and_prompt_fit() {
    timeout(RESIDENT_SELECTION_TIMEOUT, async {
        let model_id = resident_sparse_moe_model_id();
        let model_directory = configured_installed_model_directory_by_id(model_id);
        let residency_geometry =
            measure_sparse_artifact_residency_geometry(&model_directory);
        let prompt_content = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Explain how haste shapes the tragedy in one concise sentence.",
            SMALL_PROMPT_TOKEN_COUNT,
        );
        let prompt_context_bytes =
            context_memory_reservation_bytes(&model_directory, SMALL_PROMPT_TOKEN_COUNT + 2);
        let resident_memory_ceiling_bytes = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_add(prompt_context_bytes);
        assert!(
            resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
            "the resident acceptance cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
        );

        let worker_home = tempfile::tempdir()
            .expect("the resident engine-selection worker home should be created");
        write_acceptance_config(
            worker_home.path(),
            &model_directory,
            model_id,
            resident_memory_ceiling_bytes,
            false,
        );
        let real_model_rest_server = launch_real_model_rest_server(
            model_id,
            model_directory,
            worker_home.path(),
            resident_memory_ceiling_bytes,
        )
        .await;
        complete_chat_request(
            real_model_rest_server.server_address,
            model_id,
            json!([{"role": "user", "content": prompt_content}]),
            "resident-selection",
        )
        .await;
        let status_document =
            get_json_endpoint(real_model_rest_server.server_address, "/v1/status").await;
        assert_eq!(status_document["expert_memory_mode"], "resident");
        assert_eq!(status_document["activity"], "idle");
        let resident_expert_payload_bytes = status_document["expert_residency"]
            ["resident_expert_payload_bytes"]
            .as_u64()
            .expect("resident status should report its owned expert payload");
        assert!(resident_expert_payload_bytes > 0);
        stop_real_model_rest_server(real_model_rest_server).await;
    })
    .await
    .expect("the resident engine-selection journey must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "starts the production worker with an artifact-derived ceiling below complete-residency headroom"]
#[serial]
async fn should_select_streaming_engine_when_complete_residency_headroom_does_not_fit() {
    timeout(SSD_JOURNEY_TIMEOUT, async {
        let model_id = large_sparse_moe_model_id();
        let model_directory = configured_installed_model_directory_by_id(model_id);
        let residency_geometry =
            measure_sparse_artifact_residency_geometry(&model_directory);
        let streaming_memory_ceiling_bytes = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_sub(1)
            .min(MAXIMUM_REAL_MODEL_MEMORY_BYTES);
        let prompt_content = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Explain how haste shapes the tragedy in one concise sentence.",
            SMALL_PROMPT_TOKEN_COUNT,
        );
        let worker_home = tempfile::tempdir()
            .expect("the streaming engine-selection worker home should be created");
        write_acceptance_config(
            worker_home.path(),
            &model_directory,
            model_id,
            streaming_memory_ceiling_bytes,
            false,
        );
        let real_model_rest_server = launch_real_model_rest_server(
            model_id,
            model_directory,
            worker_home.path(),
            streaming_memory_ceiling_bytes,
        )
        .await;
        complete_chat_request(
            real_model_rest_server.server_address,
            model_id,
            json!([{"role": "user", "content": prompt_content}]),
            "streaming-selection",
        )
        .await;
        let status_document =
            get_json_endpoint(real_model_rest_server.server_address, "/v1/status").await;
        assert!(
            matches!(
                status_document["expert_memory_mode"].as_str(),
                Some("paged" | "hybrid")
            ),
            "the engine selected below complete-residency headroom must remain nonresident: {status_document}"
        );
        assert!(
            status_document["expert_residency"].get("complete_layer_count").is_none()
                && status_document["expert_residency"]
                    .get("partial_layer_count")
                    .is_none(),
            "streaming status must not publish complete-layer ownership: {status_document}"
        );
        stop_real_model_rest_server(real_model_rest_server).await;
    })
    .await
    .expect("the streaming engine-selection SSD journey must finish within 60 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident engine, forces one invisible streaming retry, and checks persisted-prefix reuse"]
#[serial]
async fn should_restore_cached_prefix_after_an_invisible_resident_to_streaming_retry() {
    timeout(SSD_JOURNEY_TIMEOUT, async {
        let model_id = resident_sparse_moe_model_id();
        let model_directory = configured_installed_model_directory_by_id(model_id);
        let residency_geometry =
            measure_sparse_artifact_residency_geometry(&model_directory);
        let cache_seed_context_bytes =
            context_memory_reservation_bytes(&model_directory, CACHE_SEED_PROMPT_TOKEN_COUNT + 2);
        let resident_memory_ceiling_bytes = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_add(cache_seed_context_bytes);
        assert!(
            resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
            "the retry/cache acceptance cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
        );
        let initial_user_message = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Read this Romeo and Juliet excerpt and answer with one concise sentence.",
            CACHE_SEED_PROMPT_TOKEN_COUNT,
        );
        let appended_user_message = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Use this additional Romeo and Juliet context without quoting it.",
            CACHE_APPEND_PROMPT_TOKEN_COUNT,
        );
        let worker_home = tempfile::tempdir()
            .expect("the retry/cache worker home should be created");
        write_acceptance_config(
            worker_home.path(),
            &model_directory,
            model_id,
            resident_memory_ceiling_bytes,
            true,
        );
        let real_model_rest_server = launch_real_model_rest_server(
            model_id,
            model_directory,
            worker_home.path(),
            resident_memory_ceiling_bytes,
        )
        .await;
        let server_address = real_model_rest_server.server_address;
        let first_assistant_response = complete_chat_request(
            server_address,
            model_id,
            json!([{"role": "user", "content": initial_user_message}]),
            "cache-seed",
        )
        .await;
        let streaming_retry_memory_ceiling_gb = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_sub(1)
            / 1_000_000_000;
        let lowered_memory_ceiling_document = put_json_endpoint(
            server_address,
            "/v1/config/maximum-mlx-memory",
            &json!({"maximum_mlx_memory_gb": streaming_retry_memory_ceiling_gb}),
        )
        .await;
        assert_eq!(
            lowered_memory_ceiling_document["configured_maximum_mlx_memory_gb"].as_u64(),
            Some(streaming_retry_memory_ceiling_gb)
        );
        assert!(
            lowered_memory_ceiling_document["pending_mlx_memory_ceiling_bytes"].is_null(),
            "the retry should start after the lower memory ceiling is effective: {lowered_memory_ceiling_document}"
        );
        complete_chat_request(
            server_address,
            model_id,
            json!([
                {"role": "user", "content": initial_user_message},
                {"role": "assistant", "content": first_assistant_response},
                {"role": "user", "content": appended_user_message}
            ]),
            "resident-to-streaming-retry",
        )
        .await;
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        assert!(
            matches!(
                status_document["expert_memory_mode"].as_str(),
                Some("paged" | "hybrid")
            ),
            "the pre-output resident failure must leave the worker on its streaming retry engine: {status_document}"
        );
        stop_real_model_rest_server(real_model_rest_server).await;

        let generation_reports =
            read_generation_attribution_reports(worker_home.path());
        assert!(
            generation_reports.len() >= 2,
            "both user-visible requests should have generation attribution reports"
        );
        let report_outcomes_and_operation_names = generation_reports
            .iter()
            .map(|report| {
                let operation_names = report["operations"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|operation| operation["operation"].as_str())
                    .collect::<Vec<_>>();
                (report["outcome"].as_str(), operation_names)
            })
            .collect::<Vec<_>>();
        let retry_report = generation_reports
            .iter()
            .rev()
            .find(|report| {
                report["operations"].as_array().is_some_and(|operations| {
                    operations.iter().any(|operation| {
                        operation["operation"] == "resident_to_streaming_retry"
                    })
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "the retried request should attribute the resident-to-streaming swap; report_outcomes_and_operations={report_outcomes_and_operation_names:?}"
                )
            });
        let prompt_token_count = attribution_counter(retry_report, "prompt_token_count");
        let restored_prefix_token_count = attribution_counter(
            retry_report,
            "restored_persistent_prompt_cache_token_count",
        );
        assert!(
            restored_prefix_token_count > 0 && restored_prefix_token_count < prompt_token_count,
            "the retry must restore a cached prefix and process only its uncached tail: prompt={prompt_token_count} restored={restored_prefix_token_count}"
        );
        let retry_operation = retry_report["operations"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|operation| operation["operation"] == "resident_to_streaming_retry")
            .expect("the generation report should attribute the resident-to-streaming swap");
        assert_eq!(retry_operation["occurrence_count"], 1);
        assert!(
            retry_operation["total_elapsed_nanoseconds"]
                .as_u64()
                .is_some_and(|elapsed_nanoseconds| elapsed_nanoseconds > 0),
            "the retry operation should include a nonzero elapsed interval"
        );
    })
    .await
    .expect("the retry/cache SSD journey must finish within 60 seconds");
}

struct SparseArtifactResidencyGeometry {
    complete_residency_with_headroom_bytes: u64,
}

fn measure_sparse_artifact_residency_geometry(
    model_directory: &Path,
) -> SparseArtifactResidencyGeometry {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let (model_geometry, required_headroom_bytes) =
        mlx_ram_budget_model_geometry_from_validated_artifact(&validated_artifact, model_directory)
            .expect("the configured sparse artifact should expose disk RAM geometry");
    let complete_residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
        model_geometry,
        required_headroom_bytes,
    );
    SparseArtifactResidencyGeometry {
        complete_residency_with_headroom_bytes: complete_residency_boundary
            .static_complete_residency_bytes
            .saturating_add(complete_residency_boundary.required_headroom_bytes),
    }
}

fn context_memory_reservation_bytes(model_directory: &Path, context_token_count: usize) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let reservation_bytes = validated_artifact
        .config()
        .context_memory_reservation_bytes(context_token_count)
        .expect("the acceptance prompt context should fit the model geometry");
    u64::try_from(reservation_bytes)
        .expect("the acceptance prompt context reservation should fit u64")
}

fn write_acceptance_config(
    worker_home: &Path,
    model_directory: &Path,
    model_id: &str,
    mlx_memory_ceiling_bytes: u64,
    persistent_prompt_cache_enabled: bool,
) {
    let configuration_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&configuration_directory)
        .expect("the isolated acceptance configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": [acceptance_model_scan_root(model_directory)],
            "maximum_mlx_memory_gb": mlx_memory_ceiling_bytes.div_ceil(1_000_000_000),
        },
        "prompt_cache": {
            "enabled": persistent_prompt_cache_enabled,
        },
        "diagnostics": {"performance_attribution_enabled": true},
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": FIXED_PREFILL_CHUNK_SIZE_TOKENS
        },
        "models": {
            (model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
                },
            },
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the acceptance configuration should serialize"),
    )
    .expect("the acceptance configuration should be written");
}

async fn complete_chat_request(
    server_address: std::net::SocketAddr,
    model_id: &str,
    messages: Value,
    request_label: &str,
) -> String {
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");
    let completion_request = json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 1,
        "thinking_budget": 0,
        "max_tokens": 1
    });
    let mut streamed_completion = match openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
    {
        Ok(streamed_completion) => streamed_completion,
        Err(request_error) => {
            let status_document = get_json_endpoint(server_address, "/v1/status").await;
            panic!(
                "{request_label} should start: {request_error}; worker_status={status_document}"
            );
        }
    };
    let mut assistant_response = String::new();
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.unwrap_or_else(|stream_error| {
            panic!("{request_label} should stream a valid response: {stream_error}")
        });
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(text_fragment) = choice["delta"]["content"].as_str() {
                assistant_response.push_str(text_fragment);
            }
        }
    }
    assert!(
        !assistant_response.trim().is_empty(),
        "{request_label} should return visible model output"
    );
    assistant_response
}

fn read_generation_attribution_reports(worker_home: &Path) -> Vec<Value> {
    let attribution_log_path =
        worker_home.join(".astronomical-dev/logs/performance-attribution.jsonl");
    fs::read_to_string(&attribution_log_path)
        .unwrap_or_else(|read_error| {
            panic!(
                "{} should be readable after the worker stops: {read_error}",
                attribution_log_path.display()
            )
        })
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("each acceptance attribution row should be valid JSON")
        })
        .filter(|report| report["report_kind"] == "generation")
        .collect()
}

fn attribution_counter(report: &Value, counter_identifier: &str) -> u64 {
    report["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|counter| counter["counter"] == counter_identifier)
        .and_then(|counter| counter["amount"].as_u64())
        .unwrap_or(0)
}
