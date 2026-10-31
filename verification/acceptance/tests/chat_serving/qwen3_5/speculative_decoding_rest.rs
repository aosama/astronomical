//! Speculative-decoding serving journey for the resident Qwen3.5 engine.
//!
//! This journey proves the production wiring end to end: the daemon resolves
//! the chunking configuration's speculative block length, the worker passes
//! it into the resident engine, the engine loads the paired DFlash draft
//! beside its target, an admitted session prefills through the
//! tap-capturing path, and decode runs draft-and-verify rounds — greedily
//! at temperature zero, or with one tempered target sample per position at
//! positive temperature (the exact sampled output law). The proof is
//! the generation attribution report: the speculative counters must show
//! rounds ran and the target confirmed part of what the draft proposed.

use std::fs;
use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{
    get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};
use crate::support::{
    artifact_bytes::acceptance_model_scan_root, configured_installed_model_directory_by_id,
    exact_model_prompt, resident_sparse_moe_model_id,
};

use super::automatic_engine_selection_rest::{
    MAXIMUM_REAL_MODEL_MEMORY_BYTES, ROMEO_AND_JULIET_SOURCE,
    measure_sparse_artifact_residency_geometry, read_generation_attribution_reports,
};

/// The measured sweet spot for this target-and-draft pair (the direct-MLX
/// proof journey holds the reproducible 1.58x speedup at this block length).
const SPECULATIVE_BLOCK_LENGTH_TOKENS: u64 = 4;
const SPECULATIVE_JOURNEY_TIMEOUT: Duration = Duration::from_secs(120);
const SPECULATIVE_PROMPT_TOKEN_COUNT: usize = 256;
/// Enough output tokens that several rounds must run, cheaply.
const SPECULATIVE_OUTPUT_TOKEN_COUNT: u64 = 64;
/// Headroom the resident ceiling buys for the draft's weights (~0.8 GB) and
/// its context key-value cache, beyond the target's own residency.
const DRAFT_RESIDENCY_HEADROOM_BYTES: u64 = 2_000_000_000;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "DFlash speculative decoding is disabled by design (see qwen3_5_resident model_loading: measured 0.50x-1.15x throughput at serving temperature); this journey is the diagnostic instrument for the re-enable decision and cannot pass while the feature is off"]
#[serial]
async fn should_serve_a_chat_through_speculative_decoding_with_the_paired_draft() {
    timeout(
        SPECULATIVE_JOURNEY_TIMEOUT,
        should_serve_through_speculative_decoding(0, "greedy-speculative-decoding"),
    )
    .await
    .expect("the speculative serving journey must finish within 120 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "DFlash speculative decoding is disabled by design (see qwen3_5_resident model_loading: measured 0.50x-1.15x throughput at serving temperature); this journey is the diagnostic instrument for the re-enable decision and cannot pass while the feature is off"]
#[serial]
async fn should_serve_a_sampled_chat_through_speculative_decoding() {
    timeout(
        SPECULATIVE_JOURNEY_TIMEOUT,
        should_serve_through_speculative_decoding(700, "sampled-speculative-decoding"),
    )
    .await
    .expect("the sampled speculative serving journey must finish within 120 seconds");
}

async fn should_serve_through_speculative_decoding(
    temperature_thousandths: u64,
    request_label: &str,
) {
    let model_id = resident_sparse_moe_model_id();
    let model_directory = configured_installed_model_directory_by_id(model_id);
    let residency_geometry = measure_sparse_artifact_residency_geometry(&model_directory);
    let speculative_memory_ceiling_bytes = residency_geometry
        .complete_residency_with_headroom_bytes
        .saturating_add(DRAFT_RESIDENCY_HEADROOM_BYTES);
    assert!(
        speculative_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
        "the speculative resident cell must remain within 36 GB: ceiling={speculative_memory_ceiling_bytes}"
    );

    let worker_home = tempfile::tempdir().expect("the speculative worker home should be created");
    write_speculative_acceptance_config(
        worker_home.path(),
        &model_directory,
        model_id,
        speculative_memory_ceiling_bytes,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id,
        model_directory.clone(),
        worker_home.path(),
        speculative_memory_ceiling_bytes,
    )
    .await;

    let prompt_content = exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Name the two families in one short sentence.",
        SPECULATIVE_PROMPT_TOKEN_COUNT,
    );
    let assistant_response = complete_chat_request(
        real_model_rest_server.server_address,
        model_id,
        json!([{"role": "user", "content": prompt_content}]),
        temperature_thousandths,
        request_label,
    )
    .await;

    let status_document =
        get_json_endpoint(real_model_rest_server.server_address, "/v1/status").await;
    assert_eq!(
        status_document["expert_memory_mode"], "resident",
        "the speculative journey must serve on the resident engine: {status_document}"
    );
    stop_real_model_rest_server(real_model_rest_server).await;

    assert!(
        !assistant_response.trim().is_empty(),
        "the speculative session should return visible model output"
    );
    let generation_reports = read_generation_attribution_reports(worker_home.path());
    let speculative_report = generation_reports
        .iter()
        .find(|report| attribution_counter(report, "speculative_round_count") > 0)
        .unwrap_or_else(|| {
            panic!(
                "at least one generation report must carry speculative round counters; report_count={}; reports={generation_reports:?}",
                generation_reports.len()
            )
        });
    let speculative_round_count =
        attribution_counter(speculative_report, "speculative_round_count");
    let speculative_proposed_token_count =
        attribution_counter(speculative_report, "speculative_proposed_token_count");
    let speculative_accepted_token_count =
        attribution_counter(speculative_report, "speculative_accepted_token_count");
    assert!(
        speculative_round_count >= 2,
        "a {SPECULATIVE_OUTPUT_TOKEN_COUNT}-token completion must run several speculative rounds, ran {speculative_round_count}"
    );
    assert!(
        speculative_proposed_token_count >= speculative_accepted_token_count,
        "accepted proposals ({speculative_accepted_token_count}) cannot exceed proposed tokens ({speculative_proposed_token_count})"
    );
    assert!(
        speculative_accepted_token_count > 0,
        "the draft must confirm at least one proposal across {speculative_round_count} rounds"
    );
}

fn write_speculative_acceptance_config(
    worker_home: &Path,
    model_directory: &Path,
    model_id: &str,
    speculative_memory_ceiling_bytes: u64,
) {
    let configuration_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&configuration_directory)
        .expect("the speculative acceptance configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": [acceptance_model_scan_root(model_directory)],
            "maximum_mlx_memory_gb": speculative_memory_ceiling_bytes.div_ceil(1_000_000_000),
        },
        // The cache stays enabled to match the control journey's working
        // attribution path; admitted speculative sessions bypass it in both
        // directions regardless (decided at session admission).
        "prompt_cache": {"enabled": true},
        "diagnostics": {"performance_attribution_enabled": true},
        "chunking": {
            "speculative_decoding_block_length_tokens": SPECULATIVE_BLOCK_LENGTH_TOKENS,
        },
        "models": {
            (model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": SPECULATIVE_OUTPUT_TOKEN_COUNT,
                },
            },
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the speculative acceptance configuration should serialize"),
    )
    .expect("the speculative acceptance configuration should be written");
}

async fn complete_chat_request(
    server_address: std::net::SocketAddr,
    model_id: &str,
    messages: Value,
    temperature_thousandths: u64,
    request_label: &str,
) -> String {
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");
    let completion_request = json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        // Temperature zero is greedy argmax; a positive temperature runs
        // sampled verification — one tempered target sample per position —
        // which preserves the sampled output law exactly.
        "temperature": temperature_thousandths as f64 / 1_000.0,
        "thinking_budget": 0,
        "max_tokens": SPECULATIVE_OUTPUT_TOKEN_COUNT,
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
    let mut observed_finish_reasons = Vec::new();
    let mut observed_stream_error = Option::<String>::None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = match stream_item {
            Ok(stream_chunk) => stream_chunk,
            Err(stream_error) => {
                observed_stream_error = Some(stream_error.to_string());
                break;
            }
        };
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(text_fragment) = choice["delta"]["content"].as_str() {
                assistant_response.push_str(text_fragment);
            }
            if let Some(finish_reason) = choice["finish_reason"].as_str() {
                observed_finish_reasons.push(finish_reason.to_owned());
            }
        }
        if stream_chunk.get("error").is_some() {
            observed_stream_error = Some(stream_chunk["error"].to_string());
        }
    }
    assert!(
        observed_stream_error.is_none(),
        "{request_label} stream must not end in an error: {observed_stream_error:?}"
    );
    assert_eq!(
        observed_finish_reasons,
        vec!["stop".to_owned()],
        "{request_label} must finish with a natural stop, saw {observed_finish_reasons:?}"
    );
    assistant_response
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
