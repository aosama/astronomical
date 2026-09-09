//! Shared production-boundary helpers for the thinking-budget REST journeys.
//!
//! The canonical `thinking_budget` journey and the coding-agent field-name
//! journey stream the same public Chat Completions surface with the same
//! fixture and the same isolated worker home, so the server-sent-event
//! parsing, the forced-transition attribution check, and the acceptance
//! configuration live here once instead of being duplicated per journey.

use std::{fs, path::Path};

use serde_json::{Value, json};

/// The Romeo and Juliet fixture is the mandated source input for model tests.
pub(crate) const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

/// One reasoning token is the smallest budget that still demands a forced
/// model-owned transition, which makes the attribution assertion deterministic.
pub(crate) const THINKING_BUDGET_TOKEN_COUNT: u16 = 1;
pub(crate) const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 128;

/// The transition text the model emits when the think channel is committed.
pub(crate) const MODEL_OWNED_TRANSITION_TEXT: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

pub(crate) struct StreamedCompletion {
    pub(crate) reasoning_content: String,
    pub(crate) visible_content: String,
    pub(crate) reasoning_arrived_after_visible_content: bool,
}

pub(crate) fn parse_streamed_completion(http_response: &str) -> StreamedCompletion {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "the thinking-budget REST request should succeed: {http_response}"
    );
    let mut reasoning_content = String::new();
    let mut visible_content = String::new();
    let mut reasoning_arrived_after_visible_content = false;
    let mut stream_completed = false;
    for response_line in http_response.lines() {
        let Some(server_sent_event_payload) = response_line.strip_prefix("data: ") else {
            continue;
        };
        if server_sent_event_payload == "[DONE]" {
            stream_completed = true;
            continue;
        }
        let stream_document = serde_json::from_str::<Value>(server_sent_event_payload)
            .expect("each thinking-budget server-sent event should contain valid JSON");
        let Some(delta_document) = stream_document.pointer("/choices/0/delta") else {
            continue;
        };
        if let Some(reasoning_fragment) = delta_document["reasoning_content"].as_str() {
            reasoning_arrived_after_visible_content |= !visible_content.is_empty();
            reasoning_content.push_str(reasoning_fragment);
        }
        if let Some(visible_fragment) = delta_document["content"].as_str() {
            visible_content.push_str(visible_fragment);
        }
    }
    assert!(
        stream_completed,
        "the thinking-budget REST stream should complete cleanly"
    );
    StreamedCompletion {
        reasoning_content,
        visible_content,
        reasoning_arrived_after_visible_content,
    }
}

/// Proves enforcement through the attribution ledger, not just the sampled
/// stream: a model that closes its think channel naturally would satisfy the
/// behavioral assertions by luck, while a real forced transition always
/// accounts for every committed forced token in the generation report.
pub(crate) fn assert_forced_transition_attribution(
    isolated_worker_home: &Path,
    expected_forced_transition_token_count: u64,
) {
    let attribution_log_path =
        isolated_worker_home.join(".astronomical-dev/logs/performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the enabled worker should write performance-attribution reports");
    let generation_report = attribution_log
        .lines()
        .map(|report_line| {
            serde_json::from_str::<Value>(report_line)
                .expect("each performance-attribution row should contain valid JSON")
        })
        .find(|report_document| report_document["report_kind"] == "generation")
        .expect("the completed REST request should write one generation attribution report");
    let forced_transition_token_count = generation_report["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|counter| counter["counter"] == "forced_thinking_transition_token_count")
        .and_then(|counter| counter["amount"].as_u64());
    assert_eq!(
        forced_transition_token_count,
        Some(expected_forced_transition_token_count),
        "attribution must account for every forced token committed by the model"
    );
}

pub(crate) fn write_thinking_budget_acceptance_config(
    isolated_worker_home: &Path,
    model_id: &str,
    model_directory: &Path,
    maximum_output_tokens: u16,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the isolated Astronomical configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": { "model_directories": [model_directory] },
        "prompt_cache": { "enabled": false, "maximum_size_gb": 50 },
        "models": {
            (model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": maximum_output_tokens,
                },
            },
        },
        "diagnostics": { "performance_attribution_enabled": true },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the thinking-budget acceptance configuration should serialize"),
    )
    .expect("the thinking-budget acceptance configuration should be written");
}
