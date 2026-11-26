//! The "continue the Romeo and Juliet story" request the dense and
//! prefill-recovery journeys send, with the fresh-install configuration they
//! run under and the check that a stream was served, so each journey states
//! only the limit and the prompt sizes it is about.

use std::{fs, path::Path};

use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::support::openai_client::ChatCompletionStream;

pub(super) const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

/// The settings a fresh install runs with; only the model folders are named
/// and the log level is raised so the evidence folder records the run.
pub(super) fn write_everyday_configuration(worker_home: &Path) {
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {"model_directories": crate::support::configured_model_directory_roots()},
        "diagnostics": {"log_level": "info"},
    });
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}

pub(super) fn story_prompt(model_directory: &Path, prompt_token_count: usize) -> String {
    crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        prompt_token_count,
    )
}

/// Temperature 0 keeps the reply deterministic, so a failure to serve the
/// request cannot be a sampling accident.
pub(super) fn story_request(
    model_id: &str,
    prompt: &str,
    maximum_output_token_count: u64,
) -> Value {
    json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": maximum_output_token_count,
        "temperature": 0,
    })
}

/// True when the stream finished with a finish reason and carried no error.
pub(super) async fn consume_completion(
    mut stream: ChatCompletionStream,
    journey_log_label: &str,
) -> bool {
    let mut finished = false;
    while let Some(fragment) = stream.next().await {
        let fragment = fragment.expect("the completion stream must stay readable");
        if !fragment["error"].is_null() {
            eprintln!(
                "[{journey_log_label}] request failed: {}",
                fragment["error"]
            );
            return false;
        }
        for choice in fragment["choices"].as_array().into_iter().flatten() {
            finished |= matches!(choice["finish_reason"].as_str(), Some("stop" | "length"));
        }
    }
    finished
}
