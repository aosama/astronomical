//! The streamed chat request with a thinking budget that the large-model
//! memory journeys send, and the way they read it back: the full reply text
//! and its finish reason, plus the configuration file each journey's isolated
//! worker home starts from.

use std::{fs, path::Path};

use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::support::openai_client::ChatCompletionStream;

pub(super) struct CompletedStream {
    pub(super) model_text: String,
    pub(super) finish_reason: Option<String>,
}

impl CompletedStream {
    /// A journey that streamed nothing, or stopped for any reason but a
    /// normal end or the token limit, proved nothing about memory.
    pub(super) fn assert_produced_output_and_finished(&self) {
        assert!(
            !self.model_text.is_empty(),
            "the model must produce output text"
        );
        assert!(matches!(
            self.finish_reason.as_deref(),
            Some("stop" | "length")
        ));
    }
}

pub(super) fn thinking_budget_completion_request(
    model_id: &str,
    user_message: &str,
    maximum_output_token_count: u32,
    thinking_budget_token_count: u32,
) -> Value {
    json!({
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": maximum_output_token_count,
        "thinking_budget": thinking_budget_token_count,
    })
}

pub(super) async fn consume_completed_stream(
    mut streamed_completion: ChatCompletionStream,
) -> CompletedStream {
    let mut streamed_model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.expect("the public REST stream should remain healthy");
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                streamed_model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    CompletedStream {
        model_text: streamed_model_text.trim().to_owned(),
        finish_reason,
    }
}

/// Writes `config.json` into the worker home's Development state folder;
/// `journey_label` names the journey in the failure messages.
pub(super) fn write_worker_configuration(
    isolated_worker_home: &Path,
    journey_label: &str,
    configuration_document: &Value,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory).unwrap_or_else(|error| {
        panic!("the {journey_label} configuration directory should be created: {error}")
    });
    let serialized_configuration = serde_json::to_vec_pretty(configuration_document)
        .unwrap_or_else(|error| {
            panic!("the {journey_label} configuration should serialize: {error}")
        });
    fs::write(
        configuration_directory.join("config.json"),
        serialized_configuration,
    )
    .unwrap_or_else(|error| panic!("the {journey_label} configuration should be written: {error}"));
}
