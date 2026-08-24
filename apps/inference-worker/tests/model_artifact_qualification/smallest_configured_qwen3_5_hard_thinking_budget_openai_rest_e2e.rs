//! Public REST qualification of Qwen's model-owned hard reasoning transition.

use std::{fs, path::Path};

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use serde_json::{Value, json};

use super::{
    deployment_litmus_model::configured_deployment_litmus_model,
    model_artifact_rest_qualification::{
        E2E_TIMEOUT, launch_model_artifact_rest_server_for_model, post_chat_completion,
        stop_model_artifact_rest_server,
    },
};

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const THINKING_BUDGET_TOKEN_COUNT: u16 = 1;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 128;
const MODEL_OWNED_TRANSITION_TEXT: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model"]
async fn should_use_the_smallest_configured_qwen3_5_model_to_commit_the_complete_hard_thinking_budget_transition_before_streaming_visible_answer_content_through_the_openai_chat_completions_rest_api()
 {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let selected_model = configured_deployment_litmus_model();
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(
                &selected_model.model_directory,
                u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT),
            )
            .expect("the smallest configured Qwen3.5 artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
            .expect("the smallest configured Qwen3.5 tokenizer should load");
        let expected_forced_transition_token_count = u64::try_from(
            tokenizer
                .forced_thinking_transition_token_ids()
                .len(),
        )
        .expect("the forced-transition token count should fit in u64");
        let isolated_worker_home = tempfile::tempdir()
            .expect("the hard-budget REST journey should create an isolated worker home");
        write_hard_thinking_budget_qualification_config(
            isolated_worker_home.path(),
            &selected_model.model_id,
            &selected_model.model_directory,
        );
        let performance_log_directory = tempfile::tempdir()
            .expect("the hard-budget REST journey should create a performance-log directory");
        let rest_server = launch_model_artifact_rest_server_for_model(
            &selected_model.model_id,
            selected_model.model_directory,
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
        )
        .await;

        eprintln!(
            "[hard-thinking-budget-rest] status=progress phase=request model={} budget_tokens={THINKING_BUDGET_TOKEN_COUNT}",
            selected_model.model_id
        );
        let chat_response = post_chat_completion(
            rest_server.server_address,
            hard_thinking_budget_request_body(&selected_model.model_id),
        )
        .await;
        let streamed_completion = parse_streamed_completion(&chat_response);
        stop_model_artifact_rest_server(rest_server).await;

        assert!(
            streamed_completion
                .reasoning_content
                .contains(MODEL_OWNED_TRANSITION_TEXT),
            "the public reasoning stream must contain the complete model-owned transition: {:?}",
            streamed_completion.reasoning_content
        );
        assert!(
            !streamed_completion.visible_content.trim().is_empty(),
            "visible answer content must follow the committed reasoning transition"
        );
        assert!(
            !streamed_completion.reasoning_arrived_after_visible_content,
            "reasoning content must not resume after visible answer streaming begins"
        );
        assert_forced_transition_attribution(
            isolated_worker_home.path(),
            expected_forced_transition_token_count,
        );
        eprintln!(
            "[hard-thinking-budget-rest] status=success forced_transition_tokens={expected_forced_transition_token_count} visible_characters={}",
            streamed_completion.visible_content.len()
        );
    })
    .await
    .expect("the hard thinking-budget REST journey must finish within 115 seconds");
}

struct StreamedCompletion {
    reasoning_content: String,
    visible_content: String,
    reasoning_arrived_after_visible_content: bool,
}

fn parse_streamed_completion(http_response: &str) -> StreamedCompletion {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "the hard-budget REST request should succeed: {http_response}"
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
            .expect("each hard-budget server-sent event should contain valid JSON");
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
        "the hard-budget REST stream should complete cleanly"
    );
    StreamedCompletion {
        reasoning_content,
        visible_content,
        reasoning_arrived_after_visible_content,
    }
}

fn assert_forced_transition_attribution(
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

fn hard_thinking_budget_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": format!(
                "Summarize this Romeo and Juliet excerpt in one sentence.\n\n{}",
                ROMEO_AND_JULIET_SOURCE.chars().take(512).collect::<String>()
            ),
        }],
        "stream": true,
        "temperature": 0,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn write_hard_thinking_budget_qualification_config(
    isolated_worker_home: &Path,
    model_id: &str,
    model_directory: &Path,
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
                    "maximum_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
                },
            },
        },
        "diagnostics": { "performance_attribution_enabled": true },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the hard-budget qualification configuration should serialize"),
    )
    .expect("the hard-budget qualification configuration should be written");
}
