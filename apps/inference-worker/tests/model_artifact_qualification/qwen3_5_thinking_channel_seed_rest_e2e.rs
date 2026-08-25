//! Qualifies config-gated Qwen reasoning seeding across both public streaming REST APIs.

use std::{fs, path::Path};

use serde_json::{Value, json};

use super::{
    deployment_litmus_model::configured_deployment_litmus_model,
    model_artifact_rest_qualification::{
        E2E_TIMEOUT, assert_successful_streaming_chat_response,
        assert_successful_streaming_responses_response,
        launch_model_artifact_rest_server_for_model, post_chat_completion,
        post_responses_completion, stop_model_artifact_rest_server,
    },
};

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const THINKING_CHANNEL_SEED: &str = "Two households, both alike in dignity, in Romeo and Juliet.";
const MAXIMUM_OUTPUT_TOKEN_COUNT: u16 = 64;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model"]
async fn should_seed_the_first_reasoning_output_across_both_streaming_rest_apis_with_a_real_qwen3_5_model()
 {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let selected_model = configured_deployment_litmus_model();
        let isolated_worker_home = tempfile::tempdir()
            .expect("the thinking-seed REST journey should create an isolated worker home");
        write_thinking_seed_qualification_state(
            isolated_worker_home.path(),
            &selected_model.model_id,
            &selected_model.model_directory,
        );
        let performance_log_directory = tempfile::tempdir()
            .expect("the thinking-seed REST journey should create a performance-log directory");
        let rest_server = launch_model_artifact_rest_server_for_model(
            &selected_model.model_id,
            selected_model.model_directory,
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
        )
        .await;

        eprintln!(
            "[thinking-seed-rest 1/2] status=progress api=chat_completions model={}",
            selected_model.model_id
        );
        let chat_response = post_chat_completion(
            rest_server.server_address,
            chat_request_body(&selected_model.model_id),
        )
        .await;
        assert_successful_streaming_chat_response(&chat_response);
        let chat_reasoning = chat_reasoning_content(&chat_response);
        assert!(
            chat_reasoning.starts_with(THINKING_CHANNEL_SEED),
            "Chat Completions must emit the configured seed first: {chat_reasoning:?}"
        );
        eprintln!(
            "[thinking-seed-rest 1/2] status=success api=chat_completions reasoning_characters={}",
            chat_reasoning.len()
        );

        eprintln!(
            "[thinking-seed-rest 2/2] status=progress api=responses model={}",
            selected_model.model_id
        );
        let responses_response = post_responses_completion(
            rest_server.server_address,
            responses_request_body(&selected_model.model_id),
        )
        .await;
        assert_successful_streaming_responses_response(&responses_response);
        let responses_reasoning = responses_reasoning_content(&responses_response);
        assert!(
            responses_reasoning.starts_with(THINKING_CHANNEL_SEED),
            "Responses must emit the configured seed first: {responses_reasoning:?}"
        );
        eprintln!(
            "[thinking-seed-rest 2/2] status=success api=responses reasoning_characters={}",
            responses_reasoning.len()
        );

        stop_model_artifact_rest_server(rest_server).await;
    })
    .await
    .expect("the thinking-seed REST journey must finish within 115 seconds");
}

fn chat_reasoning_content(http_response: &str) -> String {
    http_response
        .lines()
        .filter_map(|response_line| response_line.strip_prefix("data: "))
        .filter(|event_payload| *event_payload != "[DONE]")
        .map(|event_payload| {
            serde_json::from_str::<Value>(event_payload)
                .expect("each Chat Completions event should contain valid JSON")
        })
        .filter_map(|event_document| {
            event_document["choices"][0]["delta"]["reasoning_content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

fn responses_reasoning_content(http_response: &str) -> String {
    http_response
        .split("\n\n")
        .filter(|event_frame| {
            event_frame
                .lines()
                .any(|line| line == "event: response.reasoning_summary_text.delta")
        })
        .filter_map(|event_frame| {
            event_frame
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
        })
        .map(|event_payload| {
            serde_json::from_str::<Value>(event_payload)
                .expect("each Responses event should contain valid JSON")
        })
        .filter_map(|event_document| event_document["delta"].as_str().map(str::to_owned))
        .collect()
}

fn chat_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": qualification_prompt(),
        }],
        "stream": true,
        "temperature": 1,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn responses_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "input": qualification_prompt(),
        "stream": true,
        "temperature": 1,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn qualification_prompt() -> String {
    format!(
        "Summarize this Romeo and Juliet excerpt in one sentence.\n\n{}",
        ROMEO_AND_JULIET_SOURCE
            .chars()
            .take(512)
            .collect::<String>()
    )
}

fn write_thinking_seed_qualification_state(
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
        "runtime": {
            "model_directories": [model_directory],
            "experimental_qwen_thinking_channel_seed_enabled": true,
        },
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
            .expect("the thinking-seed qualification configuration should serialize"),
    )
    .expect("the thinking-seed qualification configuration should be written");
    fs::write(
        configuration_directory.join("thinking.md"),
        THINKING_CHANNEL_SEED,
    )
    .expect("the thinking seed should be written");
}
