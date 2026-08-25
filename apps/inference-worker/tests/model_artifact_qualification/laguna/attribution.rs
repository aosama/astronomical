//! Public long-prompt residency and switchable attribution qualification.

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::json;
use tokio::time::timeout;

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, full_romeo_and_juliet_source, resolve_reference_model_directory,
};
use crate::model_artifact_qualification::model_artifact_rest_qualification::{
    assert_successful_streaming_chat_response, get_endpoint,
    launch_model_artifact_rest_server_for_model, post_chat_completion,
    stop_model_artifact_rest_server,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "proves long-prompt serving and attribution enable/disable behavior"]
async fn should_serve_a_laguna_xs_long_prompt_with_switchable_attribution() {
    timeout(JOURNEY_TIMEOUT, run_attribution_journey())
        .await
        .expect("the Laguna attribution journey must finish within 115 seconds");
}

async fn run_attribution_journey() {
    run_one_attribution_mode(true).await;
    run_one_attribution_mode(false).await;
}

async fn run_one_attribution_mode(performance_attribution_enabled: bool) {
    let model_directory = resolve_reference_model_directory();
    let isolated_home = tempfile::tempdir().expect("an isolated Development home should exist");
    write_configuration(
        isolated_home.path(),
        &model_directory,
        performance_attribution_enabled,
    );
    let public_model_id = LAGUNA_XS_PUBLIC_MODEL_ID;
    let rest_server = launch_model_artifact_rest_server_for_model(
        public_model_id,
        model_directory,
        Some(isolated_home.path()),
        None,
    )
    .await;
    let request_body = json!({
        "model": public_model_id,
        "messages": [{
            "role": "user",
            "content": format!(
                "Use the supplied Romeo and Juliet source. Identify the play in one word.\n\n{}",
                full_romeo_and_juliet_source()
            )
        }],
        "stream": true,
        "temperature": 1,
        "max_tokens": 1
    })
    .to_string();
    eprintln!("[laguna-attribution] phase=request enabled={performance_attribution_enabled}");
    let response = post_chat_completion(rest_server.server_address, request_body).await;
    assert_successful_streaming_chat_response(&response);
    let status_response = get_endpoint(rest_server.server_address, "/v1/status").await;
    let status_body = status_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(status_response.as_str());
    let status_document: serde_json::Value = serde_json::from_str(status_body)
        .unwrap_or_else(|_| panic!("GET /v1/status should return JSON: {status_response}"));
    assert_eq!(status_document["status"], "ready");
    assert!(
        status_document["expert_memory_mode"].as_str().is_some(),
        "expert_memory_mode must be present in the status document"
    );
    assert!(
        status_document["mlx_memory_snapshot"]["active_memory_bytes"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    stop_model_artifact_rest_server(rest_server).await;

    let attribution_path = isolated_home
        .path()
        .join(".astronomical-dev/logs/performance-attribution.jsonl");
    if performance_attribution_enabled {
        let attribution_log = fs::read_to_string(attribution_path)
            .expect("enabled attribution should write bounded reports");
        assert!(
            attribution_log
                .lines()
                .any(|line| line.contains(r#""report_kind":"generation""#))
        );
    } else {
        assert!(
            !attribution_path.exists(),
            "disabled attribution must not create its report file"
        );
    }
}

fn write_configuration(home_directory: &Path, model_directory: &Path, attribution_enabled: bool) {
    let state_directory = home_directory.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the isolated state should be created");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec(&json!({
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": {
                "model_directories": [model_directory]
            },
            "prompt_cache": {"enabled": false, "maximum_size_gb": 50},
            "models": {
                (LAGUNA_XS_PUBLIC_MODEL_ID): {
                    "generation_defaults": {"maximum_output_tokens": 8}
                }
            },
            "diagnostics": {"performance_attribution_enabled": attribution_enabled},
            "chunking": {
                "fixed_prompt_processing_chunk_size_tokens": 2048
            }
        }))
        .expect("the isolated configuration should serialize"),
    )
    .expect("the isolated configuration should write");
}
