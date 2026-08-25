//! Public HTTP Chat and Responses acceptance journeys for reference Laguna artifacts.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::json;
use tokio::time::timeout;

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, compact_romeo_and_juliet_source, resolve_reference_model_directory,
};
use crate::model_artifact_qualification::model_artifact_rest_qualification::{
    assert_successful_streaming_chat_response, assert_successful_streaming_responses_response,
    get_endpoint, launch_model_artifact_rest_server_for_model, post_chat_completion,
    post_responses_completion, stop_model_artifact_rest_server,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "serves reference Laguna XS through public Chat and Responses"]
async fn should_stream_romeo_and_juliet_from_laguna_xs_chat_and_responses() {
    timeout(
        JOURNEY_TIMEOUT,
        run_public_generation_journey(
            LAGUNA_XS_PUBLIC_MODEL_ID,
            resolve_reference_model_directory(),
        ),
    )
    .await
    .expect("the Laguna XS public journey must finish within 115 seconds");
}

async fn run_public_generation_journey(model_id: &str, model_directory: PathBuf) {
    let isolated_development_home = crate::common::isolated_development_home_from_user_config();
    eprintln!("[laguna-http] phase=launch model={model_id}");
    let rest_server = launch_model_artifact_rest_server_for_model(
        model_id,
        model_directory,
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    let server_address = rest_server.server_address;
    assert_laguna_is_advertised(server_address, model_id).await;
    let source_excerpt = compact_romeo_and_juliet_source();

    eprintln!("[laguna-http] phase=chat model={model_id}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": format!("Use the supplied Romeo and Juliet source. Name the two households.\n\n{source_excerpt}"),
            }],
            "stream": true,
        "temperature": 1,
            "max_tokens": 4,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);

    eprintln!("[laguna-http] phase=responses model={model_id}");
    let responses_response = post_responses_completion(
        server_address,
        json!({
            "model": model_id,
            "input": format!("Use the supplied Romeo and Juliet source. Name the two households.\n\n{source_excerpt}"),
            "stream": true,
            "max_output_tokens": 4,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_responses_response(&responses_response);
    assert_laguna_is_advertised(server_address, model_id).await;
    stop_model_artifact_rest_server(rest_server).await;
    eprintln!("[laguna-http] phase=done model={model_id}");
}

pub(super) fn opencode_shaped_chat_request_body(
    public_model_id: &str,
    romeo_and_juliet_source: &str,
    maximum_output_tokens: u16,
) -> String {
    json!({
        "model": public_model_id,
        "messages": [{
            "role": "user",
            "content": format!("Use the supplied Romeo and Juliet source. Name the two households.\n\n{romeo_and_juliet_source}"),
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_play",
                "description": "Search the supplied play.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }],
        "tool_choice": "auto",
        "stream": true,
            "temperature": 1,
        "max_tokens": maximum_output_tokens,
    })
    .to_string()
}

pub(super) async fn assert_laguna_is_advertised(server_address: SocketAddr, public_model_id: &str) {
    let models_response = get_endpoint(server_address, "/v1/models").await;
    let response_body = models_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(models_response.as_str());
    let models_document: serde_json::Value = serde_json::from_str(response_body)
        .unwrap_or_else(|_| panic!("GET /v1/models should return JSON: {models_response}"));
    assert!(
        models_document["data"]
            .as_array()
            .is_some_and(|models| models.iter().any(|model| model["id"] == public_model_id)),
        "Laguna {public_model_id} must be advertised: {models_document}"
    );
}
