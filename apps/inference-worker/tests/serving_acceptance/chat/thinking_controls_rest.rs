//! GPU acceptance for the full thinking-control spelling family over public REST.
//!
//! Proves the issue #769 outcome end to end: `reasoning.max_tokens` enforces the
//! same hard thinking budget as the top-level spellings, `exclude` withholds the
//! client-visible reasoning while usage still reports the count, and an explicit
//! disable closes the thinking channel on the visible answer.

use serde_json::json;

use super::openai_rest::{
    E2E_TIMEOUT, launch_serving_rest_server_for_model, post_chat_completion,
    post_responses_completion, stop_serving_rest_server,
};

const THINKING_CONTROLS_MODEL_LEAF_ID: &str = "Qwen3.5-2B-4bit";
const ROMEO_AND_JULIET_LINE: &str = "O Romeo, Romeo, wherefore art thou Romeo?";
const REASONING_MAX_TOKENS: u32 = 128;
// The model-owned forced transition and its boundary add a few reasoning tokens
// after the budget counter reaches its cap; the ceiling must cover that tail.
const MAXIMUM_REASONING_TOKENS_WITH_TRANSITION: u32 = REASONING_MAX_TOKENS + 32;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 384;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Qwen3.5-2B-4bit on GPU and exercises thinking-control spellings over public REST"]
async fn should_enforce_thinking_control_spellings_over_public_rest() {
    tokio::time::timeout(E2E_TIMEOUT, run_thinking_controls_gpu_journey())
        .await
        .expect("the thinking-controls GPU journey must finish within 115 seconds");
}

async fn run_thinking_controls_gpu_journey() {
    let selected_model = configured_thinking_controls_chat_model();
    eprintln!(
        "[thinking-controls-rest] status=progress phase=launch model={}",
        selected_model.model_id
    );
    let rest_server = launch_serving_rest_server_for_model(
        &selected_model.model_id,
        selected_model.model_directory.clone(),
        None,
        None,
    )
    .await;
    let server_address = rest_server.server_address;

    eprintln!("[thinking-controls-rest 1/3] status=progress phase=responses_max_tokens");
    let responses_response = post_responses_completion(
        server_address,
        responses_request_body(&selected_model.model_id, false),
    )
    .await;
    assert_http_ok(&responses_response);
    let responses_document = http_json_body(&responses_response);
    let output_item_types = responses_document["output"]
        .as_array()
        .expect("the Responses output must be an array")
        .iter()
        .map(|output_item| {
            output_item["type"]
                .as_str()
                .expect("every output item carries a type")
        })
        .collect::<Vec<_>>();
    assert!(
        output_item_types.contains(&"reasoning"),
        "an unexcluded reasoning budget must expose a reasoning output item: {responses_document}"
    );
    assert!(output_item_types.contains(&"message"));
    let reasoning_token_count =
        responses_document["usage"]["output_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .expect("usage must report the reasoning token count");
    assert!(
        reasoning_token_count <= u64::from(MAXIMUM_REASONING_TOKENS_WITH_TRANSITION),
        "reasoning.max_tokens {REASONING_MAX_TOKENS} must cap the thinking at \
         {MAXIMUM_REASONING_TOKENS_WITH_TRANSITION} reasoning tokens, got {reasoning_token_count}"
    );
    assert!(
        reasoning_token_count >= u64::from(REASONING_MAX_TOKENS),
        "the budget must have been active: the model should exhaust the \n         {REASONING_MAX_TOKENS}-token allowance before the forced transition, got {reasoning_token_count}"
    );
    eprintln!(
        "[thinking-controls-rest 1/3] status=success phase=responses_max_tokens reasoning_tokens={reasoning_token_count}"
    );

    eprintln!("[thinking-controls-rest 2/3] status=progress phase=responses_excluded");
    let excluded_responses_response = post_responses_completion(
        server_address,
        responses_request_body(&selected_model.model_id, true),
    )
    .await;
    assert_http_ok(&excluded_responses_response);
    let excluded_responses_document = http_json_body(&excluded_responses_response);
    let excluded_output_item_types = excluded_responses_document["output"]
        .as_array()
        .expect("the Responses output must be an array")
        .iter()
        .map(|output_item| {
            output_item["type"]
                .as_str()
                .expect("every output item carries a type")
        })
        .collect::<Vec<_>>();
    assert!(
        !excluded_output_item_types.contains(&"reasoning"),
        "exclude must withhold the reasoning output item: {excluded_responses_document}"
    );
    assert!(excluded_output_item_types.contains(&"message"));
    let excluded_reasoning_token_count =
        excluded_responses_document["usage"]["output_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .expect("usage must report the reasoning token count even when excluded");
    assert!(
        excluded_reasoning_token_count <= u64::from(MAXIMUM_REASONING_TOKENS_WITH_TRANSITION),
        "the excluded reasoning must still respect its budget, got {excluded_reasoning_token_count}"
    );
    assert!(
        excluded_reasoning_token_count >= u64::from(REASONING_MAX_TOKENS),
        "exclusion must not change how much the model thinks: the \n         {REASONING_MAX_TOKENS}-token allowance should still be exhausted, got {excluded_reasoning_token_count}"
    );
    eprintln!(
        "[thinking-controls-rest 2/3] status=success phase=responses_excluded reasoning_tokens={excluded_reasoning_token_count}"
    );

    eprintln!("[thinking-controls-rest 3/3] status=progress phase=chat_disabled");
    let chat_response = post_chat_completion(
        server_address,
        chat_disabled_request_body(&selected_model.model_id),
    )
    .await;
    assert_http_ok(&chat_response);
    let chat_document = http_json_body(&chat_response);
    assert!(
        chat_document["choices"][0]["message"]["reasoning_content"].is_null(),
        "reasoning_effort none must close the thinking channel: {chat_document}"
    );
    let visible_text = chat_document["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .trim();
    assert!(
        !visible_text.is_empty(),
        "a disabled-thinking request must still produce a visible answer: {chat_document}"
    );
    eprintln!("[thinking-controls-rest 3/3] status=success phase=chat_disabled");

    stop_serving_rest_server(rest_server).await;
    eprintln!(
        "[thinking-controls-rest] status=success model={}",
        selected_model.model_id
    );
}

fn configured_thinking_controls_chat_model() -> astronomical_config::DiscoveredModel {
    let discovered_models = crate::support::configured_discovered_models();
    let selected_model = discovered_models
        .into_iter()
        .find(|discovered_model| {
            discovered_model.model_id == THINKING_CONTROLS_MODEL_LEAF_ID
                || discovered_model.model_id == "mlx-community/Qwen3.5-2B-4bit"
                || discovered_model
                    .model_id
                    .rsplit('/')
                    .next()
                    == Some(THINKING_CONTROLS_MODEL_LEAF_ID)
        })
        .unwrap_or_else(|| {
            panic!(
                "Development discovery must include {THINKING_CONTROLS_MODEL_LEAF_ID} for thinking-controls acceptance"
            )
        });
    let chat_capabilities =
        crate::support::chat_capabilities(&selected_model).unwrap_or_else(|| {
            panic!(
                "{THINKING_CONTROLS_MODEL_LEAF_ID} must be a chat model for thinking-controls acceptance"
            )
        });
    astronomical_model_serving::Qwen3_5ArtifactValidator::new()
        .validate(
            &selected_model.model_directory,
            chat_capabilities.max_output_tokens,
        )
        .unwrap_or_else(|artifact_validation_error| {
            panic!("{THINKING_CONTROLS_MODEL_LEAF_ID} must validate: {artifact_validation_error}")
        });
    selected_model
}

fn responses_request_body(model_id: &str, exclude: bool) -> String {
    let mut reasoning_object = json!({ "max_tokens": REASONING_MAX_TOKENS });
    if exclude {
        reasoning_object["exclude"] = json!(true);
    }
    json!({
        "model": model_id,
        "input": format!("Who speaks this Romeo and Juliet line? {ROMEO_AND_JULIET_LINE} Answer with one name."),
        "stream": false,
        "temperature": 1,
        "reasoning": reasoning_object,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn chat_disabled_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": format!("Who speaks this Romeo and Juliet line? {ROMEO_AND_JULIET_LINE} Answer with one name.")
        }],
        "stream": false,
        "temperature": 1,
        "reasoning_effort": "none",
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn assert_http_ok(http_response: &str) {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected HTTP response: {http_response}"
    );
}

fn http_json_body(http_response: &str) -> serde_json::Value {
    let response_body = http_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or("")
        .trim_start_matches('\u{feff}');
    serde_json::from_str(response_body).unwrap_or_else(|json_error| {
        panic!("HTTP body should be JSON ({json_error}): {http_response}")
    })
}
