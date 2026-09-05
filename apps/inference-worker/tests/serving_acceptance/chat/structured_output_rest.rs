//! GPU acceptance for OpenAI structured output on a live chat model.

use astronomical_config::DiscoveredModel;
use astronomical_rest_contract::{
    UNENFORCED_RESPONSE_FORMAT_WARNING, extract_json_value_from_text,
};
use serde_json::{Value, json};

use super::openai_rest::{
    E2E_TIMEOUT, get_endpoint, launch_serving_rest_server_for_model, post_chat_completion,
    post_responses_completion, stop_serving_rest_server,
};

const STRUCTURED_OUTPUT_MODEL_LEAF_ID: &str = "Qwen3.5-2B-4bit";
const ROMEO_AND_JULIET_LINE: &str = "O Romeo, Romeo, wherefore art thou Romeo?";
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 384;
// Thinking-enabled Qwen otherwise fills max_tokens with reasoning and returns empty JSON.
const THINKING_TOKEN_BUDGET: u32 = 64;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads Qwen3.5-2B-4bit on GPU and exercises structured output over public REST"]
async fn should_serve_structured_json_from_romeo_and_juliet_on_chat_and_responses() {
    tokio::time::timeout(E2E_TIMEOUT, run_structured_output_gpu_journey())
        .await
        .expect("the structured-output GPU journey must finish within 115 seconds");
}

async fn run_structured_output_gpu_journey() {
    let selected_model = configured_structured_output_chat_model();
    eprintln!(
        "[structured-output-rest] status=progress phase=launch model={}",
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

    eprintln!("[structured-output-rest 1/5] status=progress phase=models_capability");
    let models_response = get_endpoint(server_address, "/v1/models").await;
    assert_http_ok(&models_response);
    let models_document = http_json_body(&models_response);
    let advertised_model = advertised_model_document(&models_document, &selected_model.model_id);
    assert_eq!(advertised_model["supports_structured_outputs"], true);
    assert_eq!(advertised_model["structured_output_enforcement"], "none");
    eprintln!("[structured-output-rest 1/5] status=success phase=models_capability");

    eprintln!("[structured-output-rest 2/5] status=progress phase=chat_json_schema");
    let chat_schema_response = post_chat_completion(
        server_address,
        chat_json_schema_request_body(&selected_model.model_id, false),
    )
    .await;
    assert_structured_json_http_response(&chat_schema_response, chat_visible_content);
    eprintln!("[structured-output-rest 2/5] status=success phase=chat_json_schema");

    eprintln!("[structured-output-rest 3/5] status=progress phase=chat_json_object");
    let chat_object_response = post_chat_completion(
        server_address,
        chat_json_object_request_body(&selected_model.model_id),
    )
    .await;
    assert_structured_json_http_response(&chat_object_response, chat_visible_content);
    eprintln!("[structured-output-rest 3/5] status=success phase=chat_json_object");

    eprintln!("[structured-output-rest 4/5] status=progress phase=responses_json_schema");
    let responses_schema_response = post_responses_completion(
        server_address,
        responses_json_schema_request_body(&selected_model.model_id),
    )
    .await;
    assert_structured_json_http_response(&responses_schema_response, responses_visible_content);
    eprintln!("[structured-output-rest 4/5] status=success phase=responses_json_schema");

    eprintln!("[structured-output-rest 5/5] status=progress phase=chat_json_schema_stream");
    let chat_stream_response = post_chat_completion(
        server_address,
        chat_json_schema_request_body(&selected_model.model_id, true),
    )
    .await;
    assert_http_ok(&chat_stream_response);
    assert_unenforced_warning(&chat_stream_response);
    assert!(
        chat_stream_response.contains("data: [DONE]"),
        "streaming structured chat must finish cleanly: {chat_stream_response}"
    );
    eprintln!("[structured-output-rest 5/5] status=success phase=chat_json_schema_stream");

    stop_serving_rest_server(rest_server).await;
    eprintln!(
        "[structured-output-rest] status=success model={}",
        selected_model.model_id
    );
}

fn configured_structured_output_chat_model() -> DiscoveredModel {
    let discovered_models = crate::support::configured_discovered_models();
    let selected_model = discovered_models
        .into_iter()
        .find(|discovered_model| {
            discovered_model.model_id == STRUCTURED_OUTPUT_MODEL_LEAF_ID
                || discovered_model.model_id == "mlx-community/Qwen3.5-2B-4bit"
                || discovered_model
                    .model_id
                    .rsplit('/')
                    .next()
                    == Some(STRUCTURED_OUTPUT_MODEL_LEAF_ID)
        })
        .unwrap_or_else(|| {
            panic!(
                "Development discovery must include {STRUCTURED_OUTPUT_MODEL_LEAF_ID} for structured-output acceptance"
            )
        });
    let chat_capabilities =
        crate::support::chat_capabilities(&selected_model).unwrap_or_else(|| {
            panic!(
                "{STRUCTURED_OUTPUT_MODEL_LEAF_ID} must be a chat model for structured-output acceptance"
            )
        });
    astronomical_model_serving::Qwen3_5ArtifactValidator::new()
        .validate(
            &selected_model.model_directory,
            chat_capabilities.max_output_tokens,
        )
        .unwrap_or_else(|artifact_validation_error| {
            panic!("{STRUCTURED_OUTPUT_MODEL_LEAF_ID} must validate: {artifact_validation_error}")
        });
    selected_model
}

fn chat_json_schema_request_body(model_id: &str, stream: bool) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": format!(
                "Classify this Romeo and Juliet line as JSON. Line: {ROMEO_AND_JULIET_LINE}"
            ),
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "romeo_line",
                "schema": romeo_line_schema(),
            },
        },
        "stream": stream,
        "temperature": 1,
        "thinking_budget": THINKING_TOKEN_BUDGET,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn chat_json_object_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": format!(
                "Return JSON with keys speaker and play for this Romeo and Juliet line: {ROMEO_AND_JULIET_LINE}"
            ),
        }],
        "response_format": {"type": "json_object"},
        "stream": false,
        "temperature": 1,
        "thinking_budget": THINKING_TOKEN_BUDGET,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn responses_json_schema_request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "input": format!(
            "Classify this Romeo and Juliet line as JSON. Line: {ROMEO_AND_JULIET_LINE}"
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "romeo_line",
                "schema": romeo_line_schema(),
            },
        },
        "stream": false,
        "temperature": 1,
        "thinking_budget": THINKING_TOKEN_BUDGET,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn romeo_line_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "play": {"type": "string"},
        },
        "required": ["speaker", "play"],
    })
}

fn assert_structured_json_http_response(http_response: &str, visible_content: fn(&Value) -> &str) {
    assert_http_ok(http_response);
    assert_unenforced_warning(http_response);
    let response_document = http_json_body(http_response);
    let visible_text = visible_content(&response_document);
    let extracted_json = extract_json_value_from_text(visible_text).unwrap_or_else(|| {
        panic!(
            "structured output must be parseable JSON, visible={visible_text:?} body={response_document}"
        )
    });
    assert!(
        extracted_json.is_object() || extracted_json.is_array(),
        "structured output must be a JSON object or array, got: {extracted_json}"
    );
}

fn chat_visible_content(response_document: &Value) -> &str {
    response_document["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
}

fn responses_visible_content(response_document: &Value) -> &str {
    if let Some(output_text) = response_document["output_text"].as_str()
        && !output_text.is_empty()
    {
        return output_text;
    }
    ""
}

fn advertised_model_document<'response_document>(
    models_document: &'response_document Value,
    model_id: &str,
) -> &'response_document Value {
    models_document["data"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|advertised_model| advertised_model["id"].as_str() == Some(model_id))
        .unwrap_or_else(|| panic!("GET /v1/models must advertise {model_id}"))
}

fn assert_http_ok(http_response: &str) {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected HTTP response: {http_response}"
    );
}

fn assert_unenforced_warning(http_response: &str) {
    let warning_header = http_header_value(http_response, "warning")
        .expect("structured output without grammar must send Warning");
    assert_eq!(warning_header, UNENFORCED_RESPONSE_FORMAT_WARNING);
}

fn http_header_value(http_response: &str, header_name: &str) -> Option<String> {
    let header_section = http_response.split("\r\n\r\n").next()?;
    header_section.lines().find_map(|header_line| {
        let (header_label, header_value) = header_line.split_once(':')?;
        header_label
            .eq_ignore_ascii_case(header_name)
            .then(|| header_value.trim().to_owned())
    })
}

fn http_json_body(http_response: &str) -> Value {
    let response_body = http_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or("")
        .trim_start_matches('\u{feff}');
    serde_json::from_str(response_body).unwrap_or_else(|json_error| {
        panic!("HTTP body should be JSON ({json_error}): {http_response}")
    })
}
