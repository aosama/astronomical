use astronomical_rest_contract::{
    OpenAiChatCompletionRequest, OpenAiResponsesRequest, OpenAiStructuredOutput,
    OpenAiStructuredOutputValidationError, UNENFORCED_RESPONSE_FORMAT_WARNING,
    compact_extracted_json_text, extract_json_value_from_text,
};
use serde_json::json;

#[test]
fn should_accept_json_object_response_format_on_chat_completions() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "response_format": {"type": "json_object"}
        }"#,
    )
    .expect("json_object response_format should deserialize");

    let request_parts = request
        .into_parts()
        .expect("json_object response_format should validate");

    assert_eq!(
        request_parts.structured_output,
        Some(OpenAiStructuredOutput::JsonObject)
    );
}

#[test]
fn should_accept_json_schema_response_format_on_chat_completions() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "romeo_line",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "play": {"type": "string"}
                        },
                        "required": ["speaker", "play"]
                    },
                    "strict": true
                }
            }
        }"#,
    )
    .expect("json_schema response_format should deserialize");

    let request_parts = request
        .into_parts()
        .expect("json_schema response_format should validate");

    match request_parts.structured_output {
        Some(OpenAiStructuredOutput::JsonSchema { name, strict, .. }) => {
            assert_eq!(name, "romeo_line");
            assert!(strict);
        }
        other_structured_output => panic!("expected json_schema, got {other_structured_output:?}"),
    }
}

#[test]
fn should_reject_an_unsupported_response_format_type() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "response_format": {"type": "xml"}
        }"#,
    )
    .expect("unsupported response_format should deserialize");

    let validation_error = request
        .into_parts()
        .expect_err("unsupported response_format types must fail before worker admission");

    assert_eq!(
        validation_error.to_string(),
        OpenAiStructuredOutputValidationError::UnsupportedType {
            format_type: "xml".to_owned(),
        }
        .to_string()
    );
}

#[test]
fn should_accept_responses_text_format_json_schema() {
    let request = serde_json::from_str::<OpenAiResponsesRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "input": "O Romeo, Romeo, wherefore art thou Romeo?",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "romeo_line",
                    "schema": {"type": "object"}
                }
            }
        }"#,
    )
    .expect("Responses text.format json_schema should deserialize");

    let request_parts = request
        .into_parts()
        .expect("Responses text.format json_schema should validate");

    assert!(matches!(
        request_parts.structured_output,
        Some(OpenAiStructuredOutput::JsonSchema { .. })
    ));
}

#[test]
fn should_extract_json_from_fenced_model_text_without_filling_fields() {
    let extracted_json = extract_json_value_from_text(
        "Juliet says:\n```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```\n",
    )
    .expect("fenced JSON should extract");

    assert_eq!(
        extracted_json,
        json!({"speaker": "Juliet", "play": "Romeo and Juliet"})
    );
    assert_eq!(compact_extracted_json_text("not json at all"), None);
}

#[test]
fn should_name_unenforced_grammar_in_the_warning_header() {
    assert!(
        UNENFORCED_RESPONSE_FORMAT_WARNING.contains("grammar-constrained decoding unavailable")
    );
}
