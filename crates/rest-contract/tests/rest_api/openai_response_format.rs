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

#[test]
fn should_accept_structured_outputs_choice() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "structured_outputs": {"choice": ["Juliet", "Romeo"]}
        }"#,
    )
    .expect("choice extra body should deserialize");
    let request_parts = request
        .into_parts()
        .expect("choice extra body should validate");
    assert!(matches!(
        request_parts.enforced_structured_generation,
        Some(astronomical_rest_contract::EnforcedStructuredGeneration::Choice { .. })
    ));
}

#[test]
fn should_enforce_structured_outputs_regex() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "structured_outputs": {"regex": "[A-Z]+"}
        }"#,
    )
    .expect("regex extra body should deserialize");
    let request_parts = request
        .into_parts()
        .expect("a compilable regex must be enforced, not rejected");
    assert_eq!(
        request_parts.enforced_structured_generation,
        Some(
            astronomical_rest_contract::EnforcedStructuredGeneration::Regex {
                pattern: "[A-Z]+".to_owned(),
            }
        )
    );
}

#[test]
fn should_reject_an_uncompilable_regex_pattern() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "structured_outputs": {"regex": "(["}
        }"#,
    )
    .expect("an uncompilable regex extra body should deserialize");
    let validation_error = request
        .into_parts()
        .expect_err("an uncompilable regex must fail closed");
    assert!(validation_error.to_string().contains("regex"));
}

#[test]
fn should_reject_an_oversized_regex_pattern() {
    let oversized_pattern =
        "a".repeat(astronomical_rest_contract::MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES + 1);
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(&format!(
        r#"{{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}}],
            "structured_outputs": {{"regex": "{}"}}
        }}"#,
        oversized_pattern
    ))
    .expect("an oversized regex extra body should deserialize");
    let validation_error = request
        .into_parts()
        .expect_err("an oversized regex pattern must fail closed");
    assert!(validation_error.to_string().contains("bounded"));
}

#[test]
fn should_reject_guided_grammar_until_enforced() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "guided_grammar": "root ::= \"Juliet\" | \"Romeo\""
        }"#,
    )
    .expect("guided_grammar should deserialize");
    let validation_error = request
        .validate()
        .expect_err("unenforced guided_grammar must fail closed");
    assert!(validation_error.to_string().contains("guided_grammar"));
}

#[test]
fn should_reject_structured_outputs_and_guided_grammar_together() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "structured_outputs": {"choice": ["Juliet"]},
            "guided_grammar": "root ::= \"Juliet\""
        }"#,
    )
    .expect("conflicting extra body should deserialize");
    let validation_error = request
        .validate()
        .expect_err("both extra-body fields must fail closed");
    assert!(validation_error.to_string().contains("only one"));
}

#[test]
fn should_accept_structured_outputs_json_object() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "structured_outputs": {"json": {}}
        }"#,
    )
    .expect("json extra body should deserialize");
    let request_parts = request
        .into_parts()
        .expect("empty json extra body should compile to json_object");
    assert_eq!(
        request_parts.enforced_structured_generation,
        Some(astronomical_rest_contract::EnforcedStructuredGeneration::JsonObject)
    );
}
