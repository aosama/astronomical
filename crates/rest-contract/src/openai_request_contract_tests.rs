//! What the Chat Completions and Responses requests promise alike, held side
//! by side so the two cannot drift: every thinking-control spelling is a known
//! field on both, Responses still refuses a field it does not know while Chat
//! Completions absorbs it, and the bounds both enforce are reported in the
//! same words.

use serde_json::{Value, json};

use crate::{
    MAX_OUTPUT_TOKENS, OpenAiChatCompletionRequest, OpenAiChatCompletionValidationError,
    OpenAiResponsesRequest, OpenAiResponsesValidationError,
};

const MODEL_ID: &str = "astronomical/fake-mixture-of-experts";

/// Every thinking-control field either request reads, with a value it accepts.
fn thinking_control_spellings() -> Vec<(&'static str, Value)> {
    vec![
        ("thinking_budget", json!(64)),
        ("thinking_token_budget", json!(64)),
        ("thinking_budget_tokens", json!(64)),
        ("reasoning", json!({"effort": "low"})),
        ("reasoning_effort", json!("low")),
        ("enable_thinking", json!(true)),
        ("chat_template_kwargs", json!({"enable_thinking": true})),
    ]
}

fn chat_request(extra_fields: Value) -> Result<(), OpenAiChatCompletionValidationError> {
    let mut request_document = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Shall I compare thee to a summer's day?"}]
    });
    merge_fields(&mut request_document, extra_fields);
    let request: OpenAiChatCompletionRequest =
        serde_json::from_value(request_document).expect("the chat request should parse");
    request.into_parts().map(|_parts| ())
}

fn responses_request(extra_fields: Value) -> Result<(), OpenAiResponsesValidationError> {
    let mut request_document = json!({
        "model": MODEL_ID,
        "input": "Shall I compare thee to a summer's day?"
    });
    merge_fields(&mut request_document, extra_fields);
    let request: OpenAiResponsesRequest =
        serde_json::from_value(request_document).expect("the responses request should parse");
    request.into_parts().map(|_parts| ())
}

fn merge_fields(request_document: &mut Value, extra_fields: Value) {
    let Value::Object(extra_members) = extra_fields else {
        panic!("extra fields should be an object");
    };
    for (field_name, field_value) in extra_members {
        request_document[field_name] = field_value;
    }
}

#[test]
fn should_accept_every_thinking_control_spelling_on_both_requests() {
    for (field_name, field_value) in thinking_control_spellings() {
        let extra_fields = json!({ field_name: field_value });
        assert_eq!(
            chat_request(extra_fields.clone()),
            Ok(()),
            "chat: {field_name}"
        );
        assert_eq!(
            responses_request(extra_fields),
            Ok(()),
            "responses: {field_name}"
        );
    }
}

#[test]
fn should_refuse_an_unknown_field_on_responses_and_absorb_it_on_chat() {
    let unknown_field = json!({"thinking_budget": 64, "sonnet_number": 18});
    assert_eq!(chat_request(unknown_field.clone()), Ok(()));
    assert_eq!(
        responses_request(unknown_field),
        Err(OpenAiResponsesValidationError::UnknownField {
            field_name: "sonnet_number".to_owned(),
        })
    );
}

#[test]
fn should_accept_structured_generation_fields_on_both_requests() {
    let structured_fields = json!({
        "response_format": {"type": "json_object"},
        "guided_grammar": null
    });
    assert_eq!(chat_request(structured_fields.clone()), Ok(()));
    assert_eq!(responses_request(structured_fields), Ok(()));
}

#[test]
fn should_report_an_out_of_range_sampling_setting_in_the_same_words() {
    let chat_error = chat_request(json!({"temperature": 2.5})).expect_err("2.5 is too hot");
    let responses_error =
        responses_request(json!({"temperature": 2.5})).expect_err("2.5 is too hot");
    assert_eq!(
        chat_error.to_string(),
        "temperature is outside the supported range 0..=2"
    );
    assert_eq!(responses_error.to_string(), chat_error.to_string());
    let top_p_error = responses_request(json!({"top_p": -0.1})).expect_err("below zero");
    assert_eq!(
        top_p_error.to_string(),
        "top_p is outside the supported range 0..=1"
    );
}

#[test]
fn should_report_an_out_of_range_output_budget_in_the_same_words() {
    let chat_error = chat_request(json!({"max_tokens": 0})).expect_err("zero tokens");
    let responses_error =
        responses_request(json!({"max_output_tokens": 0})).expect_err("zero tokens");
    assert_eq!(
        chat_error.to_string(),
        format!("output token count is 0, outside the 1..={MAX_OUTPUT_TOKENS} token range")
    );
    assert_eq!(responses_error.to_string(), chat_error.to_string());
    assert!(!chat_error.is_request_too_large());
    assert!(!responses_error.is_request_too_large());
}

#[test]
fn should_accept_the_largest_output_budget_on_both_requests() {
    assert_eq!(
        chat_request(json!({"max_tokens": MAX_OUTPUT_TOKENS})),
        Ok(())
    );
    assert_eq!(
        responses_request(json!({"max_output_tokens": MAX_OUTPUT_TOKENS})),
        Ok(())
    );
}
