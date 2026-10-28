use super::*;

/// Issue #1321: a standard OpenAI tool schema that declares the documented
/// `strict` field used to be rejected with an unknown-field error, which failed
/// every tool call. The documented field must be accepted so the request proceeds.
#[test]
fn should_accept_the_documented_strict_tool_field() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [ { "role": "user", "content": "What is 2 + 2?" } ],
        "tools": [ { "type": "function", "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
              "type": "object",
              "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
              },
              "required": [ "a", "b" ]
            },
            "strict": true
        } } ],
        "max_tokens": 16
    }
    "#;

    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the documented strict tool field must be accepted, not rejected as unknown");

    chat_completion_request
        .validate()
        .expect("a request that declares strict must validate");

    let request_parts = chat_completion_request
        .into_parts()
        .expect("a request that declares strict must expose validated parts");

    assert_eq!(request_parts.tools.len(), 1);
    assert_eq!(request_parts.tools[0].name, "add");
}

/// The field is optional and defaults to false, so callers that omit it keep the
/// existing behaviour and must not start failing.
#[test]
fn should_default_the_strict_tool_field_when_omitted() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [ { "role": "user", "content": "What is 2 + 2?" } ],
        "tools": [ { "type": "function", "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
              "type": "object",
              "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
              }
            }
        } } ]
    }
    "#;

    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the request without strict must deserialize as before");

    chat_completion_request
        .validate()
        .expect("a request without strict must validate");
}

/// The documented field is accepted for either truthiness, so a client that sends
/// `strict: false` alongside the schema is unaffected too.
#[test]
fn should_accept_strict_tool_field_set_to_false() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [ { "role": "user", "content": "What is 2 + 2?" } ],
        "tools": [ { "type": "function", "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
              "type": "object",
              "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
              }
            },
            "strict": false
        } } ]
    }
    "#;

    serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("strict:false must be accepted");
}

/// A lenient client may serialize an absent boolean as an explicit null; the field is
/// modeled like the sibling optional fields, so null must be accepted rather than
/// rejected with a type error the way an unknown field is.
#[test]
fn should_accept_an_explicit_strict_null() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [ { "role": "user", "content": "What is 2 + 2?" } ],
        "tools": [ { "type": "function", "function": {
            "name": "add",
            "parameters": { "type": "object" },
            "strict": null
        } } ]
    }
    "#;

    serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("strict:null must be accepted like an omitted strict field");
}

/// Accepting the documented `strict` field must not open the tool function object to
/// undocumented fields: an unknown key still fails deserialization, so a misspelled
/// or future field is reported instead of being silently ignored.
#[test]
fn should_still_reject_undocumented_fields_inside_the_tool_function_object() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [ { "role": "user", "content": "What is 2 + 2?" } ],
        "tools": [ { "type": "function", "function": {
            "name": "add",
            "parameters": { "type": "object" },
            "strick": true
        } } ]
    }
    "#;

    let deserialization_error = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect_err("an undocumented field inside the tool function object must still be rejected");

    let error_text = deserialization_error.to_string();
    assert!(
        error_text.contains("unknown field"),
        "the rejection must name the unknown field, got: {error_text}"
    );
}
