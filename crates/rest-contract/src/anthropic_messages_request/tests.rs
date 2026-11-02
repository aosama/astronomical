use serde_json::json;

use crate::{
    AnthropicMessageParts, AnthropicMessagesRequest, AnthropicMessagesRequestParts,
    AnthropicMessagesValidationError, AnthropicToolChoiceParts, AnthropicUserSegmentParts,
    MAX_IMAGE_PARTS_PER_REQUEST,
};

// A one-pixel PNG, small enough to inline and real enough to decode.
const ONE_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn parts_of(request_document: serde_json::Value) -> AnthropicMessagesRequestParts {
    serde_json::from_value::<AnthropicMessagesRequest>(request_document)
        .expect("the request should deserialize")
        .into_parts()
        .expect("the request should validate")
}

fn rejection_of(request_document: serde_json::Value) -> AnthropicMessagesValidationError {
    serde_json::from_value::<AnthropicMessagesRequest>(request_document)
        .expect("the request should deserialize")
        .into_parts()
        .expect_err("the request should be rejected")
}

fn minimal_request() -> serde_json::Value {
    json!({
        "model": "local-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Shall I compare thee to a summer's day?"}]
    })
}

#[test]
fn should_accept_the_minimal_string_content_request() {
    let request_parts = parts_of(minimal_request());

    assert_eq!(request_parts.model, "local-model");
    assert_eq!(request_parts.maximum_output_tokens, 64);
    assert_eq!(request_parts.tool_choice, AnthropicToolChoiceParts::Auto);
    assert_eq!(request_parts.thinking_budget, None);
    assert!(!request_parts.stream);
    assert_eq!(
        request_parts.messages,
        vec![AnthropicMessageParts::User {
            segments: vec![AnthropicUserSegmentParts::Text(
                "Shall I compare thee to a summer's day?".to_owned()
            )]
        }]
    );
}

#[test]
fn should_join_system_blocks_and_absorb_cache_control() {
    let mut request_document = minimal_request();
    request_document["system"] = json!([
        {"type": "text", "text": "You are a sonneteer.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Answer in iambic pentameter."}
    ]);
    request_document["metadata"] = json!({"user_id": "local"});

    let request_parts = parts_of(request_document);

    assert_eq!(
        request_parts.system.as_deref(),
        Some("You are a sonneteer.\n\nAnswer in iambic pentameter.")
    );
}

fn tool_conversation_request() -> serde_json::Value {
    let mut request_document = minimal_request();
    request_document["tools"] = json!([{
        "name": "read_sonnet",
        "description": "Reads one sonnet",
        "input_schema": {"type": "object", "properties": {"number": {"type": "integer"}}}
    }]);
    request_document["messages"] = json!([
        {"role": "user", "content": "Read sonnet 18."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I should read it.", "signature": ""},
            {"type": "tool_use", "id": "toolu_1", "name": "read_sonnet", "input": {"number": 18}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [{"type": "text", "text": "Shall I compare thee"}]},
            {"type": "text", "text": "Now summarize it."}
        ]}
    ]);
    request_document
}

#[test]
fn should_carry_an_assistant_turn_with_thinking_and_tool_use() {
    let request_parts = parts_of(tool_conversation_request());

    assert_eq!(request_parts.tools[0].name, "read_sonnet");
    let AnthropicMessageParts::Assistant {
        thinking,
        tool_uses,
        ..
    } = &request_parts.messages[1]
    else {
        panic!("the second turn should be the assistant turn");
    };
    assert_eq!(thinking, "I should read it.");
    assert_eq!(tool_uses[0].input_json, r#"{"number":18}"#);
}

#[test]
fn should_carry_a_user_turn_with_a_tool_result_before_its_text() {
    let request_parts = parts_of(tool_conversation_request());

    let AnthropicMessageParts::User { segments } = &request_parts.messages[2] else {
        panic!("the third turn should be the user turn");
    };
    assert!(matches!(
        &segments[0],
        AnthropicUserSegmentParts::ToolResult { tool_use_id, text, is_error: false, .. }
            if tool_use_id == "toolu_1" && text == "Shall I compare thee"
    ));
    assert_eq!(
        segments[1],
        AnthropicUserSegmentParts::Text("Now summarize it.".to_owned())
    );
}

#[test]
fn should_decode_a_base64_image_block() {
    let mut request_document = minimal_request();
    request_document["messages"] = json!([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ONE_PIXEL_PNG_BASE64}},
        {"type": "text", "text": "Describe this."}
    ]}]);

    let request_parts = parts_of(request_document);

    let AnthropicMessageParts::User { segments } = &request_parts.messages[0] else {
        panic!("the turn should be a user turn");
    };
    assert!(
        matches!(&segments[0], AnthropicUserSegmentParts::Image(image) if image.mime_type() == "image/png")
    );
}

#[test]
fn should_map_each_thinking_mode_onto_the_worker_budget() {
    for (thinking_object, expected_budget) in [
        (
            json!({"type": "enabled", "budget_tokens": 2048}),
            Some(2048),
        ),
        (json!({"type": "disabled"}), Some(0)),
        (json!({"type": "adaptive"}), None),
    ] {
        let mut request_document = minimal_request();
        request_document["thinking"] = thinking_object;

        assert_eq!(parts_of(request_document).thinking_budget, expected_budget);
    }
}

#[test]
fn should_accept_the_maximum_output_token_boundary_and_reject_beyond_it() {
    let mut request_document = minimal_request();
    request_document["max_tokens"] = json!(65_535);
    assert_eq!(
        parts_of(request_document.clone()).maximum_output_tokens,
        65_535
    );

    request_document["max_tokens"] = json!(65_536);
    assert!(matches!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::OutputTokenCountOutOfRange { .. }
    ));
    request_document["max_tokens"] = json!(0);
    assert!(matches!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::OutputTokenCountOutOfRange { .. }
    ));
}

#[test]
fn should_reject_options_the_worker_cannot_honor() {
    for (field_name, field_value, expected_option_name) in [
        ("stop_sequences", json!(["\n\nHuman:"]), "stop_sequences"),
        ("top_k", json!(40), "top_k"),
        ("tool_choice", json!({"type": "any"}), "tool_choice.type"),
        ("thinking", json!({"type": "sometimes"}), "thinking.type"),
    ] {
        let mut request_document = minimal_request();
        request_document[field_name] = field_value;

        assert_eq!(
            rejection_of(request_document),
            AnthropicMessagesValidationError::UnsupportedOption {
                option_name: expected_option_name
            }
        );
    }
}

#[test]
fn should_accept_an_empty_stop_sequence_list() {
    let mut request_document = minimal_request();
    request_document["stop_sequences"] = json!([]);

    assert!(parts_of(request_document).system.is_none());
}

#[test]
fn should_reject_an_unknown_top_level_field() {
    let mut request_document = minimal_request();
    request_document["mcp_servers"] = json!([]);

    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnknownField {
            field_name: "mcp_servers".to_owned()
        }
    );
}

#[test]
fn should_reject_a_server_tool() {
    let mut request_document = minimal_request();
    request_document["tools"] = json!([{"type": "web_search_20250305", "name": "web_search"}]);

    assert!(matches!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedToolType { .. }
    ));
}

#[test]
fn should_reject_a_url_image_source_and_redacted_thinking() {
    let mut request_document = minimal_request();
    request_document["messages"] = json!([{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
    ]}]);
    assert!(matches!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::UnsupportedImageSource { .. }
    ));

    request_document["messages"] = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]}
    ]);
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedReasoningReplay
    );
}

#[test]
fn should_reject_empty_messages_empty_content_and_unknown_roles() {
    let mut request_document = minimal_request();
    request_document["messages"] = json!([]);
    assert_eq!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::EmptyMessages
    );

    request_document["messages"] = json!([{"role": "user", "content": []}]);
    assert_eq!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::EmptyMessageContent
    );

    request_document["messages"] = json!([{"role": "tool", "content": "hello"}]);
    assert!(matches!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedRole { .. }
    ));
}

#[test]
fn should_reject_temperature_above_the_anthropic_range() {
    let mut request_document = minimal_request();
    request_document["temperature"] = json!(1.0);
    assert!(parts_of(request_document.clone()).temperature.is_some());

    request_document["temperature"] = json!(1.5);
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::SamplingParameterOutOfRange {
            parameter_name: "temperature"
        }
    );
}

#[test]
fn should_reject_more_images_than_the_portable_budget_as_too_large() {
    let image_block = json!({"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": ONE_PIXEL_PNG_BASE64
    }});
    let mut request_document = minimal_request();
    request_document["messages"] = json!([{
        "role": "user",
        "content": vec![image_block; MAX_IMAGE_PARTS_PER_REQUEST + 1]
    }]);

    let validation_error = rejection_of(request_document);

    assert!(validation_error.is_request_too_large());
}

/// The top-level shape Claude Code sends on every request: adaptive thinking
/// with a display preference, an effort level, and context-management edits.
fn claude_code_shaped_request() -> serde_json::Value {
    let mut request_document = minimal_request();
    request_document["thinking"] = json!({"type": "adaptive", "display": "omitted"});
    request_document["output_config"] = json!({"effort": "high"});
    request_document["context_management"] =
        json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]});
    request_document
}

#[test]
fn should_accept_the_claude_code_request_shape_and_map_effort_to_a_budget() {
    let request_parts = parts_of(claude_code_shaped_request());

    assert_eq!(request_parts.thinking_budget, Some(16_384));
}

#[test]
fn should_let_disabled_thinking_override_the_effort_level() {
    let mut request_document = claude_code_shaped_request();
    request_document["thinking"] = json!({"type": "disabled"});

    assert_eq!(parts_of(request_document).thinking_budget, Some(0));
}

#[test]
fn should_reject_an_unknown_effort_level_and_other_output_controls() {
    let mut request_document = claude_code_shaped_request();
    request_document["output_config"] = json!({"effort": "ludicrous"});
    assert!(matches!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::ThinkingControls(_)
    ));

    request_document["output_config"] = json!({"format": {"type": "json_schema"}});
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "output_config"
        }
    );
}

fn system_turn_request(referenced_tool_name: &str) -> serde_json::Value {
    let mut request_document = minimal_request();
    request_document["tools"] = json!([{"name": "read_play"}]);
    request_document["messages"] = json!([
        {"role": "user", "content": "Read the play."},
        {"role": "system", "content": [
            {"type": "text", "text": "The working directory changed."},
            {"type": "tool_addition", "tool": {"type": "tool_reference", "name": referenced_tool_name},
             "cache_control": {"type": "ephemeral"}}
        ]}
    ]);
    request_document
}

#[test]
fn should_keep_system_turn_text_and_its_declared_tool_references() {
    let request_parts = parts_of(system_turn_request("read_play"));

    assert_eq!(
        request_parts.messages[1],
        AnthropicMessageParts::System {
            text: "The working directory changed.".to_owned(),
            referenced_tool_names: vec!["read_play".to_owned()],
            added_tools: Vec::new(),
        }
    );
}

/// The second-turn shape Claude Code sends once it loads a deferred tool:
/// a `tool_definition` addition carrying the whole tool.
fn tool_definition_addition_request(added_tool_name: &str) -> serde_json::Value {
    let mut request_document = minimal_request();
    request_document["tools"] = json!([{"name": "read_play"}]);
    request_document["messages"] = json!([
        {"role": "user", "content": "Read the play."},
        {"role": "system", "content": [
            {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {
                "name": added_tool_name,
                "description": "Searches the folio",
                "input_schema": {"type": "object", "properties": {"line": {"type": "string"}}}
            }}},
            {"type": "tool_addition", "tool": {"type": "tool_reference", "name": added_tool_name}}
        ]}
    ]);
    request_document
}

#[test]
fn should_add_a_tool_defined_mid_conversation_to_the_declared_tools() {
    let request_parts = parts_of(tool_definition_addition_request("search_folio"));

    let mut tool_names = Vec::new();
    for tool_definition in &request_parts.tools {
        tool_names.push(tool_definition.name.as_str());
    }
    assert_eq!(tool_names, vec!["read_play", "search_folio"]);
}

#[test]
fn should_not_duplicate_a_tool_that_is_defined_again_mid_conversation() {
    let request_parts = parts_of(tool_definition_addition_request("read_play"));

    assert_eq!(request_parts.tools.len(), 1);
}

#[test]
fn should_validate_a_mid_conversation_tool_definition_like_a_declared_tool() {
    assert_eq!(
        rejection_of(tool_definition_addition_request("search folio")),
        AnthropicMessagesValidationError::InvalidToolName {
            tool_name: "search folio".to_owned()
        }
    );
}

#[test]
fn should_reject_a_tool_definition_addition_without_its_definition() {
    let mut request_document = minimal_request();
    request_document["messages"] = json!([
        {"role": "user", "content": "Read the play."},
        {"role": "system", "content": [
            {"type": "tool_addition", "tool": {"type": "tool_definition"}}
        ]}
    ]);

    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedContentBlock {
            location: "system message"
        }
    );
}

#[test]
fn should_reject_a_tool_reference_to_an_undeclared_tool() {
    assert_eq!(
        rejection_of(system_turn_request("burn_the_globe")),
        AnthropicMessagesValidationError::UndeclaredToolReference {
            tool_name: "burn_the_globe".to_owned()
        }
    );
}

#[test]
fn should_reject_a_tool_addition_that_carries_more_than_a_reference() {
    let mut request_document = minimal_request();
    request_document["messages"] = json!([
        {"role": "user", "content": "hello"},
        {"role": "system", "content": [
            {"type": "tool_addition", "tool": {"type": "custom", "name": "new_tool"}}
        ]}
    ]);

    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedContentBlock {
            location: "system message"
        }
    );
}

#[test]
fn should_honor_each_thinking_display_value() {
    for (display_value, expected_omitted) in [("summarized", false), ("omitted", true)] {
        let mut request_document = minimal_request();
        request_document["thinking"] = json!({"type": "adaptive", "display": display_value});

        assert_eq!(
            parts_of(request_document).is_thinking_display_omitted,
            expected_omitted
        );
    }
    assert!(!parts_of(minimal_request()).is_thinking_display_omitted);
}

#[test]
fn should_reject_an_unknown_thinking_display() {
    let mut request_document = minimal_request();
    request_document["thinking"] = json!({"type": "adaptive", "display": "interpretive_dance"});

    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "thinking.display"
        }
    );
}

#[test]
fn should_require_a_positive_budget_for_enabled_thinking() {
    for thinking_object in [
        json!({"type": "enabled"}),
        json!({"type": "enabled", "budget_tokens": 0}),
    ] {
        let mut request_document = minimal_request();
        request_document["thinking"] = thinking_object;

        assert_eq!(
            rejection_of(request_document),
            AnthropicMessagesValidationError::ThinkingBudgetRequired
        );
    }
}

#[test]
fn should_reject_disabling_parallel_tool_use() {
    let mut request_document = minimal_request();
    request_document["tool_choice"] = json!({"type": "auto", "disable_parallel_tool_use": true});

    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedOption {
            option_name: "tool_choice.disable_parallel_tool_use"
        }
    );
}

#[test]
fn should_accept_tool_choice_none() {
    let mut request_document = minimal_request();
    request_document["tool_choice"] = json!({"type": "none"});

    assert_eq!(
        parts_of(request_document).tool_choice,
        AnthropicToolChoiceParts::None
    );
}

#[test]
fn should_reject_an_invalid_tool_name_and_an_overly_deep_schema() {
    let mut request_document = minimal_request();
    request_document["tools"] = json!([{"name": "read play"}]);
    assert!(matches!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::InvalidToolName { .. }
    ));

    let mut deep_schema = json!({"type": "object"});
    for _ in 0..crate::MAX_TOOL_SCHEMA_NESTING_DEPTH {
        deep_schema = json!({"properties": deep_schema});
    }
    request_document["tools"] = json!([{"name": "read_play", "input_schema": deep_schema}]);
    assert!(matches!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::ToolSchemaNestingTooDeep { .. }
    ));
}

#[test]
fn should_reject_content_blocks_outside_their_roles() {
    let image_block = json!({"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": ONE_PIXEL_PNG_BASE64
    }});
    let mut request_document = minimal_request();
    request_document["messages"] = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [image_block]}
    ]);
    assert_eq!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::UnsupportedContentBlock {
            location: "assistant message"
        }
    );

    request_document["messages"] = json!([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "read_play", "input": {}}
        ]}
    ]}]);
    assert_eq!(
        rejection_of(request_document.clone()),
        AnthropicMessagesValidationError::UnsupportedContentBlock {
            location: "tool_result content"
        }
    );

    request_document["system"] = json!([{"type": "image", "text": ""}]);
    request_document["messages"] = json!([{"role": "user", "content": "hello"}]);
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::UnsupportedContentBlock { location: "system" }
    );
}

#[test]
fn should_reject_an_empty_model_and_top_p_out_of_range() {
    let mut request_document = minimal_request();
    request_document["model"] = json!("");
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::EmptyModel
    );

    let mut request_document = minimal_request();
    request_document["top_p"] = json!(1.01);
    assert_eq!(
        rejection_of(request_document),
        AnthropicMessagesValidationError::SamplingParameterOutOfRange {
            parameter_name: "top_p"
        }
    );
}

#[test]
fn should_accept_the_lowest_boundaries_of_max_tokens_and_temperature() {
    let mut request_document = minimal_request();
    request_document["max_tokens"] = json!(1);
    request_document["temperature"] = json!(0.0);

    let request_parts = parts_of(request_document);

    assert_eq!(request_parts.maximum_output_tokens, 1);
    assert_eq!(request_parts.temperature, Some(0.0));
}

#[test]
fn should_reject_decoded_images_beyond_the_inline_media_budget_as_too_large() {
    // Four images just under the per-image cap together exceed the
    // request-wide inline-media budget.
    let near_cap_image_bytes = vec![0_u8; crate::MAX_IMAGE_DECODED_BYTES - 1];
    let near_cap_image_base64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &near_cap_image_bytes,
    );
    let image_block = json!({"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": near_cap_image_base64
    }});
    let mut request_document = minimal_request();
    request_document["messages"] = json!([{"role": "user", "content": vec![image_block; 4]}]);

    let validation_error = rejection_of(request_document);

    assert!(matches!(
        validation_error,
        AnthropicMessagesValidationError::InlineMediaExceedsBudget { .. }
    ));
    assert!(validation_error.is_request_too_large());
}
