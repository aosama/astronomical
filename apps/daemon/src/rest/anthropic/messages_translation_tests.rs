use astronomical_ipc_protocol::{ChatMessage, ChatToolChoice, RequestId};
use astronomical_rest_contract::{AnthropicMessagesRequest, AnthropicMessagesRequestParts};
use serde_json::json;

use super::messages_translation::{
    AnthropicMessagesTranslationError, translate_anthropic_messages_request_parts,
};

fn request_parts_of(request_document: serde_json::Value) -> AnthropicMessagesRequestParts {
    serde_json::from_value::<AnthropicMessagesRequest>(request_document)
        .expect("the request should deserialize")
        .into_parts()
        .expect("the request should validate")
}

fn translated_messages(request_document: serde_json::Value) -> Vec<ChatMessage> {
    translate_anthropic_messages_request_parts(
        RequestId::new(7),
        request_parts_of(request_document),
    )
    .expect("the request should translate")
    .messages
}

#[test]
fn should_place_the_system_prompt_first_and_carry_sampling_settings() {
    let chat_generation_command = translate_anthropic_messages_request_parts(
        RequestId::new(7),
        request_parts_of(json!({
            "model": "local-model",
            "max_tokens": 512,
            "temperature": 0.7,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": "Speak as Prospero.",
            "messages": [{"role": "user", "content": "Our revels now are ended."}]
        })),
    )
    .expect("the request should translate");

    assert_eq!(
        chat_generation_command.messages[0],
        ChatMessage::System {
            content: "Speak as Prospero.".to_owned()
        }
    );
    assert_eq!(chat_generation_command.settings.max_output_tokens, 512);
    assert_eq!(
        chat_generation_command.settings.temperature_thousandths,
        Some(700)
    );
    assert_eq!(chat_generation_command.settings.thinking_budget, Some(1024));
    assert_eq!(chat_generation_command.tool_choice, ChatToolChoice::Auto);
}

#[test]
fn should_split_a_user_turn_into_tool_messages_then_its_text() {
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "tools": [{"name": "read_play", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "Read The Tempest."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read_play", "input": {"act": 4}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "We are such stuff"},
                {"type": "text", "text": "Quote the next line."}
            ]}
        ]
    }));

    assert!(matches!(
        &chat_messages[1],
        ChatMessage::Assistant { tool_calls, content: None, .. }
            if tool_calls[0].id == "toolu_1" && tool_calls[0].function.arguments_json == r#"{"act":4}"#
    ));
    assert_eq!(
        chat_messages[2],
        ChatMessage::Tool {
            tool_call_id: "toolu_1".to_owned(),
            content: "We are such stuff".to_owned()
        }
    );
    assert!(matches!(
        &chat_messages[3],
        ChatMessage::User { content, .. } if content == "Quote the next line."
    ));
    assert_eq!(chat_messages.len(), 4);
}

#[test]
fn should_prefix_a_failed_tool_result_so_the_model_sees_the_failure() {
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "tools": [{"name": "read_play"}],
        "messages": [
            {"role": "user", "content": "Read it."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read_play", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file not found", "is_error": true}
            ]}
        ]
    }));

    assert_eq!(
        chat_messages[2],
        ChatMessage::Tool {
            tool_call_id: "toolu_1".to_owned(),
            content: "Error: file not found".to_owned()
        }
    );
    assert_eq!(chat_messages.len(), 3);
}

#[test]
fn should_reject_a_tool_result_that_answers_no_declared_tool_call() {
    let translation_outcome = translate_anthropic_messages_request_parts(
        RequestId::new(7),
        request_parts_of(json!({
            "model": "local-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_missing", "content": "orphan"}
            ]}]
        })),
    );

    assert!(matches!(
        translation_outcome,
        Err(AnthropicMessagesTranslationError::IpcValidation(_))
    ));
}

#[test]
fn should_place_a_later_system_turn_as_a_system_update_on_the_prior_user_message() {
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Read the play."},
            {"role": "system", "content": [{"type": "text", "text": "Prefer Act <4>."}]}
        ]
    }));

    assert_eq!(chat_messages.len(), 1);
    assert_eq!(
        chat_messages[0],
        ChatMessage::User {
            content: "Read the play.\n<system-update>\nPrefer Act &lt;4&gt;.\n</system-update>"
                .to_owned(),
            images: Vec::new()
        }
    );
}

#[test]
fn should_drop_a_system_turn_that_held_only_tool_references() {
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "tools": [{"name": "read_play"}],
        "messages": [
            {"role": "user", "content": "Read the play."},
            {"role": "system", "content": [
                {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "read_play"}}
            ]}
        ]
    }));

    assert!(matches!(
        &chat_messages[..],
        [ChatMessage::User { content, .. }] if content == "Read the play."
    ));
}

#[test]
fn should_move_tool_result_images_into_the_following_user_message() {
    // A one-pixel PNG, small enough to inline and real enough to decode.
    let one_pixel_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "tools": [{"name": "screenshot"}],
        "messages": [
            {"role": "user", "content": "Capture the stage."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                    {"type": "text", "text": "captured"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": one_pixel_png_base64}}
                ]},
                {"type": "text", "text": "Describe it."}
            ]}
        ]
    }));

    assert!(matches!(
        &chat_messages[2],
        ChatMessage::Tool { content, .. } if content == "captured"
    ));
    assert!(matches!(
        &chat_messages[3],
        ChatMessage::User { content, images } if content == "Describe it." && images.len() == 1
    ));
}

#[test]
fn should_join_consecutive_user_text_blocks_with_a_newline() {
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Our revels now are ended."},
            {"type": "text", "text": "These our actors were all spirits."}
        ]}]
    }));

    assert_eq!(
        chat_messages,
        vec![ChatMessage::User {
            content: "Our revels now are ended.\nThese our actors were all spirits.".to_owned(),
            images: Vec::new()
        }]
    );
}

#[test]
fn should_reject_a_thinking_budget_the_worker_cannot_represent() {
    let translation_outcome = translate_anthropic_messages_request_parts(
        RequestId::new(7),
        request_parts_of(json!({
            "model": "local-model",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 65_536},
            "messages": [{"role": "user", "content": "hello"}]
        })),
    );

    assert!(matches!(
        translation_outcome,
        Err(AnthropicMessagesTranslationError::ThinkingBudgetTooLarge)
    ));
}

#[test]
fn should_fold_a_leading_system_turn_into_the_top_level_system_prompt() {
    let chat_messages = translated_messages(json!({
        "model": "local-model",
        "max_tokens": 64,
        "system": "Speak as Prospero.",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Answer in verse."}]},
            {"role": "user", "content": "Speak."}
        ]
    }));

    assert_eq!(
        chat_messages[0],
        ChatMessage::System {
            content: "Speak as Prospero.\n\nAnswer in verse.".to_owned()
        }
    );
    assert!(matches!(&chat_messages[1], ChatMessage::User { content, .. } if content == "Speak."));
}
