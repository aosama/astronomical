//! Images a tool returns in a Chat Completions `tool` message, such as a
//! screenshot. They used to be accepted and silently dropped; they now reach
//! the parts the daemon translates, under the same per-request budgets.

use serde_json::{Value, json};

use super::*;
use crate::request_budget::MAX_IMAGE_PARTS_PER_REQUEST;

/// A 1x1 red PNG, the smallest image the decoder accepts as real.
const RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

fn red_pixel_image_part() -> Value {
    json!({
        "type": "image_url",
        "image_url": {"url": format!("data:image/png;base64,{RED_PIXEL_PNG_BASE64}")}
    })
}

fn request_with_tool_content(tool_content: Value) -> OpenAiChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Capture the stage."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "screenshot", "arguments": "{}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_content}
        ]
    }))
    .expect("a tool message request should deserialize")
}

#[test]
fn should_keep_image_parts_of_a_tool_message() {
    let request = request_with_tool_content(json!([
        {"type": "text", "text": "Captured the stage."},
        red_pixel_image_part()
    ]));

    let request_parts = request
        .into_parts()
        .expect("a screenshot result should validate");

    let Some(OpenAiChatMessageParts::Tool {
        tool_call_id,
        content,
        images,
    }) = request_parts.messages.last()
    else {
        panic!("expected the tool message last");
    };
    assert_eq!(tool_call_id, "call_1");
    assert_eq!(content, "Captured the stage.");
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].mime_type(), "image/png");
}

#[test]
fn should_keep_a_plain_text_tool_message_without_images() {
    let request = request_with_tool_content(json!("Exit, pursued by a bear."));

    let request_parts = request.into_parts().expect("a text result should validate");

    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Tool { content, images, .. })
            if content == "Exit, pursued by a bear." && images.is_empty()
    ));
}

#[test]
fn should_reject_an_http_image_in_a_tool_message() {
    let request = request_with_tool_content(json!([{
        "type": "image_url",
        "image_url": {"url": "https://example.com/screenshot.png"}
    }]));

    let validation_error = request
        .into_parts()
        .expect_err("a remote image must be refused to keep requests local");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme
    ));
}

#[test]
fn should_count_tool_message_images_toward_the_image_part_budget() {
    let one_image_over_the_budget = vec![red_pixel_image_part(); MAX_IMAGE_PARTS_PER_REQUEST + 1];
    let request = request_with_tool_content(Value::Array(one_image_over_the_budget));

    let validation_error = request
        .into_parts()
        .expect_err("tool images count toward the same per-request image budget");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::ImagePartCountExceedsBudget {
            excess_image_part_count: 1,
            ..
        }
    ));
}

#[test]
fn should_accept_tool_message_images_exactly_at_the_image_part_budget() {
    let images_at_the_budget = vec![red_pixel_image_part(); MAX_IMAGE_PARTS_PER_REQUEST];
    let request = request_with_tool_content(Value::Array(images_at_the_budget));

    let request_parts = request.into_parts().expect("the budget itself is allowed");

    assert!(matches!(
        request_parts.messages.last(),
        Some(OpenAiChatMessageParts::Tool { images, .. }) if images.len() == MAX_IMAGE_PARTS_PER_REQUEST
    ));
}
