//! Unit tests for `crate::rest::tool_result_images`: holding a run of tool
//! results' images and adding them as one user message after the run.

use ipc_protocol::ChatMessage;
use rest_contract::{DecodedImageInput, OpenAiChatCompletionRequest, OpenAiChatMessageParts};

use super::{PendingToolResultImages, TOOL_RESULT_IMAGES_LABEL, chat_image_input};

/// A 1x1 red PNG, the smallest image the decoder accepts as real.
const RED_PIXEL_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

/// Decoded images obtained the way production gets them: by validating a
/// request, since the contract offers no other constructor.
fn decoded_red_pixels(image_count: usize) -> Vec<DecodedImageInput> {
    let image_parts = vec![
        serde_json::json!({"type": "image_url", "image_url": {"url": RED_PIXEL_DATA_URI}});
        image_count
    ];
    let request = serde_json::from_value::<OpenAiChatCompletionRequest>(serde_json::json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": image_parts}]
    }))
    .expect("the image request should deserialize");
    let request_parts = request
        .into_parts()
        .expect("the image request should validate");
    let Some(OpenAiChatMessageParts::User { images, .. }) =
        request_parts.messages.into_iter().next()
    else {
        panic!("expected one user message");
    };
    images
}

#[test]
fn should_add_no_message_when_no_tool_result_carried_an_image() {
    let mut pending_tool_result_images = PendingToolResultImages::default();
    pending_tool_result_images.hold(&[]);
    let mut chat_messages = Vec::new();

    pending_tool_result_images.flush_into(&mut chat_messages);

    assert!(chat_messages.is_empty());
}

#[test]
fn should_add_every_held_image_as_one_user_message_in_order() {
    let mut pending_tool_result_images = PendingToolResultImages::default();
    let first_result_images = decoded_red_pixels(2);
    let second_result_images = decoded_red_pixels(1);
    pending_tool_result_images.hold(&first_result_images);
    pending_tool_result_images.hold(&second_result_images);
    let mut chat_messages = Vec::new();

    pending_tool_result_images.flush_into(&mut chat_messages);

    assert!(matches!(
        chat_messages.as_slice(),
        [ChatMessage::User { content, images }]
            if content == TOOL_RESULT_IMAGES_LABEL && images.len() == 3
    ));
}

#[test]
fn should_start_empty_again_after_a_flush() {
    let mut pending_tool_result_images = PendingToolResultImages::default();
    pending_tool_result_images.hold(&decoded_red_pixels(1));
    let mut chat_messages = Vec::new();
    pending_tool_result_images.flush_into(&mut chat_messages);

    pending_tool_result_images.flush_into(&mut chat_messages);

    assert_eq!(chat_messages.len(), 1, "a second flush adds nothing");
}

#[test]
fn should_carry_the_image_bytes_and_type_unchanged() {
    let decoded_images = decoded_red_pixels(1);

    let worker_image = chat_image_input(&decoded_images[0]);

    assert_eq!(worker_image.mime_type, "image/png");
    assert_eq!(
        worker_image.decoded_bytes,
        decoded_images[0].decoded_bytes()
    );
}

#[test]
fn should_keep_holding_while_the_next_message_is_another_tool_result() {
    let mut pending_tool_result_images = PendingToolResultImages::default();
    pending_tool_result_images.hold(&decoded_red_pixels(1));
    let mut chat_messages = Vec::new();

    pending_tool_result_images.end_run_before(true, &mut chat_messages);
    assert!(
        chat_messages.is_empty(),
        "the run of tool results continues"
    );

    pending_tool_result_images.end_run_before(false, &mut chat_messages);
    assert_eq!(chat_messages.len(), 1, "a non-tool message ends the run");
}
