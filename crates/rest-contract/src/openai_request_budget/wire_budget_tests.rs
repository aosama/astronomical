use crate::{
    MAX_OPENAI_IMAGE_DECODED_BYTES, MAX_OPENAI_IMAGE_PARTS_PER_REQUEST,
    MAX_OPENAI_INLINE_MEDIA_BYTES, MAX_OPENAI_REQUEST_BODY_BYTES, OpenAiChatCompletionRequest,
    OpenAiChatCompletionValidationError, OpenAiChatMessageParts, OpenAiResponsesRequest,
    OpenAiResponsesValidationError,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::json;

const MODEL_ID: &str = "astronomical/fake-mixture-of-experts";

fn image_data_uri(decoded_byte_count: usize) -> String {
    format!(
        "data:image/png;base64,{}",
        BASE64_STANDARD.encode(vec![0x42u8; decoded_byte_count])
    )
}

fn chat_request_with_image_data_uris(image_data_uris: &[String]) -> OpenAiChatCompletionRequest {
    let content_parts =
        std::iter::once(json!({"type": "text", "text": "Describe the screenshots."}))
            .chain(image_data_uris.iter().map(
                |image_data_uri| json!({"type": "image_url", "image_url": {"url": image_data_uri}}),
            ))
            .collect::<Vec<_>>();
    let request_json = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content_parts}]
    })
    .to_string();
    serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
        .expect("the chat request with image parts should deserialize")
}

fn responses_request_with_image_data_uris(image_data_uris: &[String]) -> OpenAiResponsesRequest {
    let content_parts =
        std::iter::once(json!({"type": "input_text", "text": "Describe the screenshots."}))
            .chain(image_data_uris.iter().map(|image_data_uri| {
                json!({
                    "type": "input_image",
                    "image_url": image_data_uri,
                    "detail": "auto"
                })
            }))
            .collect::<Vec<_>>();
    let request_json = json!({
        "model": MODEL_ID,
        "input": [{"role": "user", "content": content_parts}]
    })
    .to_string();
    serde_json::from_str::<OpenAiResponsesRequest>(&request_json)
        .expect("the Responses request with image parts should deserialize")
}

#[test]
fn should_pin_the_budget_values_to_their_documented_mainstream_figures() {
    // Guard against accidental drift: each value is documented with its source
    // and retrieval date in the openai_request_budget module. Retrieved
    // 2026-10-22 from the vendor pages cited there.
    assert_eq!(MAX_OPENAI_REQUEST_BODY_BYTES, 32_000_000);
    assert_eq!(MAX_OPENAI_IMAGE_DECODED_BYTES, 7 * 1024 * 1024);
    assert_eq!(MAX_OPENAI_INLINE_MEDIA_BYTES, 20 * 1024 * 1024);
    assert_eq!(MAX_OPENAI_IMAGE_PARTS_PER_REQUEST, 100);
}

#[test]
fn should_accept_a_chat_image_at_exactly_the_per_image_budget() {
    let chat_completion_request =
        chat_request_with_image_data_uris(&[image_data_uri(MAX_OPENAI_IMAGE_DECODED_BYTES)]);

    chat_completion_request
        .validate()
        .expect("an image at exactly the per-image budget should be accepted");
    chat_completion_request
        .into_parts()
        .expect("an image at exactly the per-image budget should decode");
}

#[test]
fn should_reject_a_chat_image_one_byte_over_the_per_image_budget() {
    let chat_completion_request =
        chat_request_with_image_data_uris(&[image_data_uri(MAX_OPENAI_IMAGE_DECODED_BYTES + 1)]);

    chat_completion_request
        .validate()
        .expect("the image-count budget is still respected so validate passes");
    let validation_error = chat_completion_request
        .into_parts()
        .expect_err("one byte over the per-image budget must be rejected during decode");

    assert_eq!(
        validation_error,
        OpenAiChatCompletionValidationError::ImageTooLarge {
            actual_bytes: MAX_OPENAI_IMAGE_DECODED_BYTES + 1,
            maximum_bytes: MAX_OPENAI_IMAGE_DECODED_BYTES,
            excess_bytes: 1,
        }
    );
    assert!(validation_error.is_request_too_large());
    let rejection_message = validation_error.to_string();
    assert!(rejection_message.contains("by 1 byte"));
    assert!(rejection_message.contains("Google Cloud Gemini Enterprise"));
}

#[test]
fn should_accept_a_chat_request_at_exactly_the_inline_media_budget() {
    let first_image_decoded_bytes = MAX_OPENAI_IMAGE_DECODED_BYTES;
    let second_image_decoded_bytes = MAX_OPENAI_IMAGE_DECODED_BYTES;
    let third_image_decoded_bytes =
        MAX_OPENAI_INLINE_MEDIA_BYTES - first_image_decoded_bytes - second_image_decoded_bytes;
    let chat_completion_request = chat_request_with_image_data_uris(&[
        image_data_uri(first_image_decoded_bytes),
        image_data_uri(second_image_decoded_bytes),
        image_data_uri(third_image_decoded_bytes),
    ]);

    let request_parts = chat_completion_request
        .into_parts()
        .expect("images summing to exactly the inline-media budget should be accepted");

    let total_decoded_media_bytes = request_parts
        .messages
        .iter()
        .map(|message_parts| match message_parts {
            OpenAiChatMessageParts::User { images, .. } => images
                .iter()
                .map(|image_input| image_input.decoded_bytes().len())
                .sum::<usize>(),
            _ => 0,
        })
        .sum::<usize>();
    assert_eq!(total_decoded_media_bytes, MAX_OPENAI_INLINE_MEDIA_BYTES);
}

#[test]
fn should_reject_a_chat_request_one_byte_over_the_inline_media_budget() {
    let first_image_decoded_bytes = MAX_OPENAI_IMAGE_DECODED_BYTES;
    let second_image_decoded_bytes = MAX_OPENAI_IMAGE_DECODED_BYTES;
    let third_image_decoded_bytes =
        MAX_OPENAI_INLINE_MEDIA_BYTES - first_image_decoded_bytes - second_image_decoded_bytes + 1;
    let chat_completion_request = chat_request_with_image_data_uris(&[
        image_data_uri(first_image_decoded_bytes),
        image_data_uri(second_image_decoded_bytes),
        image_data_uri(third_image_decoded_bytes),
    ]);

    let validation_error = chat_completion_request
        .into_parts()
        .expect_err("one byte over the inline-media budget must be rejected");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::InlineMediaExceedsBudget {
            maximum_inline_media_bytes: MAX_OPENAI_INLINE_MEDIA_BYTES,
            excess_inline_media_bytes: 1,
            ..
        }
    ));
    assert!(validation_error.is_request_too_large());
    assert!(validation_error.to_string().contains("Gemini API"));
}

#[test]
fn should_accept_a_chat_request_with_exactly_the_image_count_budget() {
    let image_data_uris = (0..MAX_OPENAI_IMAGE_PARTS_PER_REQUEST)
        .map(|_| image_data_uri(1))
        .collect::<Vec<_>>();
    let chat_completion_request = chat_request_with_image_data_uris(&image_data_uris);

    chat_completion_request
        .into_parts()
        .expect("exactly the image-count budget of images should be accepted");
}

#[test]
fn should_reject_a_chat_request_one_image_over_the_image_count_budget() {
    let image_data_uris = (0..=MAX_OPENAI_IMAGE_PARTS_PER_REQUEST)
        .map(|_| image_data_uri(1))
        .collect::<Vec<_>>();
    let chat_completion_request = chat_request_with_image_data_uris(&image_data_uris);

    let validation_error = chat_completion_request
        .validate()
        .expect_err("one image over the image-count budget must be rejected");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::ImagePartCountExceedsBudget {
            actual_image_part_count: 101,
            maximum_image_part_count: MAX_OPENAI_IMAGE_PARTS_PER_REQUEST,
            excess_image_part_count: 1,
        }
    ));
    assert!(validation_error.is_request_too_large());
    assert!(validation_error.to_string().contains("Anthropic"));
}

#[test]
fn should_reject_the_image_count_before_decoding_any_oversized_image() {
    // Every image individually busts the per-image budget, so any decode would
    // report ImageTooLarge; the image-count rejection proves the count guard
    // runs ahead of all base64 decode work.
    let image_data_uris = (0..=MAX_OPENAI_IMAGE_PARTS_PER_REQUEST)
        .map(|_| image_data_uri(MAX_OPENAI_IMAGE_DECODED_BYTES + 1))
        .collect::<Vec<_>>();
    let chat_completion_request = chat_request_with_image_data_uris(&image_data_uris);

    let validation_error = chat_completion_request
        .validate()
        .expect_err("the image-count budget must fire before any decode");

    assert!(matches!(
        validation_error,
        OpenAiChatCompletionValidationError::ImagePartCountExceedsBudget { .. }
    ));
}

#[test]
fn should_accept_a_responses_request_within_both_image_budgets() {
    let responses_request =
        responses_request_with_image_data_uris(&[image_data_uri(1_000), image_data_uri(1_000)]);

    responses_request
        .into_parts()
        .expect("small Responses image input should be accepted");
}

#[test]
fn should_reject_a_responses_request_one_byte_over_the_inline_media_budget() {
    // Each image individually stays inside the per-image budget so the
    // rejection proves the cross-image inline-media accumulation fires.
    let first_image_decoded_bytes = MAX_OPENAI_IMAGE_DECODED_BYTES;
    let second_image_decoded_bytes = MAX_OPENAI_IMAGE_DECODED_BYTES;
    let third_image_decoded_bytes =
        MAX_OPENAI_INLINE_MEDIA_BYTES - first_image_decoded_bytes - second_image_decoded_bytes + 1;
    let responses_request = responses_request_with_image_data_uris(&[
        image_data_uri(first_image_decoded_bytes),
        image_data_uri(second_image_decoded_bytes),
        image_data_uri(third_image_decoded_bytes),
    ]);

    let validation_error = responses_request.into_parts().expect_err(
        "one byte over the inline-media budget must be rejected on the Responses surface",
    );

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::InlineMediaExceedsBudget {
            maximum_inline_media_bytes: MAX_OPENAI_INLINE_MEDIA_BYTES,
            excess_inline_media_bytes: 1,
            ..
        }
    ));
    assert!(validation_error.is_request_too_large());
}

#[test]
fn should_accept_a_responses_request_at_the_image_count_budget() {
    let image_data_uris = (0..MAX_OPENAI_IMAGE_PARTS_PER_REQUEST)
        .map(|_| image_data_uri(1))
        .collect::<Vec<_>>();
    let responses_request = responses_request_with_image_data_uris(&image_data_uris);

    responses_request.into_parts().expect(
        "exactly the image-count budget of images should be accepted on the Responses surface",
    );
}

#[test]
fn should_reject_a_responses_request_one_image_over_the_image_count_budget() {
    let image_data_uris = (0..=MAX_OPENAI_IMAGE_PARTS_PER_REQUEST)
        .map(|_| image_data_uri(1))
        .collect::<Vec<_>>();
    let responses_request = responses_request_with_image_data_uris(&image_data_uris);

    let validation_error = responses_request.into_parts().expect_err(
        "one image over the image-count budget must be rejected on the Responses surface",
    );

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::ImagePartCountExceedsBudget {
            actual_image_part_count: 101,
            ..
        }
    ));
    assert!(validation_error.is_request_too_large());
}

#[test]
fn should_accept_a_screenshot_heavy_chat_request_that_fits_every_mainstream_ceiling() {
    // Journey test for issue #689: three 1,000,000-byte screenshots stay inside
    // every mainstream ceiling, so the locally accepted request must also be
    // accepted by a simulation of each. Vendor figures retrieved 2026-10-22:
    // - Anthropic Messages API: 32 MB request body, 10 MB base64 per image,
    //   100 images per request for 200k-context models
    //   (platform.claude.com/docs/en/build-with-claude/vision).
    // - Gemini API: 20 MB total request size for inline data plus text and
    //   system instructions; 3,600 images per request; 7 MB per inline file on
    //   Google Cloud Gemini Enterprise
    //   (ai.google.dev/gemini-api/docs/image-understanding,
    //   docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/image-understanding).
    // - OpenAI: 512 MB total payload, 1,500 images per request
    //   (developers.openai.com/api/docs/guides/images-vision).
    // - Hosted serverless platforms: 4.5 MB request body
    //   (vercel.com/docs/functions/limitations).
    const ANTHROPIC_REQUEST_BODY_BYTES: usize = 32_000_000;
    const ANTHROPIC_PER_IMAGE_BASE64_BYTES: usize = 10_000_000;
    const ANTHROPIC_IMAGE_COUNT: usize = 100;
    const GEMINI_TOTAL_REQUEST_BYTES: usize = 20_000_000;
    const GEMINI_ENTERPRISE_PER_FILE_BYTES: usize = 7_000_000;
    const GEMINI_IMAGE_COUNT: usize = 3_600;
    const OPENAI_TOTAL_PAYLOAD_BYTES: usize = 512_000_000;
    const OPENAI_IMAGE_COUNT: usize = 1_500;
    const SERVERLESS_PLATFORM_BODY_BYTES: usize = 4_500_000;

    let screenshot_decoded_bytes = 1_000_000;
    let chat_completion_request = chat_request_with_image_data_uris(&[
        image_data_uri(screenshot_decoded_bytes),
        image_data_uri(screenshot_decoded_bytes),
        image_data_uri(screenshot_decoded_bytes),
    ]);

    chat_completion_request
        .validate()
        .expect("Astronomical must accept the screenshot-heavy request");
    chat_completion_request
        .into_parts()
        .expect("Astronomical must decode the screenshot-heavy request");

    // The wire simulation measures the exact JSON body a client sends.
    let content_parts = json!({
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the screenshots."},
                {"type": "image_url", "image_url": {"url": image_data_uri(screenshot_decoded_bytes)}},
                {"type": "image_url", "image_url": {"url": image_data_uri(screenshot_decoded_bytes)}},
                {"type": "image_url", "image_url": {"url": image_data_uri(screenshot_decoded_bytes)}}
            ]
        }]
    })
    .to_string();
    let wire_body_bytes = content_parts.len();
    let decoded_bytes_per_image = screenshot_decoded_bytes;
    let wire_bytes_per_image = crate::base64_wire_byte_count(decoded_bytes_per_image);
    let image_part_count = 3;

    assert!(wire_body_bytes <= ANTHROPIC_REQUEST_BODY_BYTES);
    assert!(wire_bytes_per_image <= ANTHROPIC_PER_IMAGE_BASE64_BYTES);
    assert!(image_part_count <= ANTHROPIC_IMAGE_COUNT);
    assert!(wire_body_bytes <= GEMINI_TOTAL_REQUEST_BYTES);
    assert!(decoded_bytes_per_image <= GEMINI_ENTERPRISE_PER_FILE_BYTES);
    assert!(image_part_count <= GEMINI_IMAGE_COUNT);
    assert!(wire_body_bytes <= OPENAI_TOTAL_PAYLOAD_BYTES);
    assert!(image_part_count <= OPENAI_IMAGE_COUNT);
    assert!(wire_body_bytes <= SERVERLESS_PLATFORM_BODY_BYTES);
}
