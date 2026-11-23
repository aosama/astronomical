//! Wire-contract tests for images a tool returns in a `function_call_output`.
//!
//! The Responses API lets a tool's result be a list of `input_text` and
//! `input_image` parts, which is how a client such as GitHub Copilot hands a
//! screenshot back to a vision model. These cross no process or model
//! boundary: literal JSON in, validated parts out.

use serde_json::{Value, json};

use crate::RequestBoundsError;
use crate::request_budget::MAX_IMAGE_PARTS_PER_REQUEST;
use crate::{
    OpenAiResponseInputItemParts, OpenAiResponseInputParts, OpenAiResponsesRequest,
    OpenAiResponsesValidationError,
};

/// A 1x1 red PNG, the smallest image the decoder accepts as real.
const RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

fn red_pixel_image_part() -> Value {
    json!({
        "type": "input_image",
        "image_url": format!("data:image/png;base64,{RED_PIXEL_PNG_BASE64}"),
        "detail": "auto",
        "file_id": null
    })
}

fn request_with_tool_output(tool_output: Value) -> OpenAiResponsesRequest {
    serde_json::from_value(json!({
        "model": "astronomical/fake-mixture-of-experts",
        "input": [
            {"role": "user", "content": "Capture the stage."},
            {"type": "function_call", "call_id": "call_1", "name": "screenshot", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": tool_output}
        ]
    }))
    .expect("a function_call_output request should deserialize")
}

fn tool_output_parts(
    request: OpenAiResponsesRequest,
) -> Result<OpenAiResponseInputItemParts, OpenAiResponsesValidationError> {
    let OpenAiResponseInputParts::Items(mut input_items) = request.into_parts()?.input else {
        panic!("expected response input items");
    };
    Ok(input_items.remove(2))
}

#[test]
fn should_decode_text_and_image_parts_of_a_function_call_output() {
    let request = request_with_tool_output(json!([
        {"type": "input_text", "text": "Captured the stage."},
        red_pixel_image_part()
    ]));

    let tool_output = tool_output_parts(request).expect("a screenshot result should validate");

    let OpenAiResponseInputItemParts::FunctionCallOutput {
        call_id,
        output,
        images,
    } = tool_output
    else {
        panic!("expected a function_call_output");
    };
    assert_eq!(call_id, "call_1");
    assert_eq!(output, "Captured the stage.");
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].mime_type(), "image/png");
}

#[test]
fn should_keep_a_plain_text_function_call_output_without_images() {
    let request = request_with_tool_output(json!("Exit, pursued by a bear."));

    let tool_output = tool_output_parts(request).expect("a text result should validate");

    assert!(matches!(
        tool_output,
        OpenAiResponseInputItemParts::FunctionCallOutput { output, images, .. }
            if output == "Exit, pursued by a bear." && images.is_empty()
    ));
}

#[test]
fn should_refuse_a_function_call_output_image_named_by_file_id() {
    let request = request_with_tool_output(json!([{
        "type": "input_image",
        "image_url": format!("data:image/png;base64,{RED_PIXEL_PNG_BASE64}"),
        "file_id": "file-abc123"
    }]));

    let validation_error = tool_output_parts(request)
        .expect_err("a stored-file image this endpoint never received must be refused");

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::UnsupportedOption {
            option_name: "input_image.file_id"
        }
    ));
}

#[test]
fn should_refuse_an_input_image_without_an_image_url_by_name() {
    let request = request_with_tool_output(json!([{"type": "input_image", "detail": "auto"}]));

    let validation_error =
        tool_output_parts(request).expect_err("an image part with no image cannot be decoded");

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::MissingImageUrl
    ));
}

#[test]
fn should_refuse_an_empty_function_call_output_part_list() {
    let request = request_with_tool_output(json!([]));

    let validation_error =
        tool_output_parts(request).expect_err("an empty part list carries no result");

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::EmptyContentParts
    ));
}

#[test]
fn should_refuse_an_http_image_in_a_function_call_output() {
    let request = request_with_tool_output(json!([{
        "type": "input_image",
        "image_url": "https://example.com/screenshot.png"
    }]));

    let validation_error = tool_output_parts(request)
        .expect_err("a remote image must be refused to keep requests local");

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::ImageInput(_)
    ));
}

#[test]
fn should_count_function_call_output_images_toward_the_image_part_budget() {
    let one_image_over_the_budget = vec![red_pixel_image_part(); MAX_IMAGE_PARTS_PER_REQUEST + 1];
    let request = request_with_tool_output(Value::Array(one_image_over_the_budget));

    let validation_error = tool_output_parts(request)
        .expect_err("tool images count toward the same per-request image budget");

    assert!(matches!(
        validation_error,
        OpenAiResponsesValidationError::RequestBounds(
            RequestBoundsError::ImagePartCountExceedsBudget {
                excess_image_part_count: 1,
                ..
            }
        )
    ));
}

#[test]
fn should_accept_function_call_output_images_exactly_at_the_image_part_budget() {
    let images_at_the_budget = vec![red_pixel_image_part(); MAX_IMAGE_PARTS_PER_REQUEST];
    let request = request_with_tool_output(Value::Array(images_at_the_budget));

    let tool_output = tool_output_parts(request).expect("the budget itself is allowed");

    assert_eq!(tool_output.images().len(), MAX_IMAGE_PARTS_PER_REQUEST);
}
