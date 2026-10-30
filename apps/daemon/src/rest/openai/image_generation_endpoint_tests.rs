//! Beside-source coverage for the pure image-endpoint rejections: the request
//! body parse and the validation-error to OpenAI-shaped HTTP 400 mapping. The
//! endpoint orchestration itself runs under the daemon integration journeys.

use axum::http::StatusCode;

use super::{image_validation_rejection_response, parse_image_generation_request};
use astronomical_rest_contract::OpenAiImageGenerationValidationError;

#[test]
fn should_parse_a_valid_image_generation_request_body() {
    let image_request = parse_image_generation_request(
        br#"{"model":"qwen-image","prompt":"a lighthouse at dusk","width":512,"height":512,"response_format":"b64_json"}"#,
    );

    assert!(image_request.is_ok());
}

#[test]
fn should_reject_malformed_json_with_http_400() {
    let rejection = parse_image_generation_request(b"not json");

    assert_eq!(rejection.unwrap_err().status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_reject_an_empty_request_body_with_http_400() {
    let rejection = parse_image_generation_request(b"");

    assert_eq!(rejection.unwrap_err().status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_still_parse_unknown_fields_and_leave_rejection_to_validation() {
    let image_request = parse_image_generation_request(
        br#"{"model":"qwen-image","prompt":"a lighthouse at dusk","width":512,"height":512,"response_format":"b64_json","quality":"high"}"#,
    );

    assert!(image_request.is_ok());
}

#[test]
fn should_map_dimension_rejections_to_http_400() {
    let response = image_validation_rejection_response(
        OpenAiImageGenerationValidationError::UnsupportedDimension {
            parameter_name: "width",
            actual_pixels: 33,
            minimum_pixels: 64,
            maximum_pixels: 1_024,
        },
    );

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_map_unknown_field_rejections_to_http_400() {
    let response =
        image_validation_rejection_response(OpenAiImageGenerationValidationError::UnknownField {
            field_name: "quality".to_owned(),
        });

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_map_image_count_rejections_to_http_400() {
    let response = image_validation_rejection_response(
        OpenAiImageGenerationValidationError::UnsupportedImageCount { actual_images: 2 },
    );

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
