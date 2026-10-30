//! Beside-source coverage for the pure embeddings-endpoint units: the request
//! body parse, the validation-error to OpenAI-shaped HTTP 400 mapping, and
//! the IPC command translation. The endpoint orchestration itself runs under
//! the daemon integration journeys.

use axum::http::StatusCode;

use super::{
    build_embeddings_command, embeddings_validation_parameter,
    embeddings_validation_rejection_response, parse_embeddings_request,
};
use astronomical_ipc_protocol::{EmbeddingEncodingFormat, RequestId};
use astronomical_rest_contract::{
    OpenAiEmbeddingEncodingFormat as RestEncodingFormat, OpenAiEmbeddingsRequestParts,
    OpenAiEmbeddingsValidationError,
};

fn sample_request_parts(encoding_format: RestEncodingFormat) -> OpenAiEmbeddingsRequestParts {
    OpenAiEmbeddingsRequestParts {
        model: "embedder".to_owned(),
        inputs: vec!["shade".to_owned()],
        encoding_format,
        dimensions: None,
    }
}

#[test]
fn should_parse_a_valid_embeddings_request_body() {
    let embeddings_request = parse_embeddings_request(
        br#"{"model":"embedder","input":"shade","encoding_format":"float"}"#,
    );

    assert!(embeddings_request.is_ok());
}

#[test]
fn should_reject_malformed_json_with_http_400() {
    let rejection = parse_embeddings_request(b"not json");

    assert_eq!(rejection.unwrap_err().status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_reject_an_empty_request_body_with_http_400() {
    let rejection = parse_embeddings_request(b"");

    assert_eq!(rejection.unwrap_err().status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_still_parse_unknown_fields_and_leave_rejection_to_validation() {
    let embeddings_request =
        parse_embeddings_request(br#"{"model":"embedder","input":"shade","top_k":3}"#);

    assert!(embeddings_request.is_ok());
}

#[test]
fn should_map_validation_rejections_to_http_400() {
    let response =
        embeddings_validation_rejection_response(OpenAiEmbeddingsValidationError::EmptyModel);

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn should_map_each_validation_error_to_its_request_parameter() {
    assert_eq!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::EmptyModel),
        "model"
    );
    assert_eq!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::EmptyInput),
        "input"
    );
    assert_eq!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::InvalidDimensions),
        "dimensions"
    );
    assert_eq!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::UnknownField {
            field_name: "top_k".to_owned()
        }),
        "request"
    );
    assert_eq!(
        embeddings_validation_parameter(
            &OpenAiEmbeddingsValidationError::UnsupportedEncodingFormat {
                encoding_format: "float16".to_owned()
            }
        ),
        "encoding_format"
    );
}

#[test]
fn should_not_blame_the_model_for_input_sized_rejections() {
    // Negative: a non-model failure must never be blamed on "model".
    assert_ne!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::EmptyInput),
        "model"
    );
    // Boundary: every input-sized rejection shares the "input" parameter.
    assert_eq!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::InputCountExceeded {
            actual_input_count: 257,
            maximum_input_count: 256,
        }),
        "input"
    );
    assert_eq!(
        embeddings_validation_parameter(&OpenAiEmbeddingsValidationError::InputTextTooLarge {
            actual_input_bytes: 8_193,
            maximum_input_bytes: 8_192,
        }),
        "input"
    );
    assert_eq!(
        embeddings_validation_parameter(
            &OpenAiEmbeddingsValidationError::TotalInputBytesExceeded {
                actual_total_bytes: 1_000_001,
                maximum_total_bytes: 1_000_000,
            }
        ),
        "input"
    );
}

#[test]
fn should_build_a_float_command_from_validated_parts() {
    let embeddings_command = build_embeddings_command(
        11,
        "embedder",
        sample_request_parts(RestEncodingFormat::Float),
    );

    assert_eq!(embeddings_command.request_id, RequestId::new(11));
    assert_eq!(embeddings_command.model, "embedder");
    assert_eq!(embeddings_command.inputs, vec!["shade".to_owned()]);
    assert_eq!(
        embeddings_command.encoding_format,
        EmbeddingEncodingFormat::Float
    );
    assert_eq!(embeddings_command.dimensions, None);
}

#[test]
fn should_translate_base64_encoding_into_the_ipc_command() {
    let embeddings_command = build_embeddings_command(
        11,
        "embedder",
        sample_request_parts(RestEncodingFormat::Base64),
    );

    assert_eq!(
        embeddings_command.encoding_format,
        EmbeddingEncodingFormat::Base64
    );
}

#[test]
fn should_keep_requested_dimensions_in_the_ipc_command() {
    let mut request_parts = sample_request_parts(RestEncodingFormat::Float);
    request_parts.dimensions = Some(768);

    let embeddings_command = build_embeddings_command(11, "embedder", request_parts);

    assert_eq!(embeddings_command.dimensions, Some(768));
}

#[test]
fn should_not_leak_another_request_identifier_into_the_command() {
    let embeddings_command = build_embeddings_command(
        11,
        "embedder",
        sample_request_parts(RestEncodingFormat::Float),
    );

    assert_ne!(embeddings_command.request_id, RequestId::new(12));
}
