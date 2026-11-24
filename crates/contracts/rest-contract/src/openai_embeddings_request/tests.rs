//! Wire-contract tests for the embeddings request DTO.
//!
//! Moved from the integration lane (issue #1610): every test exercises
//! exactly this module's parse-and-validate surface with literal JSON.

use crate::{OpenAiEmbeddingEncodingFormat, OpenAiEmbeddingsRequest};

#[test]
fn should_validate_a_single_string_input_into_one_part() {
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(
        r#"{
            "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
            "input": "O Romeo, Romeo, wherefore art thou Romeo?"
        }"#,
    )
    .expect("single-input embeddings request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("single-input embeddings request should validate");

    assert_eq!(
        request_parts.inputs,
        vec!["O Romeo, Romeo, wherefore art thou Romeo?".to_owned()]
    );
    assert_eq!(
        request_parts.encoding_format,
        OpenAiEmbeddingEncodingFormat::Float
    );
    assert_eq!(request_parts.dimensions, None);
}

#[test]
fn should_preserve_list_order_for_an_array_of_inputs() {
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(
        r#"{
            "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
            "input": ["Two households, both alike in dignity.", "O Romeo, Romeo, wherefore art thou Romeo?"],
            "encoding_format": "base64",
            "dimensions": 256
        }"#,
    )
    .expect("array-input embeddings request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("array-input embeddings request should validate");

    assert_eq!(request_parts.inputs.len(), 2);
    assert_eq!(
        request_parts.inputs[1],
        "O Romeo, Romeo, wherefore art thou Romeo?"
    );
    assert_eq!(
        request_parts.encoding_format,
        OpenAiEmbeddingEncodingFormat::Base64
    );
    assert_eq!(request_parts.dimensions, Some(256));
}

#[test]
fn should_reject_an_empty_input_list() {
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(
        r#"{
            "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
            "input": []
        }"#,
    )
    .expect("empty-list embeddings request should deserialize");

    let validation_error = request
        .into_parts()
        .expect_err("an empty input list must fail before queue admission");

    assert!(validation_error.to_string().contains("at least one"));
}

#[test]
fn should_reject_zero_dimensions_before_queue_admission() {
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(
        r#"{
            "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
            "input": "O Romeo, Romeo, wherefore art thou Romeo?",
            "dimensions": 0
        }"#,
    )
    .expect("zero-dimensions embeddings request should deserialize");

    let validation_error = request
        .into_parts()
        .expect_err("dimensions 0 must fail before queue admission");

    assert!(validation_error.to_string().contains("positive"));
}

#[test]
fn should_reject_an_unsupported_encoding_format() {
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(
        r#"{
            "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
            "input": "hello",
            "encoding_format": "hex"
        }"#,
    )
    .expect("hex encoding_format should deserialize");

    let validation_error = request
        .into_parts()
        .expect_err("unsupported encoding_format must fail closed");

    assert!(validation_error.to_string().contains("hex"));
}

#[test]
fn should_reject_an_oversized_embedding_input() {
    let oversized_text = "x".repeat(8_193);
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(&format!(
        r#"{{"model":"mlx-community/nomicai-modernbert-embed-base-8bit","input":"{oversized_text}"}}"#
    ))
    .expect("oversized embedding input should deserialize");

    let validation_error = request
        .into_parts()
        .expect_err("an oversized single input must fail closed");

    assert!(validation_error.to_string().contains("8193"));
}

#[test]
fn should_reject_unknown_fields_like_max_length_for_now() {
    let request = serde_json::from_str::<OpenAiEmbeddingsRequest>(
        r#"{
            "model": "mlx-community/nomicai-modernbert-embed-base-8bit",
            "input": "hello",
            "max_length": 512
        }"#,
    )
    .expect("unknown-field request should deserialize");

    let validation_error = request
        .into_parts()
        .expect_err("unknown fields must fail closed on the embeddings boundary");

    assert!(validation_error.to_string().contains("max_length"));
}
