use astronomical_rest_contract::{
    OpenAiEmbedding, OpenAiEmbeddingEncodingFormat, OpenAiEmbeddingModelParts,
    OpenAiEmbeddingVector, OpenAiEmbeddingsRequest, OpenAiEmbeddingsResponse, OpenAiModel,
};

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

#[test]
fn should_serialize_an_openai_embeddings_list_with_float_rows() {
    let response = OpenAiEmbeddingsResponse::new(
        vec![
            OpenAiEmbedding::new(0, OpenAiEmbeddingVector::Float(vec![1.0, 0.0])),
            OpenAiEmbedding::new(1, OpenAiEmbeddingVector::Float(vec![0.0, 1.0])),
        ],
        "mlx-community/nomicai-modernbert-embed-base-8bit".to_owned(),
        astronomical_rest_contract::OpenAiTokenUsage::new(14, 0)
            .expect("token accounting should be valid"),
    );

    let serialized =
        serde_json::to_string(&response).expect("the embeddings response should serialize");
    assert!(
        serialized.contains(r#""object":"list""#),
        "embedding list must declare object list: {serialized}"
    );
    assert!(
        serialized.contains(r#""object":"embedding""#),
        "each row must declare object embedding: {serialized}"
    );
    assert!(serialized.contains(r#""index":1"#));
    assert!(serialized.contains(r#""total_tokens":14"#));
}

#[test]
fn should_advertise_an_embedding_model_with_only_the_embeddings_endpoint() {
    let model = OpenAiModel::from_embedding_parts(OpenAiEmbeddingModelParts {
        model_id: "nomicai-modernbert-embed-base-8bit".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        vector_width: 768,
        max_input_tokens: 8_192,
    })
    .expect("a valid embedding model should serialize");

    let serialized = serde_json::to_string(&model).expect("the embedding model should serialize");
    assert!(serialized.contains(r#""supported_endpoints":["/v1/embeddings"]"#));
    assert!(serialized.contains(r#""output_modalities":["embedding"]"#));
    assert!(serialized.contains(r#""supports_streaming":false"#));
}

#[test]
fn should_reject_a_zero_vector_width_embedding_model() {
    let validation_error = OpenAiModel::from_embedding_parts(OpenAiEmbeddingModelParts {
        model_id: "nomicai-modernbert-embed-base-8bit".to_owned(),
        created: 1_784_231_803,
        owned_by: "astronomical".to_owned(),
        vector_width: 0,
        max_input_tokens: 8_192,
    })
    .expect_err("a zero vector width must fail advertisement");

    assert!(validation_error.to_string().contains("vector width"));
}
