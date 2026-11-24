//! Embedding-model advertisement contracts, moved beside `OpenAiModel`
//! from the integration lane (issue #1610).

use crate::{OpenAiEmbeddingModelParts, OpenAiModel};

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
