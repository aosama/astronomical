//! Serialization contract for the embeddings response DTO, moved from the
//! integration lane beside its type (issue #1610).

use crate::{OpenAiEmbedding, OpenAiEmbeddingVector, OpenAiEmbeddingsResponse};

#[test]
fn should_serialize_an_openai_embeddings_list_with_float_rows() {
    let response = OpenAiEmbeddingsResponse::new(
        vec![
            OpenAiEmbedding::new(0, OpenAiEmbeddingVector::Float(vec![1.0, 0.0])),
            OpenAiEmbedding::new(1, OpenAiEmbeddingVector::Float(vec![0.0, 1.0])),
        ],
        "mlx-community/nomicai-modernbert-embed-base-8bit".to_owned(),
        crate::OpenAiTokenUsage::new(14, 0).expect("token accounting should be valid"),
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
