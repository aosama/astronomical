//! Unit tests for trunk tensor naming.

use super::{ModernBertWeightLayout, TokenEmbeddingStorage};

#[test]
fn should_name_embedding_artifact_tensors_under_the_model_prefix() {
    let weight_layout = ModernBertWeightLayout::embedding_artifact();
    assert_eq!(
        weight_layout.tensor_name("layers.0.attn.Wo"),
        "model.layers.0.attn.Wo"
    );
    assert_eq!(
        weight_layout.token_embedding_storage(),
        TokenEmbeddingStorage::AffineQuantized
    );
}

#[test]
fn should_name_tensors_under_any_given_prefix() {
    let weight_layout = ModernBertWeightLayout::new("encoder", TokenEmbeddingStorage::Dense);
    assert_eq!(
        weight_layout.tensor_name("final_norm.weight"),
        "encoder.final_norm.weight"
    );
    assert_eq!(
        weight_layout.token_embedding_storage(),
        TokenEmbeddingStorage::Dense
    );
}
