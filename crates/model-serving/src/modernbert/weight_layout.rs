//! Where a ModernBERT trunk's tensors live inside an artifact's weight file.
//!
//! The trunk serves more than one kind of artifact. An embedding artifact
//! stores the trunk under `model.` with quantized token embeddings; a decision
//! model stores it under `encoder.` beside its head and keeps the token
//! embeddings dense. The layout names those differences so the forward pass
//! itself stays the same for both.

/// How an artifact stores the token-embedding table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TokenEmbeddingStorage {
    /// Affine-quantized `weight`, `scales` and `biases`, dequantized per looked-up row.
    AffineQuantized,
    /// One dense `weight` table, looked up as stored.
    ///
    /// No artifact served today stores its table this way; decision-model
    /// checkpoints (issue #2032) do, and the trunk tests cover this path
    /// until that backend loads one.
    #[cfg_attr(not(test), expect(dead_code))]
    Dense,
}

/// The tensor-name prefix and embedding storage of one artifact's trunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModernBertWeightLayout {
    tensor_prefix: String,
    token_embedding_storage: TokenEmbeddingStorage,
}

impl ModernBertWeightLayout {
    pub(crate) fn new(
        tensor_prefix: impl Into<String>,
        token_embedding_storage: TokenEmbeddingStorage,
    ) -> Self {
        Self {
            tensor_prefix: tensor_prefix.into(),
            token_embedding_storage,
        }
    }

    /// The layout of the mlx-community ModernBERT embedding artifact.
    pub(crate) fn embedding_artifact() -> Self {
        Self::new("model", TokenEmbeddingStorage::AffineQuantized)
    }

    pub(crate) const fn token_embedding_storage(&self) -> TokenEmbeddingStorage {
        self.token_embedding_storage
    }

    /// The full name of the trunk tensor with `tensor_suffix`, such as `layers.0.attn.Wo`.
    pub(crate) fn tensor_name(&self, tensor_suffix: &str) -> String {
        format!("{}.{tensor_suffix}", self.tensor_prefix)
    }
}

#[cfg(test)]
#[path = "weight_layout_tests.rs"]
mod tests;
