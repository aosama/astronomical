/// Validated persistence contract for projected image embeddings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentVisualEmbeddingModelContract {
    model_id: String,
    model_revision: String,
    projected_embedding_hidden_size: usize,
    maximum_visual_embedding_token_count: usize,
}

impl PersistentVisualEmbeddingModelContract {
    /// Binds a projected embedding layout to one exact model artifact.
    #[must_use]
    pub fn new(
        model_id: String,
        model_revision: String,
        projected_embedding_hidden_size: usize,
        maximum_visual_embedding_token_count: usize,
    ) -> Self {
        Self {
            model_id,
            model_revision,
            projected_embedding_hidden_size,
            maximum_visual_embedding_token_count,
        }
    }

    /// Returns the validated model ID bound to visual embeddings.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the validated model revision bound to visual embeddings.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// Returns the persisted projected visual embedding shape.
    #[must_use]
    pub const fn visual_embedding_shape(&self, visual_token_count: usize) -> [usize; 2] {
        [visual_token_count, self.projected_embedding_hidden_size]
    }

    /// Returns the projected visual embedding width consumed by the text model.
    #[must_use]
    pub const fn visual_embedding_hidden_size(&self) -> usize {
        self.projected_embedding_hidden_size
    }

    /// Returns the maximum visual rows accepted in one persisted image file.
    #[must_use]
    pub const fn maximum_visual_embedding_token_count(&self) -> usize {
        self.maximum_visual_embedding_token_count
    }
}
