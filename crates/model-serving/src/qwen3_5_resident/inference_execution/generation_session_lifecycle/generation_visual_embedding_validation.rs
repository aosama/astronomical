//! Precomputed visual-embedding shape validation for one generation session.

use crate::InferenceEngineError;
use crate::qwen3_5_resident::model::memory_admission;

/// Which precomputed visual-embedding rule failed. The split into two
/// validation calls preserves the established check order: the row-count
/// rules run before the model-owned hidden size is fetched, so the same
/// malformed request keeps producing the same failure text.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum VisualEmbeddingValidationError {
    EmptyVisualEmbeddings,
    ImagePadTokenCountMismatch,
    HiddenBufferSizeMismatch,
}

impl VisualEmbeddingValidationError {
    /// Maps each rule onto its established engine failure so error text and
    /// fatal-versus-invalid-request classification stay identical.
    pub(super) fn into_generation_failure(self) -> InferenceEngineError {
        match self {
            Self::EmptyVisualEmbeddings => {
                super::fatal_engine_error("image request has empty visual embeddings")
            }
            Self::ImagePadTokenCountMismatch => memory_admission::invalid_request_error(
                "image pad token count does not match visual embedding row count",
            ),
            Self::HiddenBufferSizeMismatch => super::fatal_engine_error(
                "visual embedding buffer does not match the expected hidden size",
            ),
        }
    }
}

/// Validates the row-count rules before any model-owned geometry is fetched.
pub(super) fn validate_precomputed_visual_embedding_row_counts(
    prompt_image_pad_token_count: usize,
    visual_embedding_row_count: usize,
    visual_embedding_values_len: usize,
) -> Result<(), VisualEmbeddingValidationError> {
    if visual_embedding_values_len == 0 || visual_embedding_row_count == 0 {
        return Err(VisualEmbeddingValidationError::EmptyVisualEmbeddings);
    }
    if prompt_image_pad_token_count != visual_embedding_row_count {
        return Err(VisualEmbeddingValidationError::ImagePadTokenCountMismatch);
    }
    Ok(())
}

/// Validates the buffer-size rule against the model-owned embedding width.
pub(super) fn validate_precomputed_visual_embedding_buffer_size(
    visual_embedding_row_count: usize,
    visual_embedding_values_len: usize,
    visual_embedding_hidden_size: usize,
) -> Result<(), VisualEmbeddingValidationError> {
    if visual_embedding_values_len
        != visual_embedding_row_count.saturating_mul(visual_embedding_hidden_size)
    {
        return Err(VisualEmbeddingValidationError::HiddenBufferSizeMismatch);
    }
    Ok(())
}

/// Counts the image-pad placeholder tokens in a prompt. Each placeholder
/// marks a position the vision encoder's output rows will later overwrite,
/// and the count must equal the total visual embedding row count.
pub(super) fn count_image_pad_placeholder_tokens(
    prompt_token_ids: &[u32],
    image_pad_token_id: u32,
) -> usize {
    let mut image_pad_placeholder_token_count = 0;
    for token_id in prompt_token_ids {
        if *token_id == image_pad_token_id {
            image_pad_placeholder_token_count += 1;
        }
    }
    image_pad_placeholder_token_count
}

#[cfg(test)]
#[path = "generation_visual_embedding_validation_tests.rs"]
mod generation_visual_embedding_validation_tests;
