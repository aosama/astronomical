//! Validates user-resolved capacities before constructing the Laguna adapter.

use thiserror::Error;

pub(super) fn prompt_processing_chunk_size_tokens_from_u32(
    prompt_processing_chunk_size_tokens: u32,
) -> Result<usize, LagunaPromptProcessingChunkSizerError> {
    let prompt_processing_chunk_size_tokens = usize::try_from(prompt_processing_chunk_size_tokens)
        .map_err(|_| LagunaPromptProcessingChunkSizerError::ExceedsPlatformRange)?;
    if prompt_processing_chunk_size_tokens == 0 {
        return Err(LagunaPromptProcessingChunkSizerError::MustBePositive);
    }
    Ok(prompt_processing_chunk_size_tokens)
}

/// Invalid explicit Laguna prompt-processing chunk size.
#[derive(Clone, Debug, Error)]
pub enum LagunaPromptProcessingChunkSizerError {
    #[error("prompt-processing chunk size exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prompt-processing chunk size must be positive")]
    MustBePositive,
}
