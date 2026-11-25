//! Validates user-resolved capacities before constructing the chunk sizer.

use thiserror::Error;

pub(super) fn prompt_processing_chunk_size_tokens_from_u32(
    prompt_processing_chunk_size_tokens: u32,
) -> Result<usize, PromptProcessingChunkSizerError> {
    let prompt_processing_chunk_size_tokens = usize::try_from(prompt_processing_chunk_size_tokens)
        .map_err(|_| PromptProcessingChunkSizerError::ExceedsPlatformRange)?;
    if prompt_processing_chunk_size_tokens == 0 {
        return Err(PromptProcessingChunkSizerError::MustBePositive);
    }
    Ok(prompt_processing_chunk_size_tokens)
}

/// Invalid explicit prompt-processing chunk size.
#[derive(Clone, Debug, Error)]
pub enum PromptProcessingChunkSizerError {
    #[error("prompt-processing chunk size exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prompt-processing chunk size must be positive")]
    MustBePositive,
}
