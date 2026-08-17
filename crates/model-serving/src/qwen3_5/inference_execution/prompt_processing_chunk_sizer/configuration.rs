//! Validates user-resolved capacities before constructing the Qwen adapter.

use thiserror::Error;

pub(super) fn prompt_processing_chunk_size_tokens_from_u32(
    prompt_processing_chunk_size_tokens: u32,
) -> Result<usize, Qwen3_5PromptProcessingChunkSizerError> {
    let prompt_processing_chunk_size_tokens = usize::try_from(prompt_processing_chunk_size_tokens)
        .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?;
    if prompt_processing_chunk_size_tokens == 0 {
        return Err(Qwen3_5PromptProcessingChunkSizerError::MustBePositive);
    }
    Ok(prompt_processing_chunk_size_tokens)
}

/// Invalid explicit Qwen3.5 prompt-processing chunk size.
#[derive(Clone, Debug, Error)]
pub enum Qwen3_5PromptProcessingChunkSizerError {
    #[error("prompt-processing chunk size exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prompt-processing chunk size must be positive")]
    MustBePositive,
    #[error("SSD-streaming prompt-processing chunk size must not exceed the resident chunk size")]
    SsdStreamingExceedsResidentChunkSize,
}
