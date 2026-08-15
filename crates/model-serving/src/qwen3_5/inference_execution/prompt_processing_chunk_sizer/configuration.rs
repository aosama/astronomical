//! Validates user-resolved capacities before constructing the Qwen adapter.

use thiserror::Error;

pub(super) fn configured_candidate_chunk_size_tokens(
    configured_candidate_chunk_size_token_counts: Vec<u32>,
    maximum_prompt_processing_chunk_size_tokens: usize,
) -> Result<Vec<usize>, Qwen3_5PromptProcessingChunkSizerError> {
    let candidate_chunk_size_tokens = configured_candidate_chunk_size_token_counts
        .into_iter()
        .map(|candidate_chunk_size_tokens| candidate_chunk_size_tokens as usize)
        .filter(|candidate_chunk_size_tokens| {
            *candidate_chunk_size_tokens <= maximum_prompt_processing_chunk_size_tokens
        })
        .collect::<Vec<_>>();
    if candidate_chunk_size_tokens.is_empty() {
        return Err(Qwen3_5PromptProcessingChunkSizerError::OptimizerRejectedCandidateSet);
    }
    Ok(candidate_chunk_size_tokens)
}

pub(super) fn maximum_prompt_processing_chunk_size_tokens_from_u32(
    maximum_prompt_processing_chunk_size_tokens: u32,
) -> Result<usize, Qwen3_5PromptProcessingChunkSizerError> {
    let maximum_prompt_processing_chunk_size_tokens =
        usize::try_from(maximum_prompt_processing_chunk_size_tokens)
            .map_err(|_| Qwen3_5PromptProcessingChunkSizerError::ExceedsPlatformRange)?;
    if maximum_prompt_processing_chunk_size_tokens == 0 {
        return Err(Qwen3_5PromptProcessingChunkSizerError::MustBePositive);
    }
    Ok(maximum_prompt_processing_chunk_size_tokens)
}

/// Invalid explicit Qwen3.5 prompt-processing chunk size.
#[derive(Clone, Debug, Error)]
pub enum Qwen3_5PromptProcessingChunkSizerError {
    #[error("prompt-processing chunk size exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prompt-processing chunk size must be positive")]
    MustBePositive,
    #[error("prompt-processing chunk size optimizer rejected the candidate set")]
    OptimizerRejectedCandidateSet,
}
