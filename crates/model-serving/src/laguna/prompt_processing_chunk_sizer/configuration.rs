//! Validates user-resolved capacities before constructing the Laguna adapter.

use thiserror::Error;

pub(super) fn configured_candidate_chunk_size_tokens(
    configured_candidate_chunk_size_token_counts: Vec<u32>,
    maximum_prompt_processing_chunk_size_tokens: usize,
) -> Result<Vec<usize>, LagunaPromptProcessingChunkSizerError> {
    let candidate_chunk_size_tokens = configured_candidate_chunk_size_token_counts
        .into_iter()
        .map(|candidate_chunk_size_tokens| candidate_chunk_size_tokens as usize)
        .filter(|candidate_chunk_size_tokens| {
            *candidate_chunk_size_tokens <= maximum_prompt_processing_chunk_size_tokens
        })
        .collect::<Vec<_>>();
    if candidate_chunk_size_tokens.is_empty() {
        return Err(LagunaPromptProcessingChunkSizerError::OptimizerRejectedCandidateSet);
    }
    Ok(candidate_chunk_size_tokens)
}

pub(super) fn maximum_prompt_processing_chunk_size_tokens_from_u32(
    maximum_prompt_processing_chunk_size_tokens: u32,
) -> Result<usize, LagunaPromptProcessingChunkSizerError> {
    let maximum_prompt_processing_chunk_size_tokens =
        usize::try_from(maximum_prompt_processing_chunk_size_tokens)
            .map_err(|_| LagunaPromptProcessingChunkSizerError::ExceedsPlatformRange)?;
    if maximum_prompt_processing_chunk_size_tokens == 0 {
        return Err(LagunaPromptProcessingChunkSizerError::MustBePositive);
    }
    Ok(maximum_prompt_processing_chunk_size_tokens)
}

/// Invalid explicit Laguna prompt-processing chunk size.
#[derive(Clone, Debug, Error)]
pub enum LagunaPromptProcessingChunkSizerError {
    #[error("prompt-processing chunk size exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prompt-processing chunk size must be positive")]
    MustBePositive,
    #[error("prompt-processing chunk size optimizer rejected the candidate set")]
    OptimizerRejectedCandidateSet,
}
