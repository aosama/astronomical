use thiserror::Error;

pub(super) fn configured_candidate_prefill_chunck_tokens(
    optimizer_prefill_chunck_token_candidates: Vec<u32>,
    maximum_prefill_chunck_tokens: usize,
) -> Result<Vec<usize>, Qwen3_5PrefillChunckSizerError> {
    let candidate_prefill_chunck_tokens = optimizer_prefill_chunck_token_candidates
        .into_iter()
        .map(|candidate_prefill_chunck_tokens| candidate_prefill_chunck_tokens as usize)
        .filter(|candidate_prefill_chunck_tokens| {
            *candidate_prefill_chunck_tokens <= maximum_prefill_chunck_tokens
        })
        .collect::<Vec<_>>();
    if candidate_prefill_chunck_tokens.is_empty() {
        return Err(Qwen3_5PrefillChunckSizerError::OptimizerRejectedCandidateSet);
    }
    Ok(candidate_prefill_chunck_tokens)
}

pub(super) fn maximum_prefill_chunck_tokens_from_u32(
    maximum_prefill_chunck_tokens: u32,
) -> Result<usize, Qwen3_5PrefillChunckSizerError> {
    let prefill_chunck_tokens = usize::try_from(maximum_prefill_chunck_tokens)
        .map_err(|_| Qwen3_5PrefillChunckSizerError::ExceedsPlatformRange)?;
    if prefill_chunck_tokens == 0 {
        return Err(Qwen3_5PrefillChunckSizerError::MustBePositive);
    }
    Ok(prefill_chunck_tokens)
}

/// Invalid explicit Qwen3.5 prompt-processing prefill chunk size.
#[derive(Clone, Debug, Error)]
pub enum Qwen3_5PrefillChunckSizerError {
    #[error("prefill_chunck_tokens exceeds the platform integer range")]
    ExceedsPlatformRange,
    #[error("prefill_chunck_tokens must be positive")]
    MustBePositive,
    #[error("prefill_chunck_tokens optimizer rejected candidate set")]
    OptimizerRejectedCandidateSet,
}
