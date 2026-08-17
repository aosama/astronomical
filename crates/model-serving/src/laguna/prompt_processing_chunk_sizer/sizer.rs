//! Deterministic Laguna prompt-processing chunk boundaries.

use super::configuration::{
    LagunaPromptProcessingChunkSizerError, prompt_processing_chunk_size_tokens_from_u32,
};
/// Owns Laguna fixed chunk sizing without runtime learning or persistence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaPromptProcessingChunkSizer {
    fixed_prompt_processing_chunk_size_tokens: usize,
    ssd_streaming_prompt_processing_chunk_size_tokens: Option<usize>,
}

impl LagunaPromptProcessingChunkSizer {
    pub fn for_fixed_prompt_processing_chunk_size_tokens(
        fixed_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        Self::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            fixed_prompt_processing_chunk_size_tokens,
            None,
        )
    }

    /// Allows paged experts to use less prompt work than fully resident experts.
    pub fn for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
        fixed_prompt_processing_chunk_size_tokens: u32,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        let fixed_prompt_processing_chunk_size_tokens =
            prompt_processing_chunk_size_tokens_from_u32(
                fixed_prompt_processing_chunk_size_tokens,
            )?;
        let ssd_streaming_prompt_processing_chunk_size_tokens =
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .map(prompt_processing_chunk_size_tokens_from_u32)
                .transpose()?;
        if ssd_streaming_prompt_processing_chunk_size_tokens.is_some_and(
            |ssd_streaming_chunk_size_tokens| {
                ssd_streaming_chunk_size_tokens > fixed_prompt_processing_chunk_size_tokens
            },
        ) {
            return Err(
                LagunaPromptProcessingChunkSizerError::SsdStreamingExceedsResidentChunkSize,
            );
        }
        Ok(Self {
            fixed_prompt_processing_chunk_size_tokens,
            ssd_streaming_prompt_processing_chunk_size_tokens,
        })
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        sparse_experts_are_paged: bool,
    ) -> usize {
        let configured_chunk_size_tokens = if sparse_experts_are_paged {
            self.ssd_streaming_prompt_processing_chunk_size_tokens
                .unwrap_or(self.fixed_prompt_processing_chunk_size_tokens)
        } else {
            self.fixed_prompt_processing_chunk_size_tokens
        };
        chunk_start_token_position
            .saturating_add(configured_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
    }

    /// Selects one deterministic fallback after memory pressure rejects a fixed attempt.
    #[must_use]
    pub const fn next_smaller_executable_chunk_size_tokens(
        attempted_chunk_size_tokens: usize,
    ) -> Option<usize> {
        if attempted_chunk_size_tokens <= 1 {
            return None;
        }
        Some(attempted_chunk_size_tokens / 2)
    }
}
