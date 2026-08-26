//! Deterministic Laguna prompt-processing chunk boundaries.

use super::configuration::{
    LagunaPromptProcessingChunkSizerError, prompt_processing_chunk_size_tokens_from_u32,
};
/// Owns Laguna fixed chunk sizing without runtime learning or persistence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaPromptProcessingChunkSizer {
    fixed_prompt_processing_chunk_size_tokens: usize,
    ssd_streaming_prompt_processing_chunk_size_tokens: usize,
}

impl LagunaPromptProcessingChunkSizer {
    pub fn for_fixed_prompt_processing_chunk_size_tokens(
        fixed_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        Self::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            fixed_prompt_processing_chunk_size_tokens,
            fixed_prompt_processing_chunk_size_tokens,
        )
    }

    /// Paged experts use a separate chunk so complete-layer SSD reads can be amortized.
    pub fn for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
        fixed_prompt_processing_chunk_size_tokens: u32,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, LagunaPromptProcessingChunkSizerError> {
        let fixed_prompt_processing_chunk_size_tokens =
            prompt_processing_chunk_size_tokens_from_u32(
                fixed_prompt_processing_chunk_size_tokens,
            )?;
        let ssd_streaming_prompt_processing_chunk_size_tokens =
            prompt_processing_chunk_size_tokens_from_u32(
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
            )?;
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
        } else {
            self.fixed_prompt_processing_chunk_size_tokens
        };
        if sparse_experts_are_paged {
            return paged_chunk_end_after_folding_short_remainder(
                chunk_start_token_position,
                final_prompt_end_token_position_exclusive,
                configured_chunk_size_tokens,
            );
        }
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

    /// Resident admission must use this bound, not remaining prompt length.
    #[must_use]
    pub const fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.fixed_prompt_processing_chunk_size_tokens
    }
}

/// Each paged prefill forward streams every unseated complete MoE layer.
/// Fold a trailing stub smaller than one configured chunk into this forward so
/// that stub does not pay a second leftover-layer SSD sweep.
fn paged_chunk_end_after_folding_short_remainder(
    chunk_start_token_position: usize,
    final_prompt_end_token_position_exclusive: usize,
    executable_chunk_size_tokens: usize,
) -> usize {
    let remaining_prompt_token_count =
        final_prompt_end_token_position_exclusive.saturating_sub(chunk_start_token_position);
    if remaining_prompt_token_count <= executable_chunk_size_tokens {
        return chunk_start_token_position.saturating_add(remaining_prompt_token_count);
    }
    let remainder_after_full_chunk = remaining_prompt_token_count - executable_chunk_size_tokens;
    if remainder_after_full_chunk < executable_chunk_size_tokens {
        chunk_start_token_position.saturating_add(remaining_prompt_token_count)
    } else {
        chunk_start_token_position.saturating_add(executable_chunk_size_tokens)
    }
}
