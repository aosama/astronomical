//! Deterministic Qwen3.5 prompt-processing chunk boundaries.

mod configuration;

use configuration::prompt_processing_chunk_size_tokens_from_u32;

pub use configuration::Qwen3_5PromptProcessingChunkSizerError;

/// Owns fixed Qwen3.5 chunk sizing and deterministic memory-capacity reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5PromptProcessingChunkSizer {
    fixed_prompt_processing_chunk_size_tokens: usize,
    ssd_streaming_prompt_processing_chunk_size_tokens: usize,
}

impl Qwen3_5PromptProcessingChunkSizer {
    pub fn for_fixed_prompt_processing_chunk_size_tokens(
        fixed_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
        Self::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            fixed_prompt_processing_chunk_size_tokens,
            fixed_prompt_processing_chunk_size_tokens,
        )
    }

    /// Paged experts use a separate chunk so complete-layer SSD reads can be amortized.
    pub fn for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
        fixed_prompt_processing_chunk_size_tokens: u32,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, Qwen3_5PromptProcessingChunkSizerError> {
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
    ) -> usize {
        self.next_prompt_processing_chunk_end_for_expert_residency(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            false,
        )
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end_for_expert_residency(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        sparse_experts_are_paged: bool,
    ) -> usize {
        self.next_prompt_processing_chunk_end_with_maximum_executable_capacity(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            sparse_experts_are_paged,
            usize::MAX,
        )
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end_with_maximum_executable_capacity(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        sparse_experts_are_paged: bool,
        maximum_executable_chunk_size_tokens: usize,
    ) -> usize {
        let configured_chunk_size_tokens = if sparse_experts_are_paged {
            self.ssd_streaming_prompt_processing_chunk_size_tokens
        } else {
            self.fixed_prompt_processing_chunk_size_tokens
        };
        let executable_chunk_size_tokens = configured_chunk_size_tokens
            .min(maximum_executable_chunk_size_tokens)
            .max(1);
        if sparse_experts_are_paged {
            return paged_chunk_end_after_folding_short_remainder(
                chunk_start_token_position,
                final_prompt_end_token_position_exclusive,
                executable_chunk_size_tokens,
            );
        }
        chunk_start_token_position
            .saturating_add(executable_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
    }

    /// Plans every remaining activation chunk without durable-cache clamping.
    #[must_use]
    pub fn remaining_prompt_processing_chunk_ranges(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        sparse_experts_are_paged: bool,
        maximum_executable_chunk_size_tokens: usize,
    ) -> Vec<(usize, usize)> {
        let mut remaining_chunk_ranges = Vec::new();
        let mut current_chunk_start = chunk_start_token_position;
        while current_chunk_start < final_prompt_end_token_position_exclusive {
            let current_chunk_end = self
                .next_prompt_processing_chunk_end_with_maximum_executable_capacity(
                    current_chunk_start,
                    final_prompt_end_token_position_exclusive,
                    sparse_experts_are_paged,
                    maximum_executable_chunk_size_tokens,
                );
            if current_chunk_end <= current_chunk_start {
                break;
            }
            remaining_chunk_ranges.push((current_chunk_start, current_chunk_end));
            current_chunk_start = current_chunk_end;
        }
        remaining_chunk_ranges
    }

    /// Halving provides bounded deterministic recovery without runtime learning state.
    #[must_use]
    pub const fn next_smaller_executable_chunk_size_tokens(
        attempted_chunk_size_tokens: usize,
    ) -> Option<usize> {
        let smaller_chunk_size_tokens = attempted_chunk_size_tokens / 2;
        if smaller_chunk_size_tokens == 0 {
            None
        } else {
            Some(smaller_chunk_size_tokens)
        }
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
