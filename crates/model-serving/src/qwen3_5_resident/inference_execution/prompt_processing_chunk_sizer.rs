//! Deterministic Qwen3.5 prompt-processing chunk boundaries.

use crate::AdaptiveRamGrowthExecutionProfile;

mod configuration;

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
            configuration::prompt_processing_chunk_size_tokens_from_u32(
                fixed_prompt_processing_chunk_size_tokens,
            )?;
        let ssd_streaming_prompt_processing_chunk_size_tokens =
            configuration::prompt_processing_chunk_size_tokens_from_u32(
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
            AdaptiveRamGrowthExecutionProfile::Resident,
        )
    }

    /// The largest token count any single prompt-processing forward can span.
    ///
    /// Paged experts stream with their own chunk size, so the operation bound
    /// for memory planning is the larger of the two configured sizes (issue
    /// #644: activation reserves are operation-scoped and need the operation
    /// bound, not the prompt length).
    ///
    /// This is the SAFE upper bound for a question that has no residency mode,
    /// such as sizing a buffer that must hold either mode's chunk. It is the
    /// WRONG input for resolving one mode's activation promise, because a
    /// reserve sized for the larger chunk charges the smaller chunk's forwards
    /// for work they never do; use
    /// [`Self::prompt_processing_operation_bound_tokens`] there.
    #[must_use]
    pub fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.fixed_prompt_processing_chunk_size_tokens
            .max(self.ssd_streaming_prompt_processing_chunk_size_tokens)
    }

    /// The token bound that resolves one execution profile's activation promise.
    ///
    /// Each mode's forwards record activation evidence at that mode's own
    /// operation scope, so the bound must match the mode that will actually
    /// run. Resolving the resident mode's promise at the paged scope — the
    /// larger of the two chunks — inflated the reserve by the chunk ratio and
    /// rejected every later request (measured 2026-10-10).
    ///
    /// The profile argument is the decision admission already made for this
    /// request, not a prediction: evidence recorded under one profile is not
    /// transferable to the other, because paging itself changes the observed
    /// activation transient.
    #[must_use]
    pub fn prompt_processing_operation_bound_tokens(
        &self,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
    ) -> usize {
        self.configured_prompt_processing_chunk_size_tokens(execution_profile)
    }

    #[must_use]
    fn configured_prompt_processing_chunk_size_tokens(
        &self,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
    ) -> usize {
        match execution_profile {
            AdaptiveRamGrowthExecutionProfile::Resident => {
                self.fixed_prompt_processing_chunk_size_tokens
            }
            AdaptiveRamGrowthExecutionProfile::Paged => {
                self.ssd_streaming_prompt_processing_chunk_size_tokens
            }
        }
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end_for_expert_residency(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
    ) -> usize {
        self.next_prompt_processing_chunk_end_with_maximum_executable_capacity(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            execution_profile,
            usize::MAX,
        )
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end_with_maximum_executable_capacity(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
        maximum_executable_chunk_size_tokens: usize,
    ) -> usize {
        let configured_chunk_size_tokens =
            self.configured_prompt_processing_chunk_size_tokens(execution_profile);
        let executable_chunk_size_tokens = configured_chunk_size_tokens
            .min(maximum_executable_chunk_size_tokens)
            .max(1);
        if execution_profile == AdaptiveRamGrowthExecutionProfile::Paged {
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
        execution_profile: AdaptiveRamGrowthExecutionProfile,
        maximum_executable_chunk_size_tokens: usize,
    ) -> Vec<(usize, usize)> {
        let mut remaining_chunk_ranges = Vec::new();
        let mut current_chunk_start = chunk_start_token_position;
        while current_chunk_start < final_prompt_end_token_position_exclusive {
            let current_chunk_end = self
                .next_prompt_processing_chunk_end_with_maximum_executable_capacity(
                    current_chunk_start,
                    final_prompt_end_token_position_exclusive,
                    execution_profile,
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
