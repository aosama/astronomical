//! Paged prompt-chunk boundaries for SSD-streaming execution.

mod configuration;

pub use configuration::Qwen3_5StreamingPromptProcessingChunkSizerError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5StreamingPromptProcessingChunkSizer {
    ssd_streaming_prompt_processing_chunk_size_tokens: usize,
}

impl Qwen3_5StreamingPromptProcessingChunkSizer {
    pub fn for_ssd_streaming_chunk_size_tokens(
        ssd_streaming_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, Qwen3_5StreamingPromptProcessingChunkSizerError> {
        let ssd_streaming_prompt_processing_chunk_size_tokens =
            configuration::prompt_processing_chunk_size_tokens_from_u32(
                ssd_streaming_prompt_processing_chunk_size_tokens,
            )?;
        Ok(Self {
            ssd_streaming_prompt_processing_chunk_size_tokens,
        })
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
    ) -> usize {
        self.next_prompt_processing_chunk_end_with_maximum_executable_capacity(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            usize::MAX,
        )
    }

    #[must_use]
    pub fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.ssd_streaming_prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub fn prompt_processing_operation_bound_tokens(&self) -> usize {
        self.ssd_streaming_prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end_with_maximum_executable_capacity(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        maximum_executable_chunk_size_tokens: usize,
    ) -> usize {
        let executable_chunk_size_tokens = self
            .ssd_streaming_prompt_processing_chunk_size_tokens
            .min(maximum_executable_chunk_size_tokens)
            .max(1);
        paged_chunk_end_after_folding_short_remainder(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            executable_chunk_size_tokens,
        )
    }

    #[must_use]
    pub fn remaining_prompt_processing_chunk_ranges(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
        maximum_executable_chunk_size_tokens: usize,
    ) -> Vec<(usize, usize)> {
        let mut remaining_chunk_ranges = Vec::new();
        let mut current_chunk_start = chunk_start_token_position;
        while current_chunk_start < final_prompt_end_token_position_exclusive {
            let current_chunk_end = self
                .next_prompt_processing_chunk_end_with_maximum_executable_capacity(
                    current_chunk_start,
                    final_prompt_end_token_position_exclusive,
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
