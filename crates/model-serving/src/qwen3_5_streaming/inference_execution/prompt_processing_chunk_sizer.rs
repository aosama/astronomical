//! Paged prompt-chunk boundaries for SSD-streaming execution.

#[cfg(test)]
mod tests;

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
        paged_chunk_end_after_folding_short_remainder(
            chunk_start_token_position,
            final_prompt_end_token_position_exclusive,
            self.ssd_streaming_prompt_processing_chunk_size_tokens,
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
