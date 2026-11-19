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

    /// Returns the end of the next chunk, never more than the configured size.
    ///
    /// The configured size is a hard cap because the expert residency plan
    /// reserves activation memory for exactly
    /// [`Self::maximum_prompt_processing_chunk_size_tokens`]. An earlier
    /// version folded a short remainder into the current chunk to save one
    /// SSD streaming pass, which produced forwards of up to twice the
    /// configured size against a one-chunk reserve and exhausted GPU memory
    /// mid-request.
    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
    ) -> usize {
        chunk_start_token_position
            .saturating_add(self.ssd_streaming_prompt_processing_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
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
