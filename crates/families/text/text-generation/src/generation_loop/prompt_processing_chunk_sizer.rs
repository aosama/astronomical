//! Prompt-chunk boundaries for every text engine.
//!
//! The resident and streaming engines cut a prompt the same way: chunks of
//! the configured size, with a shorter final chunk. They differ only in what
//! the configured size is tuned for, which the worker decides.

#[cfg(test)]
mod tests;

mod configuration;

pub use configuration::PromptProcessingChunkSizerError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptProcessingChunkSizer {
    prompt_processing_chunk_size_tokens: usize,
}

impl PromptProcessingChunkSizer {
    pub fn for_chunk_size_tokens(
        prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, PromptProcessingChunkSizerError> {
        let prompt_processing_chunk_size_tokens =
            configuration::prompt_processing_chunk_size_tokens_from_u32(
                prompt_processing_chunk_size_tokens,
            )?;
        Ok(Self {
            prompt_processing_chunk_size_tokens,
        })
    }

    /// Returns the end of the next chunk, never more than the configured size.
    ///
    /// The configured size is a hard cap because memory planning reserves
    /// activation memory for exactly
    /// [`Self::maximum_prompt_processing_chunk_size_tokens`]. An earlier
    /// streaming version folded a short remainder into the current chunk to
    /// save one SSD streaming pass, which produced forwards of up to twice the
    /// configured size against a one-chunk reserve and exhausted GPU memory
    /// mid-request.
    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
    ) -> usize {
        chunk_start_token_position
            .saturating_add(self.prompt_processing_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
    }

    #[must_use]
    pub fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub fn prompt_processing_operation_bound_tokens(&self) -> usize {
        self.prompt_processing_chunk_size_tokens
    }
}
