//! Fixed prompt-chunk boundaries for complete-residency execution.

#[cfg(test)]
mod tests;

mod configuration;

pub use configuration::Qwen3_5ResidentPromptProcessingChunkSizerError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5ResidentPromptProcessingChunkSizer {
    fixed_prompt_processing_chunk_size_tokens: usize,
}

impl Qwen3_5ResidentPromptProcessingChunkSizer {
    pub fn for_fixed_prompt_processing_chunk_size_tokens(
        fixed_prompt_processing_chunk_size_tokens: u32,
    ) -> Result<Self, Qwen3_5ResidentPromptProcessingChunkSizerError> {
        let fixed_prompt_processing_chunk_size_tokens =
            configuration::prompt_processing_chunk_size_tokens_from_u32(
                fixed_prompt_processing_chunk_size_tokens,
            )?;
        Ok(Self {
            fixed_prompt_processing_chunk_size_tokens,
        })
    }

    #[must_use]
    pub fn next_prompt_processing_chunk_end(
        &self,
        chunk_start_token_position: usize,
        final_prompt_end_token_position_exclusive: usize,
    ) -> usize {
        chunk_start_token_position
            .saturating_add(self.fixed_prompt_processing_chunk_size_tokens)
            .min(final_prompt_end_token_position_exclusive)
    }

    #[must_use]
    pub fn maximum_prompt_processing_chunk_size_tokens(&self) -> usize {
        self.fixed_prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub fn prompt_processing_operation_bound_tokens(&self) -> usize {
        self.fixed_prompt_processing_chunk_size_tokens
    }
}
