use std::sync::Arc;

use tokenizers::Tokenizer;

use super::{LagunaTextArtifactDescriptor, LagunaTokenDecoder, LagunaTokenizerError};

/// Laguna-owned view of the tokenizer compiled once during artifact normalization.
#[derive(Clone, Debug)]
pub struct LagunaTokenizer {
    tokenizer: Arc<Tokenizer>,
    model_vocabulary_size: u32,
    end_token_ids: Arc<[u32]>,
}

impl LagunaTokenizer {
    /// Shares the tokenizer already validated by text artifact normalization.
    pub fn from_descriptor(
        descriptor: &LagunaTextArtifactDescriptor,
    ) -> Result<Self, LagunaTokenizerError> {
        Ok(Self {
            tokenizer: Arc::clone(descriptor.tokenizer()),
            model_vocabulary_size: descriptor.model_vocabulary_size(),
            end_token_ids: Arc::from(descriptor.end_token_ids()),
        })
    }

    /// Encodes one artifact-template prompt without tokenizer-side template execution.
    pub fn encode_prompt(&self, rendered_prompt: &str) -> Result<Vec<u32>, LagunaTokenizerError> {
        self.tokenizer
            .encode(rendered_prompt, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|source| LagunaTokenizerError::EncodePrompt { source })
    }

    /// Creates independent monotonic decode state for one generation request.
    #[must_use]
    pub fn incremental_decoder(&self) -> LagunaTokenDecoder {
        LagunaTokenDecoder::new(
            Arc::clone(&self.tokenizer),
            self.model_vocabulary_size,
            Arc::clone(&self.end_token_ids),
        )
    }
}
