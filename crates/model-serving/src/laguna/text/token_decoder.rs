use std::sync::Arc;

use tokenizers::Tokenizer;

use super::LagunaTokenizerError;

const MAXIMUM_PENDING_BYTE_FALLBACK_TOKENS: usize = 4;
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

/// Request-local monotonic decoder with stop-token suppression and vocabulary checks.
#[derive(Clone, Debug)]
pub struct LagunaTokenDecoder {
    tokenizer: Arc<Tokenizer>,
    model_vocabulary_size: u32,
    end_token_ids: Arc<[u32]>,
    generated_token_ids: Vec<u32>,
    pending_token_ids: Vec<u32>,
    emitted_text: String,
}

impl LagunaTokenDecoder {
    pub(super) fn new(
        tokenizer: Arc<Tokenizer>,
        model_vocabulary_size: u32,
        end_token_ids: Arc<[u32]>,
    ) -> Self {
        Self {
            tokenizer,
            model_vocabulary_size,
            end_token_ids,
            generated_token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            emitted_text: String::new(),
        }
    }

    /// Decodes one generated token after suppressing every canonical end token.
    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Option<String>, LagunaTokenizerError> {
        if self
            .end_token_ids
            .binary_search(&generated_token_id)
            .is_ok()
        {
            return Ok(None);
        }
        if generated_token_id >= self.model_vocabulary_size
            || self.tokenizer.id_to_token(generated_token_id).is_none()
        {
            return Err(LagunaTokenizerError::GeneratedTokenOutOfVocabulary {
                generated_token_id,
                model_vocabulary_size: self.model_vocabulary_size,
            });
        }
        self.generated_token_ids.push(generated_token_id);
        self.pending_token_ids.push(generated_token_id);
        let candidate_text = self
            .tokenizer
            .decode(&self.pending_token_ids, true)
            .map_err(|source| LagunaTokenizerError::DecodeGeneratedTokens { source })?;
        if !candidate_text.ends_with(REPLACEMENT_CHARACTER) {
            self.emitted_text.push_str(&candidate_text);
            self.pending_token_ids.clear();
            return Ok((!candidate_text.is_empty()).then_some(candidate_text));
        }
        if self.pending_token_ids.len() >= MAXIMUM_PENDING_BYTE_FALLBACK_TOKENS {
            return self.flush_complete_prefix();
        }
        Ok(None)
    }

    /// Flushes a final byte-fallback sequence without permitting prefix rewrites.
    pub fn finish(&mut self) -> Result<Option<String>, LagunaTokenizerError> {
        if self.pending_token_ids.is_empty() {
            return Ok(None);
        }
        self.flush_complete_prefix()
    }

    fn flush_complete_prefix(&mut self) -> Result<Option<String>, LagunaTokenizerError> {
        let complete_text = self
            .tokenizer
            .decode(&self.generated_token_ids, true)
            .map_err(|source| LagunaTokenizerError::DecodeGeneratedTokens { source })?;
        if !complete_text.starts_with(&self.emitted_text) {
            return Err(LagunaTokenizerError::DecodedTextRewrotePrefix);
        }
        let new_text = complete_text[self.emitted_text.len()..].to_owned();
        self.emitted_text = complete_text;
        self.pending_token_ids.clear();
        Ok((!new_text.is_empty()).then_some(new_text))
    }

    pub(crate) fn generated_token_ids_for_diagnostics(&self) -> &[u32] {
        &self.generated_token_ids
    }

    pub(crate) fn pending_token_ids_for_diagnostics(&self) -> &[u32] {
        &self.pending_token_ids
    }

    pub(crate) fn emitted_text_for_diagnostics(&self) -> &str {
        &self.emitted_text
    }
}
