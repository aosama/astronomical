//! Request-local monotonic decoding of generated token identifiers.
//!
//! Normal byte-pair tokenizers can decode one token or a small pending byte
//! sequence at a time. Retaining only an incomplete replacement-character suffix
//! avoids repeatedly decoding the complete generated prefix, reducing ordinary
//! streaming decode work from quadratic to linear in generated-token count.

use std::sync::Arc;

use tokenizers::Tokenizer;

use super::{Qwen3_5TokenIds, Qwen3_5TokenizerError};

const MAX_PENDING_TOKENS: usize = 4;
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

#[derive(Clone, Debug)]
pub struct Qwen3_5TokenDecoder {
    tokenizer: Arc<Tokenizer>,
    model_vocabulary_size: u32,
    token_ids: Qwen3_5TokenIds,
    generated_token_ids: Vec<u32>,
    emitted_text: String,
    pending_token_ids: Vec<u32>,
}

impl Qwen3_5TokenDecoder {
    pub(super) fn new(
        tokenizer: Arc<Tokenizer>,
        model_vocabulary_size: u32,
        token_ids: Qwen3_5TokenIds,
    ) -> Self {
        Self {
            tokenizer,
            model_vocabulary_size,
            token_ids,
            generated_token_ids: Vec::new(),
            emitted_text: String::new(),
            pending_token_ids: Vec::new(),
        }
    }

    /// Decodes one generated token while suppressing both certified stop tokens.
    pub fn push_token(
        &mut self,
        generated_token_id: u32,
    ) -> Result<Option<String>, Qwen3_5TokenizerError> {
        if generated_token_id == self.token_ids.end_of_text_token_id
            || generated_token_id == self.token_ids.im_end_token_id
        {
            return Ok(None);
        }
        if generated_token_id >= self.model_vocabulary_size
            || self.tokenizer.id_to_token(generated_token_id).is_none()
        {
            return Err(Qwen3_5TokenizerError::GeneratedTokenOutOfVocabulary {
                generated_token_id,
                model_vocabulary_size: self.model_vocabulary_size,
            });
        }
        self.generated_token_ids.push(generated_token_id);
        self.pending_token_ids.push(generated_token_id);

        let candidate_text = self
            .tokenizer
            .decode(&self.pending_token_ids, true)
            .map_err(|source| Qwen3_5TokenizerError::DecodeGeneratedTokens { source })?;
        if !candidate_text.ends_with(REPLACEMENT_CHARACTER) {
            self.emitted_text.push_str(&candidate_text);
            self.pending_token_ids.clear();
            return if candidate_text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(candidate_text))
            };
        }
        if self.pending_token_ids.len() >= MAX_PENDING_TOKENS {
            return self.flush_via_full_decode();
        }
        Ok(None)
    }

    /// Flushes a final incomplete byte-fallback sequence at request completion.
    pub fn finish(&mut self) -> Result<Option<String>, Qwen3_5TokenizerError> {
        if self.pending_token_ids.is_empty() {
            return Ok(None);
        }
        self.flush_via_full_decode()
    }

    fn flush_via_full_decode(&mut self) -> Result<Option<String>, Qwen3_5TokenizerError> {
        let full_decoded_text = self
            .tokenizer
            .decode(&self.generated_token_ids, true)
            .map_err(|source| Qwen3_5TokenizerError::DecodeGeneratedTokens { source })?;
        if !full_decoded_text.starts_with(&self.emitted_text) {
            return Err(Qwen3_5TokenizerError::DecodedTextRewrotePrefix);
        }
        let new_text_suffix = full_decoded_text[self.emitted_text.len()..].to_owned();
        self.emitted_text = full_decoded_text;
        self.pending_token_ids.clear();
        if new_text_suffix.is_empty() {
            return Ok(None);
        }
        Ok(Some(new_text_suffix))
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
