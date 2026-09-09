//! Hugging Face fast tokenizer owned by the K2 Horizon MoVA family.

use std::sync::Arc;

use thiserror::Error;
use tokenizers::Tokenizer;

use crate::k2_horizon_mova::configuration::K2HorizonMoVAConfig;

const THINK_CLOSE: &str = "</ifm|think>";
const THINK_FAST_CLOSE: &str = "</ifm|think_fast>";
const THINK_FASTER_CLOSE: &str = "</ifm|think_faster>";

/// Validated tokenizer used for prompt encoding and generated-token decode.
#[derive(Clone, Debug)]
pub struct K2HorizonMoVATokenizer {
    tokenizer: Arc<Tokenizer>,
    eos_token_ids: Vec<u32>,
    vocab_size: u32,
    think_close_token_ids: Vec<u32>,
    natural_reasoning_end_token_ids: Vec<u32>,
}

#[derive(Debug, Error)]
pub enum K2HorizonMoVATokenizerError {
    #[error("failed to load the K2 Horizon MoVA tokenizer")]
    LoadTokenizer,
    #[error("failed to encode the K2 Horizon MoVA prompt")]
    EncodePrompt,
    #[error("failed to decode a K2 Horizon MoVA token")]
    DecodeToken,
}

impl K2HorizonMoVATokenizer {
    pub fn from_json_bytes(
        tokenizer_bytes: &[u8],
        config: &K2HorizonMoVAConfig,
    ) -> Result<Self, K2HorizonMoVATokenizerError> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|_| K2HorizonMoVATokenizerError::LoadTokenizer)?;
        let think_close_token_ids = encode_special_token_ids(&tokenizer, THINK_CLOSE);
        let mut natural_reasoning_end_token_ids = think_close_token_ids.clone();
        for close_marker in [THINK_FAST_CLOSE, THINK_FASTER_CLOSE] {
            if let Some(end_token_id) = encode_special_token_ids(&tokenizer, close_marker).last() {
                if !natural_reasoning_end_token_ids.contains(end_token_id) {
                    natural_reasoning_end_token_ids.push(*end_token_id);
                }
            }
        }
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            eos_token_ids: config.eos_token_ids().to_vec(),
            vocab_size: config.vocab_size(),
            think_close_token_ids,
            natural_reasoning_end_token_ids,
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, K2HorizonMoVATokenizerError> {
        let encoding = self
            .tokenizer
            .encode_fast(prompt, false)
            .map_err(|_| K2HorizonMoVATokenizerError::EncodePrompt)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode_token(&self, token_id: u32) -> Result<String, K2HorizonMoVATokenizerError> {
        self.tokenizer
            .decode(&[token_id], false)
            .map_err(|_| K2HorizonMoVATokenizerError::DecodeToken)
    }

    #[must_use]
    pub fn is_end_of_sequence_token(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    #[must_use]
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    #[must_use]
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Token ids that close the IFM think channel, used as a forced transition.
    #[must_use]
    pub fn think_close_token_ids(&self) -> &[u32] {
        &self.think_close_token_ids
    }

    /// Token ids that end reasoning even when the model closes a faster think lane.
    #[must_use]
    pub fn natural_reasoning_end_token_ids(&self) -> &[u32] {
        &self.natural_reasoning_end_token_ids
    }

    #[must_use]
    pub fn incremental_decoder(&self) -> K2HorizonMoVATokenDecoder {
        K2HorizonMoVATokenDecoder {
            tokenizer: Arc::clone(&self.tokenizer),
            pending_token_ids: Vec::new(),
            emitted_characters: 0,
            eos_token_ids: self.eos_token_ids.clone(),
        }
    }
}

fn encode_special_token_ids(tokenizer: &Tokenizer, special_token: &str) -> Vec<u32> {
    tokenizer
        .encode_fast(special_token, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .unwrap_or_default()
}

/// Accumulates generated token IDs and emits newly decoded characters.
#[derive(Debug)]
pub struct K2HorizonMoVATokenDecoder {
    tokenizer: Arc<Tokenizer>,
    pending_token_ids: Vec<u32>,
    emitted_characters: usize,
    eos_token_ids: Vec<u32>,
}

impl K2HorizonMoVATokenDecoder {
    pub fn push_token(
        &mut self,
        token_id: u32,
    ) -> Result<Option<String>, K2HorizonMoVATokenizerError> {
        if self.eos_token_ids.contains(&token_id) {
            return Ok(None);
        }
        self.pending_token_ids.push(token_id);
        let decoded = self
            .tokenizer
            .decode(&self.pending_token_ids, false)
            .map_err(|_| K2HorizonMoVATokenizerError::DecodeToken)?;
        if decoded.chars().count() <= self.emitted_characters {
            return Ok(None);
        }
        let newly_decoded = decoded
            .chars()
            .skip(self.emitted_characters)
            .collect::<String>();
        self.emitted_characters = decoded.chars().count();
        if newly_decoded.is_empty() {
            Ok(None)
        } else {
            Ok(Some(newly_decoded))
        }
    }
}
