//! Clef's tokenizer: Qwen3.5's, used exactly as `encode_record` uses it.
//!
//! Every piece of the sequence is tokenized on its own without added
//! beginning or end tokens. Chat markers written in the text, such as
//! `<|im_start|>`, are read as the special tokens they name, as the
//! reference's tokenizer reads them.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use tokenizers::Tokenizer;

/// A Qwen3.5 tokenizer is about 20 MB; this leaves room for a larger vocabulary.
const MAXIMUM_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;
/// The chat markers the prompt is built from; a tokenizer without them is not Clef's.
const CLEF_CHAT_MARKERS: [&str; 2] = ["<|im_start|>", "<|im_end|>"];

pub(crate) struct ClefTokenizer {
    tokenizer: Tokenizer,
}

impl ClefTokenizer {
    pub(crate) fn read(model_directory: &Path) -> Result<Self, DecisionFailure> {
        let tokenizer_path = model_directory.join("tokenizer.json");
        let tokenizer_size = std::fs::metadata(&tokenizer_path)
            .map_err(|_| unusable_tokenizer("tokenizer.json is missing"))?
            .len();
        if tokenizer_size > MAXIMUM_TOKENIZER_BYTES {
            return Err(unusable_tokenizer("tokenizer.json is too large"));
        }
        let tokenizer_bytes = std::fs::read(&tokenizer_path)
            .map_err(|_| unusable_tokenizer("tokenizer.json cannot be read"))?;
        Self::from_json_bytes(&tokenizer_bytes)
    }

    pub(crate) fn from_json_bytes(tokenizer_bytes: &[u8]) -> Result<Self, DecisionFailure> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|load_error| {
            unusable_tokenizer(&format!("it failed to load: {load_error}"))
        })?;
        for chat_marker in CLEF_CHAT_MARKERS {
            if tokenizer.token_to_id(chat_marker).is_none() {
                return Err(unusable_tokenizer(&format!(
                    "it has no {chat_marker} token"
                )));
            }
        }
        Ok(Self { tokenizer })
    }

    /// `_tokens(tokenizer, text)`.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionFailure> {
        let encoding = self.tokenizer.encode(text, false).map_err(|encode_error| {
            DecisionFailure::fatal_execution(&format!("Clef's tokenizer failed: {encode_error}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }
}

fn unusable_tokenizer(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("Clef's tokenizer cannot be used: {reason}"))
}
