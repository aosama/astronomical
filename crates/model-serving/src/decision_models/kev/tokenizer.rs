//! Kev's tokenizer and the five control tokens its layout is built from.
//!
//! Kev reuses five tokens the Qwen3.5 vocabulary already has, so the
//! backbone needs no new embeddings. Caller text is escaped before it is
//! encoded, so no state or question can contain one of them.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use astronomical_decision_contract::kev_rendering::kev_escape_control_tokens;
use tokenizers::Tokenizer;

/// The state token, question token, option opening and closing tokens, and
/// decision token, in that order, as `kev/model.py`'s `SPECIAL` lists them.
const KEV_CONTROL_TOKEN_TEXTS: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];
/// A Qwen3.5 tokenizer is about 20 MB; this leaves room for a larger vocabulary.
const MAXIMUM_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;

/// The identifiers of Kev's control tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KevControlTokens {
    pub(crate) state_token_id: u32,
    pub(crate) question_token_id: u32,
    pub(crate) option_opening_token_id: u32,
    pub(crate) option_closing_token_id: u32,
    pub(crate) decision_token_id: u32,
}

pub(crate) struct KevTokenizer {
    tokenizer: Tokenizer,
    pub(crate) control_tokens: KevControlTokens,
}

impl KevTokenizer {
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
        let mut control_token_ids = [0_u32; 5];
        for (control_token_id, control_token_text) in
            control_token_ids.iter_mut().zip(KEV_CONTROL_TOKEN_TEXTS)
        {
            *control_token_id = tokenizer.token_to_id(control_token_text).ok_or_else(|| {
                unusable_tokenizer(&format!("it has no {control_token_text} token"))
            })?;
        }
        let [
            state_token_id,
            question_token_id,
            option_opening_token_id,
            option_closing_token_id,
            decision_token_id,
        ] = control_token_ids;
        Ok(Self {
            tokenizer,
            control_tokens: KevControlTokens {
                state_token_id,
                question_token_id,
                option_opening_token_id,
                option_closing_token_id,
                decision_token_id,
            },
        })
    }

    /// `user_tokens(tok, text)`: caller text, escaped, without added
    /// beginning or end tokens.
    pub(crate) fn encode_caller_text(
        &self,
        caller_text: &str,
    ) -> Result<Vec<u32>, DecisionFailure> {
        let encoding = self
            .tokenizer
            .encode(kev_escape_control_tokens(caller_text), false)
            .map_err(|encode_error| {
                DecisionFailure::fatal_execution(&format!("Kev's tokenizer failed: {encode_error}"))
            })?;
        Ok(encoding.get_ids().to_vec())
    }
}

fn unusable_tokenizer(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("Kev's tokenizer cannot be used: {reason}"))
}

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;
