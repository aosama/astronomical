//! A Laya checkpoint's tokenizer and the special tokens its layout uses.
//!
//! The special tokens are resolved from the tokenizer itself, by the names
//! `tokenizer_config.json` gives them. The encoder config also carries token
//! identifiers, but the checkpoints were trained with the tokenizer's, and
//! for the multilingual checkpoint the two disagree.

use astronomical_decision_contract::laya_sequence::LayaSpecialTokens;
use astronomical_decision_contract::{DecisionTokenEncoder, TokenEncodingError};
use thiserror::Error;
use tokenizers::Tokenizer;

/// The loaded tokenizer with its resolved special tokens.
pub(crate) struct LayaTokenizer {
    tokenizer: Tokenizer,
    pub(crate) special_tokens: LayaSpecialTokens,
    pub(crate) pad_token_id: u32,
}

/// Why a tokenizer cannot be used for Laya sequences.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum LayaTokenizerError {
    #[error("the tokenizer failed to load: {reason}")]
    Unloadable { reason: String },
    #[error("tokenizer_config.json is malformed: {reason}")]
    MalformedSettings { reason: String },
    #[error("the tokenizer has no {role} token {token_text:?}")]
    UnknownSpecialToken {
        role: &'static str,
        token_text: String,
    },
}

impl LayaTokenizer {
    pub(crate) fn from_artifact_bytes(
        tokenizer_json_bytes: &[u8],
        tokenizer_settings_bytes: &[u8],
    ) -> Result<Self, LayaTokenizerError> {
        let tokenizer = match Tokenizer::from_bytes(tokenizer_json_bytes) {
            Ok(tokenizer) => tokenizer,
            Err(load_error) => {
                return Err(LayaTokenizerError::Unloadable {
                    reason: load_error.to_string(),
                });
            }
        };
        let tokenizer_settings: serde_json::Value =
            match serde_json::from_slice(tokenizer_settings_bytes) {
                Ok(tokenizer_settings) => tokenizer_settings,
                Err(parse_error) => {
                    return Err(LayaTokenizerError::MalformedSettings {
                        reason: parse_error.to_string(),
                    });
                }
            };
        let mask_token_text = special_token_text(&tokenizer_settings, "mask");
        let special_tokens = LayaSpecialTokens {
            cls_token_id: special_token_id(&tokenizer, &tokenizer_settings, "cls")?,
            sep_token_id: special_token_id(&tokenizer, &tokenizer_settings, "sep")?,
            mask_token_id: special_token_id(&tokenizer, &tokenizer_settings, "mask")?,
            mask_token_text,
        };
        let pad_token_id = special_token_id(&tokenizer, &tokenizer_settings, "pad")?;
        Ok(Self {
            tokenizer,
            special_tokens,
            pad_token_id,
        })
    }
}

impl DecisionTokenEncoder for LayaTokenizer {
    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<u32>, TokenEncodingError> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => Ok(encoding.get_ids().to_vec()),
            Err(encode_error) => Err(TokenEncodingError {
                reason: encode_error.to_string(),
            }),
        }
    }
}

/// The text of the `<role>_token` setting, or the conventional `[ROLE]` when absent.
///
/// A setting is either the token text or an object whose `content` is the text.
fn special_token_text(tokenizer_settings: &serde_json::Value, role: &str) -> String {
    let setting = tokenizer_settings.get(format!("{role}_token"));
    let configured_text = match setting {
        Some(serde_json::Value::String(token_text)) => Some(token_text.clone()),
        Some(serde_json::Value::Object(token_fields)) => token_fields
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    };
    configured_text.unwrap_or_else(|| format!("[{}]", role.to_uppercase()))
}

fn special_token_id(
    tokenizer: &Tokenizer,
    tokenizer_settings: &serde_json::Value,
    role: &'static str,
) -> Result<u32, LayaTokenizerError> {
    let token_text = special_token_text(tokenizer_settings, role);
    match tokenizer.token_to_id(&token_text) {
        Some(token_id) => Ok(token_id),
        None => Err(LayaTokenizerError::UnknownSpecialToken { role, token_text }),
    }
}

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;
