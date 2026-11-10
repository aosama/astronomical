//! The one thing prompt rendering needs from a tokenizer.
//!
//! Rendering a question means cutting option texts and the state to token
//! budgets, which needs token counts. Depending on a tokenizer library here
//! would pull its build into the daemon, the command-line tool and every
//! other crate that only validates requests, so rendering takes this trait
//! and the model-serving crate implements it over the real tokenizer.

use thiserror::Error;

/// Turns text into the token identifiers a model family's tokenizer assigns.
pub trait DecisionTokenEncoder {
    /// Encodes `text` with no special tokens added before or after it.
    ///
    /// Renderers place every special token themselves, so an encoder that
    /// added its own would put them in the middle of a sequence.
    fn encode_without_special_tokens(&self, text: &str) -> Result<Vec<u32>, TokenEncodingError>;
}

/// A tokenizer could not encode a piece of text.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("tokenization failed: {reason}")]
pub struct TokenEncodingError {
    pub reason: String,
}
