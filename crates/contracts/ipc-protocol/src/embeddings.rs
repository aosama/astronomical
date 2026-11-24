//! Typed embedding requests, outputs, and worker-boundary validation.
//!
//! Embedding inference is one encoder forward pass with pooling, so the IPC
//! command carries text inputs only and the worker returns one vector per input.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::RequestId;

/// Bounded request-side input count that keeps one command inside one IPC frame.
pub const MAXIMUM_EMBEDDING_INPUT_COUNT: usize = 256;
const MAXIMUM_EMBEDDING_INPUT_BYTES: usize = 8_192;
const MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES: usize = 1_000_000;

/// One validated text-embedding request sent to the local inference worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingsCommand {
    pub request_id: RequestId,
    pub model: String,
    pub inputs: Vec<String>,
    pub encoding_format: EmbeddingEncodingFormat,
    /// Requested vector width when the caller truncates below the native width.
    pub dimensions: Option<u32>,
}

/// Encoding applied to each returned vector.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingEncodingFormat {
    Float,
    Base64,
}

/// Failure delivered after an embeddings request was admitted to the worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingsFailureReason {
    /// The worker independently rejected malformed structured embeddings input.
    InvalidRequest { reason: String },
    /// A fatal model-execution failure reported before the worker exits.
    FatalExecution { reason: String },
    /// Prompt tokens exceed the loaded encoder context.
    ContextLengthExceeded {
        actual_total_context_tokens: u32,
        maximum_context_tokens: u32,
    },
    /// A different embeddings request already owns the worker's bounded capacity.
    EngineBusy,
    /// Generated vectors could not be pooled or normalized into the declared contract.
    MalformedModelOutput,
}

impl EmbeddingsFailureReason {
    /// Preserves the worker's human-readable validation explanation for diagnostics and clients.
    #[must_use]
    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: reason.into(),
        }
    }
}

impl EmbeddingsCommand {
    /// Independently validates embeddings input after it crosses the worker trust boundary.
    pub fn validate(&self) -> Result<(), EmbeddingsValidationError> {
        if self.model.trim().is_empty() {
            return Err(EmbeddingsValidationError::EmptyModelId);
        }
        if self.inputs.is_empty() {
            return Err(EmbeddingsValidationError::EmptyInputs);
        }
        if self.inputs.len() > MAXIMUM_EMBEDDING_INPUT_COUNT {
            return Err(EmbeddingsValidationError::InputCountExceeded {
                actual_input_count: self.inputs.len(),
                maximum_input_count: MAXIMUM_EMBEDDING_INPUT_COUNT,
            });
        }
        let mut total_bytes = 0usize;
        for input_text in &self.inputs {
            let input_bytes = input_text.len();
            total_bytes += input_bytes;
            if input_bytes > MAXIMUM_EMBEDDING_INPUT_BYTES {
                return Err(EmbeddingsValidationError::InputTextTooLarge {
                    actual_input_bytes: input_bytes,
                    maximum_input_bytes: MAXIMUM_EMBEDDING_INPUT_BYTES,
                });
            }
        }
        if total_bytes > MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES {
            return Err(EmbeddingsValidationError::TotalInputBytesExceeded {
                actual_total_bytes: total_bytes,
                maximum_total_bytes: MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES,
            });
        }
        if self.dimensions == Some(0) {
            return Err(EmbeddingsValidationError::InvalidDimensions);
        }
        Ok(())
    }
}

/// Rejection reasons enforced independently at the worker IPC boundary.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EmbeddingsValidationError {
    #[error("model id must not be empty")]
    EmptyModelId,
    #[error("inputs must contain at least one text string")]
    EmptyInputs,
    #[error(
        "embedding input count is {actual_input_count}, outside the 1..={maximum_input_count} range"
    )]
    InputCountExceeded {
        actual_input_count: usize,
        maximum_input_count: usize,
    },
    #[error(
        "one embedding input has {actual_input_bytes} bytes, exceeding the {maximum_input_bytes}-byte limit"
    )]
    InputTextTooLarge {
        actual_input_bytes: usize,
        maximum_input_bytes: usize,
    },
    #[error(
        "aggregate embedding input has {actual_total_bytes} bytes, exceeding the {maximum_total_bytes}-byte limit"
    )]
    TotalInputBytesExceeded {
        actual_total_bytes: usize,
        maximum_total_bytes: usize,
    },
    #[error("dimensions must be a positive vector width")]
    InvalidDimensions,
}
