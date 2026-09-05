//! Strict request validation for the OpenAI-compatible embeddings boundary.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// One request stays well inside the 32 MiB IPC frame budget after JSON
/// expansion, so a single embeddings command can never overflow the wire.
const MAXIMUM_EMBEDDING_INPUT_COUNT: usize = 256;
const MAXIMUM_EMBEDDING_INPUT_BYTES: usize = 8_192;
const MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES: usize = 1_000_000;

/// One strict request to the local OpenAI-compatible embeddings endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiEmbeddingsRequest {
    model: String,
    input: OpenAiEmbeddingInput,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<u32>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

/// One or many text inputs submitted in request order.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiEmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Validated embeddings request ready for supervisor translation.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiEmbeddingsRequestParts {
    pub model: String,
    pub inputs: Vec<String>,
    pub encoding_format: OpenAiEmbeddingEncodingFormat,
    pub dimensions: Option<u32>,
}

/// Encoding applied to each returned vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiEmbeddingEncodingFormat {
    Float,
    Base64,
}

impl OpenAiEmbeddingEncodingFormat {
    /// Canonical OpenAI wire value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Base64 => "base64",
        }
    }
}

/// Rejection reasons validated before queue admission.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum OpenAiEmbeddingsValidationError {
    #[error("model must not be empty")]
    EmptyModel,
    #[error("input must contain at least one text string")]
    EmptyInput,
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
    #[error("encoding_format '{encoding_format}' is unsupported; use float or base64")]
    UnsupportedEncodingFormat { encoding_format: String },
    #[error("dimensions must be a positive vector width")]
    InvalidDimensions,
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
}

impl OpenAiEmbeddingsRequest {
    /// Validates and consumes this public request into protocol-neutral parts.
    pub fn into_parts(
        self,
    ) -> Result<OpenAiEmbeddingsRequestParts, OpenAiEmbeddingsValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(OpenAiEmbeddingsValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        if self.model.trim().is_empty() {
            return Err(OpenAiEmbeddingsValidationError::EmptyModel);
        }
        let inputs = match self.input {
            OpenAiEmbeddingInput::Single(text) => vec![text],
            OpenAiEmbeddingInput::Multiple(texts) => texts,
        };
        if inputs.is_empty() {
            return Err(OpenAiEmbeddingsValidationError::EmptyInput);
        }
        if inputs.len() > MAXIMUM_EMBEDDING_INPUT_COUNT {
            return Err(OpenAiEmbeddingsValidationError::InputCountExceeded {
                actual_input_count: inputs.len(),
                maximum_input_count: MAXIMUM_EMBEDDING_INPUT_COUNT,
            });
        }
        let mut total_bytes = 0usize;
        for input_text in &inputs {
            let input_bytes = input_text.len();
            total_bytes += input_bytes;
            if input_bytes > MAXIMUM_EMBEDDING_INPUT_BYTES {
                return Err(OpenAiEmbeddingsValidationError::InputTextTooLarge {
                    actual_input_bytes: input_bytes,
                    maximum_input_bytes: MAXIMUM_EMBEDDING_INPUT_BYTES,
                });
            }
        }
        if total_bytes > MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES {
            return Err(OpenAiEmbeddingsValidationError::TotalInputBytesExceeded {
                actual_total_bytes: total_bytes,
                maximum_total_bytes: MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES,
            });
        }
        let encoding_format = match self.encoding_format.as_deref() {
            None | Some("float") => OpenAiEmbeddingEncodingFormat::Float,
            Some("base64") => OpenAiEmbeddingEncodingFormat::Base64,
            Some(unsupported_encoding_format) => {
                return Err(OpenAiEmbeddingsValidationError::UnsupportedEncodingFormat {
                    encoding_format: unsupported_encoding_format.to_owned(),
                });
            }
        };
        if self.dimensions == Some(0) {
            return Err(OpenAiEmbeddingsValidationError::InvalidDimensions);
        }
        Ok(OpenAiEmbeddingsRequestParts {
            model: self.model,
            inputs,
            encoding_format,
            dimensions: self.dimensions,
        })
    }
}
