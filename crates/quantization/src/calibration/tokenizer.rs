//! The tokenizer seam the calibration loader encodes through: OptiQ's
//! `tokenizer.encode(text, add_special_tokens=False)` call in
//! `optiq/calibration/datasets.py`.

use std::path::Path;

use crate::error::QuantizationError;

/// What the loader needs from a tokenizer. The production implementation
/// wraps the `tokenizers` crate; tests use a whitespace fake.
pub trait CalibrationTokenizer {
    /// Encodes text to token ids without adding special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError>;
}

/// The `tokenizers` crate adapter over a `tokenizer.json`.
pub struct HuggingFaceCalibrationTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceCalibrationTokenizer {
    pub fn from_tokenizer_file(tokenizer_path: &Path) -> Result<Self, QuantizationError> {
        match tokenizers::Tokenizer::from_file(tokenizer_path) {
            Ok(tokenizer) => Ok(Self { tokenizer }),
            Err(load_error) => Err(QuantizationError::TokenizerFailure {
                description: format!("could not load {}: {load_error}", tokenizer_path.display()),
            }),
        }
    }
}

impl CalibrationTokenizer for HuggingFaceCalibrationTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => Ok(encoding.get_ids().to_vec()),
            Err(encode_error) => Err(QuantizationError::TokenizerFailure {
                description: encode_error.to_string(),
            }),
        }
    }
}
