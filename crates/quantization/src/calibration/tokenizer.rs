//! The tokenizer seam the calibration loader encodes through.

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
        let _ = tokenizer_path;
        todo!("tokenizers::Tokenizer::from_file, mapped to TokenizerFailure")
    }
}

impl CalibrationTokenizer for HuggingFaceCalibrationTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError> {
        let _ = (&self.tokenizer, text);
        todo!("self.tokenizer.encode(text, false) and collect get_ids()")
    }
}
