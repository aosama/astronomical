//! The tokenizer seam the calibration loader encodes through: OptiQ's
//! `tokenizer.encode(text, add_special_tokens=False)` call in
//! `optiq/calibration/datasets.py`.

use std::path::Path;

use crate::error::QuantizationError;

use super::chat_template::ChatTemplate;
use super::sample::CalibrationSample;

/// What the loader needs from a tokenizer. The production implementation
/// wraps the `tokenizers` crate; tests use a whitespace fake.
pub trait CalibrationTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, QuantizationError>;

    /// A chat sample as the model's own prompt text, or `None` when the
    /// tokenizer has no chat template that accepts the conversation. The
    /// default has no template, so every chat sample takes the role-prefixed
    /// fallback.
    fn render_chat(&self, _sample: &CalibrationSample) -> Option<String> {
        None
    }
}

/// A Hugging Face `tokenizer.json` with the model's chat template, when it
/// ships one.
pub struct HuggingFaceCalibrationTokenizer {
    tokenizer: tokenizers::Tokenizer,
    chat_template: Option<ChatTemplate>,
}

impl HuggingFaceCalibrationTokenizer {
    /// Loads a `tokenizer.json` on its own, with no chat template.
    pub fn from_tokenizer_file(tokenizer_path: &Path) -> Result<Self, QuantizationError> {
        match tokenizers::Tokenizer::from_file(tokenizer_path) {
            Ok(tokenizer) => Ok(Self {
                tokenizer,
                chat_template: None,
            }),
            Err(load_error) => Err(QuantizationError::TokenizerFailure {
                description: format!("could not load {}: {load_error}", tokenizer_path.display()),
            }),
        }
    }

    /// Loads the tokenizer and the chat template of a model directory, the
    /// pair OptiQ calibrates with.
    pub fn from_model_directory(model_directory: &Path) -> Result<Self, QuantizationError> {
        let mut tokenizer = Self::from_tokenizer_file(&model_directory.join("tokenizer.json"))?;
        tokenizer.chat_template = ChatTemplate::from_model_directory(model_directory);
        Ok(tokenizer)
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

    /// OptiQ's order: the template with the sample's tools, then without
    /// them (some templates reject tools they do not know), then nothing,
    /// which sends the sample to the role-prefixed fallback so one strict
    /// template cannot sink the whole calibration set.
    fn render_chat(&self, sample: &CalibrationSample) -> Option<String> {
        let chat_template = self.chat_template.as_ref()?;
        let messages = sample.messages.as_ref()?;
        if let Ok(rendered_text) = chat_template.render(messages, sample.tools.as_ref()) {
            return Some(rendered_text);
        }
        chat_template.render(messages, None).ok()
    }
}
