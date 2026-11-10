//! `README.md`: the minimal model card that marks the artifact as an MLX
//! text-generation model.
//!
//! The Hugging Face hub reads a repository's kind from the card's front
//! matter, so an artifact uploaded without one is listed with no library and
//! no task. mlx-lm writes this card on every conversion; the body is left
//! empty for the publisher.
//!
//! Ported from mlx-lm's `create_model_card` (`mlx_lm/utils.py`).

use std::path::Path;

use crate::error::QuantizationError;

#[cfg(test)]
#[path = "model_card_tests.rs"]
mod model_card_tests;

pub const MODEL_CARD_FILE_NAME: &str = "README.md";

/// The front matter mlx-lm writes for a source it has no hub card for.
const MODEL_CARD_TEXT: &str =
    "---\nlanguage: en\nlibrary_name: mlx\npipeline_tag: text-generation\ntags:\n- mlx\n---\n";

/// Writes the card unless the artifact already has a `README.md`, returning
/// whether it was written.
pub fn write_model_card(artifact_directory: &Path) -> Result<bool, QuantizationError> {
    let model_card_path = artifact_directory.join(MODEL_CARD_FILE_NAME);
    if model_card_path.exists() {
        return Ok(false);
    }
    match std::fs::write(&model_card_path, MODEL_CARD_TEXT) {
        Ok(()) => Ok(true),
        Err(write_error) => Err(QuantizationError::ArtifactWriteFailure {
            output_directory: artifact_directory.to_path_buf(),
            description: format!("the model card was not written: {write_error}"),
        }),
    }
}
