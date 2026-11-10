//! Shallow discovery rules for Clef decision-model folders.
//!
//! A Clef folder (`mlx-community/clef-flash-8bit`) is a Qwen3.5 checkpoint in
//! its own layout, so its `config.json` reads as a chat model; what marks it
//! as a decision model is its joint schema head, `joint_head.safetensors`,
//! described by `joint_head_config.json`. Discovery checks for that pair
//! before it reads the config's model type.

use std::fs;
use std::path::Path;

use super::bounded_artifact_file;
use super::marked_decision_folder::{MarkedDecisionMetadata, safetensors_total_bytes};

const CLEF_HEAD_SETTINGS_FILE_NAME: &str = "joint_head_config.json";
const CLEF_HEAD_WEIGHTS_FILE_NAME: &str = "joint_head.safetensors";
const MAXIMUM_SETTINGS_BYTES: u64 = 64 * 1024;
/// The whole sequence Clef reads: state, every question and every option
/// (`encode_record`'s `max_length`). A longer state is cut to fit.
const CLEF_MAXIMUM_TOKENS_PER_REQUEST: u32 = 16_384;

/// Whether `model_directory` holds a Clef joint head.
pub(super) fn classifies_directory(model_directory: &Path) -> bool {
    if !model_directory.join(CLEF_HEAD_WEIGHTS_FILE_NAME).is_file() {
        return false;
    }
    let Ok(settings_bytes) = bounded_artifact_file::read_bounded_nonempty_file(
        &model_directory.join(CLEF_HEAD_SETTINGS_FILE_NAME),
        MAXIMUM_SETTINGS_BYTES,
    ) else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&settings_bytes)
        .ok()
        .and_then(|settings| {
            settings
                .get("hidden_size")
                .and_then(serde_json::Value::as_u64)
        })
        .is_some()
}

/// Validates shallow Clef completeness and derives public discovery metadata.
pub(super) fn discover_model_metadata(model_directory: &Path) -> Option<MarkedDecisionMetadata> {
    for required_file_name in [
        CLEF_HEAD_WEIGHTS_FILE_NAME,
        "tokenizer.json",
        "config.json",
        "model.safetensors.index.json",
    ] {
        if !model_directory.join(required_file_name).is_file() {
            return None;
        }
    }
    let model_size_bytes = safetensors_total_bytes(model_directory)?;
    // The head alone is a quarter of a gigabyte; a folder without backbone
    // weights is an incomplete download, not a model.
    let head_bytes = fs::metadata(model_directory.join(CLEF_HEAD_WEIGHTS_FILE_NAME))
        .ok()?
        .len();
    if model_size_bytes <= head_bytes {
        return None;
    }
    Some(MarkedDecisionMetadata {
        maximum_tokens_per_question: CLEF_MAXIMUM_TOKENS_PER_REQUEST,
        model_size_bytes,
        settings_file_name: CLEF_HEAD_SETTINGS_FILE_NAME,
    })
}
