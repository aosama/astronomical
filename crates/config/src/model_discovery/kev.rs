//! Shallow discovery rules for Kev decision-model folders.
//!
//! A Kev folder is the output of `cargo xtask convert-kev`: a Qwen3.5
//! backbone with Kev's adapter folded in, kept in the base model's own
//! layout, so its `config.json` reads as a Qwen3.5 model. What marks it as a
//! decision model instead is `kev_decision.json`, beside the pointer head in
//! `kev_head.safetensors`. Discovery therefore checks for that marker before
//! it reads the config's model type.

use std::fs;
use std::path::Path;

use super::bounded_artifact_file;

const KEV_SETTINGS_FILE_NAME: &str = "kev_decision.json";
const KEV_HEAD_FILE_NAME: &str = "kev_head.safetensors";
const MAXIMUM_SETTINGS_BYTES: u64 = 64 * 1024;
/// The longest row Kev serves: a state of up to 65,536 tokens, its state
/// token included, and 8,192 more for the question (the reference's
/// `SERVE_MAX_STATE` and `SERVE_MAX_BRANCH`).
const KEV_MAXIMUM_TOKENS_PER_QUESTION: u32 = 65_536 + 8_192;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct KevDiscoveredModelMetadata {
    pub maximum_tokens_per_question: u32,
    pub model_size_bytes: u64,
}

/// Whether `model_directory` is marked as a Kev decision model.
pub(super) fn classifies_directory(model_directory: &Path) -> bool {
    let Ok(settings_bytes) = bounded_artifact_file::read_bounded_nonempty_file(
        &model_directory.join(KEV_SETTINGS_FILE_NAME),
        MAXIMUM_SETTINGS_BYTES,
    ) else {
        return false;
    };
    let Ok(settings) = serde_json::from_slice::<serde_json::Value>(&settings_bytes) else {
        return false;
    };
    settings
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        == Some("kev")
}

/// Validates shallow Kev completeness and derives public discovery metadata.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
) -> Option<KevDiscoveredModelMetadata> {
    for required_file_name in [KEV_HEAD_FILE_NAME, "tokenizer.json", "config.json"] {
        if !model_directory.join(required_file_name).is_file() {
            return None;
        }
    }
    let mut model_size_bytes = 0;
    for directory_entry in fs::read_dir(model_directory).ok()?.flatten() {
        if directory_entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            model_size_bytes += directory_entry.metadata().ok()?.len();
        }
    }
    // The head alone is a few megabytes; a folder without backbone weights
    // is an incomplete download, not a model.
    let head_bytes = fs::metadata(model_directory.join(KEV_HEAD_FILE_NAME))
        .ok()?
        .len();
    if model_size_bytes <= head_bytes {
        return None;
    }
    Some(KevDiscoveredModelMetadata {
        maximum_tokens_per_question: KEV_MAXIMUM_TOKENS_PER_QUESTION,
        model_size_bytes,
    })
}
