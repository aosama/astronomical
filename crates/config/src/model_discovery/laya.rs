//! Shallow discovery rules for executable Laya decision-model artifacts.
//!
//! A Laya artifact is the MLX conversion of a `convaiinnovations/laya`
//! checkpoint: one `config.json` whose `model_type` is `laya`, with the
//! encoder's own config nested under `encoder`, beside a tokenizer pair and
//! one weight file. A decision model has no context window or vector width;
//! what discovery derives is how many tokens one question may occupy.
//!
//! One repository can bundle several checkpoints: the published conversion
//! keeps the English checkpoint at its root and others in subfolders such as
//! `multilingual/`. Each bundled checkpoint is a model of its own, so
//! discovery reports them all.

use std::fs;
use std::path::{Path, PathBuf};

/// The sequence length a checkpoint reads when its config does not say.
const DEFAULT_MAXIMUM_TOKENS_PER_QUESTION: u64 = 512;
/// The most bundled checkpoints read from one repository. A handful are
/// published; the bound stops a directory of thousands of folders from
/// turning one discovery pass into thousands of config reads.
const MAXIMUM_BUNDLED_CHECKPOINTS: usize = 16;
const MAXIMUM_CONFIG_JSON_BYTES: u64 = 1_048_576;

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct LayaDiscoveredModelMetadata {
    pub maximum_tokens_per_question: u32,
    pub model_size_bytes: u64,
}

/// Recognizes the MLX-converted Laya decision model type.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("laya"))
}

/// Validates shallow Laya completeness and derives public discovery metadata.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
    config_value: &serde_json::Value,
) -> Option<LayaDiscoveredModelMetadata> {
    for required_file_name in [
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        if !model_directory.join(required_file_name).is_file() {
            return None;
        }
    }
    if !config_value.get("encoder")?.is_object() {
        return None;
    }
    let quantization_bits = config_value.get("quantization")?.get("bits")?.as_u64()?;
    if quantization_bits != 8 {
        // Only the reviewed affine 8-bit profile is executable; other widths
        // stay undiscoverable until an engine validates them.
        return None;
    }
    let maximum_tokens_per_question = match config_value.get("max_len") {
        None => DEFAULT_MAXIMUM_TOKENS_PER_QUESTION,
        Some(configured_limit) => configured_limit.as_u64()?,
    };
    let maximum_tokens_per_question = u32::try_from(maximum_tokens_per_question).ok()?;
    if maximum_tokens_per_question == 0 {
        return None;
    }
    let model_size_bytes = fs::metadata(model_directory.join("model.safetensors"))
        .ok()?
        .len();
    Some(LayaDiscoveredModelMetadata {
        maximum_tokens_per_question,
        model_size_bytes,
    })
}

/// One checkpoint bundled in a subfolder of a Laya repository.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BundledLayaCheckpoint {
    /// The subfolder's name, such as `multilingual`.
    pub checkpoint_name: String,
    pub checkpoint_directory: PathBuf,
}

/// The Laya checkpoints bundled in immediate subfolders of `repository_directory`,
/// in name order so their discovery order never depends on the file system.
pub(super) fn bundled_checkpoints(repository_directory: &Path) -> Vec<BundledLayaCheckpoint> {
    let Ok(directory_entries) = fs::read_dir(repository_directory) else {
        return Vec::new();
    };
    let mut bundled_checkpoints = Vec::new();
    for directory_entry in directory_entries.flatten() {
        let checkpoint_directory = directory_entry.path();
        let Some(checkpoint_name) = checkpoint_directory
            .file_name()
            .and_then(|directory_name| directory_name.to_str())
        else {
            continue;
        };
        if checkpoint_name.starts_with('.') || !holds_a_laya_config(&checkpoint_directory) {
            continue;
        }
        bundled_checkpoints.push(BundledLayaCheckpoint {
            checkpoint_name: checkpoint_name.to_owned(),
            checkpoint_directory,
        });
    }
    bundled_checkpoints.sort_by(|left, right| left.checkpoint_name.cmp(&right.checkpoint_name));
    bundled_checkpoints.truncate(MAXIMUM_BUNDLED_CHECKPOINTS);
    bundled_checkpoints
}

/// The model identity of a bundled checkpoint: the repository's identity
/// followed by the subfolder name, so `laya-MLX-8bit` bundles
/// `laya-MLX-8bit-multilingual`.
pub(super) fn bundled_checkpoint_model_id(
    repository_model_id: &str,
    checkpoint_name: &str,
) -> String {
    format!("{repository_model_id}-{checkpoint_name}")
}

fn holds_a_laya_config(checkpoint_directory: &Path) -> bool {
    let config_path = checkpoint_directory.join("config.json");
    let Ok(config_metadata) = fs::metadata(&config_path) else {
        return false;
    };
    if !config_metadata.is_file() || config_metadata.len() > MAXIMUM_CONFIG_JSON_BYTES {
        return false;
    }
    let Ok(config_bytes) = fs::read(&config_path) else {
        return false;
    };
    let Ok(config_value) = serde_json::from_slice::<serde_json::Value>(&config_bytes) else {
        return false;
    };
    recognizes_model_type(
        config_value
            .get("model_type")
            .and_then(serde_json::Value::as_str),
    )
}
