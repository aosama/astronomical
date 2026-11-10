//! The `config.json` edits that keep a converted artifact honest about
//! images: a tower that was written is registered so it can be rebuilt, and
//! a tower the config promises but the source never shipped is unpromised.
//!
//! Ported from OptiQ's `build_vision_sidecar` config step
//! (`optiq/vlm/sidecar.py`) and `_strip_vision_claims` (`optiq/sidecars.py`).

use std::path::Path;

use serde::Serialize;

use crate::error::QuantizationError;

use super::json_file::{read_json, write_pretty_json};

#[cfg(test)]
#[path = "multimodal_config_tests.rs"]
mod multimodal_config_tests;

/// The key OptiQ's front-ends read to find the tower sidecar.
pub const VISION_SIDECAR_CONFIG_KEY: &str = "optiq_vision";

/// The config fields that claim image or audio input. A text-only coder
/// fine-tuned from a vision base keeps them and drops the weights; left in
/// place they advertise images the artifact cannot see.
const MULTIMODAL_CLAIM_KEYS: [&str; 7] = [
    "vision_config",
    "audio_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "image_token_index",
];

/// The `optiq_vision` block.
#[derive(Serialize)]
struct VisionSidecarRegistration<'registration> {
    sidecar: &'registration str,
    dtype: &'static str,
    n_tensors: usize,
    base_model: &'registration str,
}

/// Records the written tower sidecar under `optiq_vision`.
pub fn register_vision_sidecar(
    config_path: &Path,
    sidecar_file_name: &str,
    sidecar_tensor_count: usize,
    base_model_name: &str,
) -> Result<(), QuantizationError> {
    let mut config_entries = read_config_entries(config_path)?;
    let registration = VisionSidecarRegistration {
        sidecar: sidecar_file_name,
        dtype: "bfloat16",
        n_tensors: sidecar_tensor_count,
        base_model: base_model_name,
    };
    let registration_value = match serde_json::to_value(registration) {
        Ok(registration_value) => registration_value,
        Err(serialize_error) => {
            return Err(config_unreadable(config_path, serialize_error.to_string()));
        }
    };
    config_entries.insert(VISION_SIDECAR_CONFIG_KEY.to_owned(), registration_value);
    write_pretty_json(&config_entries, config_path)
}

/// Removes every multimodal claim from a config whose source shipped no
/// tower, returning how many fields were removed.
pub fn strip_multimodal_claims(config_path: &Path) -> Result<usize, QuantizationError> {
    let mut config_entries = read_config_entries(config_path)?;
    let mut removed_field_count = 0;
    for claim_key in MULTIMODAL_CLAIM_KEYS {
        if config_entries.remove(claim_key).is_some() {
            removed_field_count += 1;
        }
    }
    if removed_field_count > 0 {
        write_pretty_json(&config_entries, config_path)?;
    }
    Ok(removed_field_count)
}

fn read_config_entries(
    config_path: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, QuantizationError> {
    match read_json(config_path) {
        Ok(config_entries) => Ok(config_entries),
        Err(description) => Err(config_unreadable(config_path, description)),
    }
}

fn config_unreadable(config_path: &Path, description: String) -> QuantizationError {
    QuantizationError::ModelConfigUnreadable {
        config_path: config_path.to_path_buf(),
        description,
    }
}
