//! `generation_config.json`: the publisher's recommended sampling, carried
//! into the converted artifact.
//!
//! Qwen3.5 and Qwen3.6 recommend sampling values in their model cards but
//! ship no `generation_config.json`, so a conversion that only copies files
//! produces an artifact with no sampling defaults at all. OptiQ writes the
//! documented recipe for those families and never overrides a value the
//! source did publish.
//!
//! Ported from OptiQ's `_write_recommended_generation_config`,
//! `_family_from_hf_path` and `_FAMILY_RECOMMENDED_SAMPLING`
//! (`optiq/backends/mlx_backend.py`).

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::error::QuantizationError;

use super::json_file::{read_json, write_pretty_json};

#[cfg(test)]
#[path = "generation_config_tests.rs"]
mod generation_config_tests;

/// The file transformers and OptiQ's server read sampling defaults from.
pub const GENERATION_CONFIG_FILE_NAME: &str = "generation_config.json";

/// The family's "non-thinking instruct, general tasks" recipe from its
/// model card, or `None` for a family with no recorded recipe. The family
/// is read from the last path segment of the base model name, as OptiQ does.
#[must_use]
pub fn recommended_sampling_for(base_model_name: &str) -> Option<Map<String, Value>> {
    let last_segment = base_model_name
        .rsplit('/')
        .next()
        .unwrap_or(base_model_name)
        .to_lowercase();
    let recipe = if last_segment.starts_with("qwen3.5") || last_segment.starts_with("qwen3.6") {
        json!({
            "do_sample": true,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
        })
    } else if last_segment.starts_with("minicpm5") {
        json!({ "do_sample": true, "temperature": 0.7, "top_p": 0.95 })
    } else {
        return None;
    };
    recipe.as_object().cloned()
}

/// Fills the artifact's `generation_config.json` with the family recipe
/// under whatever the source published (already copied into the artifact),
/// returning whether the file was written. A family without a recipe, or a
/// file that already holds every recommended key, is left untouched.
pub fn write_recommended_generation_config(
    artifact_directory: &Path,
    base_model_name: &str,
) -> Result<bool, QuantizationError> {
    let Some(mut merged_entries) = recommended_sampling_for(base_model_name) else {
        return Ok(false);
    };
    let generation_config_path = artifact_directory.join(GENERATION_CONFIG_FILE_NAME);
    let published_entries: Map<String, Value> = if generation_config_path.is_file() {
        read_json(&generation_config_path).unwrap_or_default()
    } else {
        Map::new()
    };
    for (published_key, published_value) in &published_entries {
        merged_entries.insert(published_key.clone(), published_value.clone());
    }
    if merged_entries == published_entries {
        return Ok(false);
    }
    write_pretty_json(&merged_entries, &generation_config_path)?;
    Ok(true)
}
