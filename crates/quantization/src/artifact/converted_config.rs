//! The `config.json` clean-up a conversion through mlx-lm performs, so an
//! artifact converted here carries the same document.
//!
//! Ported from mlx-lm's `save_config` (`mlx_lm/utils.py`) and the Qwen3.5
//! `TextModelArgs.__post_init__` key rename (`mlx_lm/models/qwen3_5.py`).

use std::path::Path;

use serde_json::{Map, Value};

use crate::error::QuantizationError;

use super::json_file::{read_json, write_pretty_json};

#[cfg(test)]
#[path = "converted_config_tests.rs"]
mod converted_config_tests;

/// A key transformers writes to remember where a config was loaded from; it
/// names a path on the machine that converted the model.
const LOADED_FROM_KEY: &str = "_name_or_path";

/// Normalizes the artifact's `config.json` in place: drops the loaded-from
/// path, and for the Qwen3.5 family spells the rotary-embedding kind as
/// `type`. Original releases write `rope_type`; mlx-lm renames it when it
/// parses the config and saves the renamed document, so every artifact it
/// converts, OptiQ's included, carries `type`.
pub fn normalize_converted_config(config_path: &Path) -> Result<(), QuantizationError> {
    let mut config_entries: Map<String, Value> = match read_json(config_path) {
        Ok(config_entries) => config_entries,
        Err(description) => {
            return Err(QuantizationError::ModelConfigUnreadable {
                config_path: config_path.to_path_buf(),
                description,
            });
        }
    };
    config_entries.remove(LOADED_FROM_KEY);
    if is_qwen3_5_family(&config_entries) {
        let text_entries = match config_entries.get_mut("text_config") {
            Some(Value::Object(text_entries)) => text_entries,
            _ => &mut config_entries,
        };
        rename_rope_kind(text_entries);
    }
    write_pretty_json(&config_entries, config_path)
}

fn is_qwen3_5_family(config_entries: &Map<String, Value>) -> bool {
    let model_type = config_entries
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let text_model_type = config_entries
        .get("text_config")
        .and_then(text_model_type_of)
        .unwrap_or_default();
    model_type.starts_with("qwen3_5") || text_model_type.starts_with("qwen3_5")
}

fn text_model_type_of(text_config: &Value) -> Option<&str> {
    text_config.get("model_type")?.as_str()
}

/// `rope_parameters.rope_type` becomes `rope_parameters.type` unless a
/// `type` is already there.
fn rename_rope_kind(text_entries: &mut Map<String, Value>) {
    let Some(Value::Object(rope_parameters)) = text_entries.get_mut("rope_parameters") else {
        return;
    };
    if rope_parameters.contains_key("type") {
        return;
    }
    if let Some(rope_kind) = rope_parameters.remove("rope_type") {
        rope_parameters.insert("type".to_owned(), rope_kind);
    }
}
