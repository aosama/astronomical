//! Builds the converted model's `config.json` and tokenizer settings.
//!
//! A PyTorch Laya checkpoint spreads its settings over `rl_agent_config.json`
//! (head settings, temperatures, and a training record) and
//! `encoder/config.json`. The MLX layout keeps one `config.json` with the
//! encoder's config nested under `encoder`.
//!
//! The output is built from a fixed list of fields, never by copying the
//! source document, because the training record names files on the machine
//! that trained the model. Whatever is copied is also scrubbed of anything
//! that looks like a local path.

use serde_json::{Map, Value};

use super::affine_quantization::{QUANTIZATION_BITS, QUANTIZATION_GROUP_SIZE};

/// The head and calibration settings the engine reads, copied when present.
const COPIED_SETTING_NAMES: [&str; 7] = [
    "head_layers",
    "max_len",
    "head_max_len",
    "act_costs",
    "temperature",
    "temperature_by_options",
    "model_name",
];
/// Hugging Face records where a model was loaded from under this key.
const SOURCE_LOCATION_KEY: &str = "_name_or_path";

/// The MLX `config.json` for a checkpoint with these two source documents.
pub(crate) fn mlx_laya_config(
    checkpoint_settings: &Value,
    encoder_config: &Value,
) -> Result<Value, String> {
    if !encoder_config.is_object() {
        return Err("encoder/config.json is not a JSON object".to_owned());
    }
    let Some(temperatures) = checkpoint_settings
        .get("temperature")
        .and_then(Value::as_array)
    else {
        return Err(
            "rl_agent_config.json has no temperature list; the checkpoint was not calibrated"
                .to_owned(),
        );
    };
    if temperatures.len() != 3 || !temperatures.iter().all(Value::is_number) {
        return Err(
            "rl_agent_config.json must hold three temperatures: choice, score and noul".to_owned(),
        );
    }
    let mut config_members = Map::new();
    config_members.insert("model_type".to_owned(), Value::from("laya"));
    config_members.insert(
        "encoder".to_owned(),
        without_local_paths(encoder_config.clone()),
    );
    for setting_name in COPIED_SETTING_NAMES {
        if let Some(setting_value) = checkpoint_settings.get(setting_name) {
            config_members.insert(
                setting_name.to_owned(),
                without_local_paths(setting_value.clone()),
            );
        }
    }
    let mut quantization_members = Map::new();
    quantization_members.insert(
        "group_size".to_owned(),
        Value::from(QUANTIZATION_GROUP_SIZE),
    );
    quantization_members.insert("bits".to_owned(), Value::from(QUANTIZATION_BITS));
    config_members.insert(
        "quantization".to_owned(),
        Value::Object(quantization_members),
    );
    Ok(Value::Object(config_members))
}

/// `document` with every source-location key and every string that looks
/// like a local path removed, at any depth.
pub(crate) fn without_local_paths(document: Value) -> Value {
    match document {
        Value::Object(members) => {
            let mut kept_members = Map::new();
            for (member_name, member_value) in members {
                if member_name == SOURCE_LOCATION_KEY || is_local_path_text(&member_value) {
                    continue;
                }
                kept_members.insert(member_name, without_local_paths(member_value));
            }
            Value::Object(kept_members)
        }
        Value::Array(elements) => {
            let mut kept_elements = Vec::with_capacity(elements.len());
            for element in elements {
                if !is_local_path_text(&element) {
                    kept_elements.push(without_local_paths(element));
                }
            }
            Value::Array(kept_elements)
        }
        other_value => other_value,
    }
}

/// Whether `value` is text that names a place on a local disk: an absolute
/// path, a home-relative path, a relative path that climbs or starts at the
/// current folder, or a Windows drive path.
fn is_local_path_text(value: &Value) -> bool {
    let Some(text) = value.as_str() else {
        return false;
    };
    let starts_like_a_path = ["/", "~/", "./", "../", "file:"]
        .iter()
        .any(|path_start| text.starts_with(path_start));
    let text_bytes = text.as_bytes();
    let is_windows_drive_path = text_bytes.len() > 2
        && text_bytes[0].is_ascii_alphabetic()
        && text_bytes[1] == b':'
        && (text_bytes[2] == b'\\' || text_bytes[2] == b'/');
    starts_like_a_path || is_windows_drive_path
}

#[cfg(test)]
#[path = "laya_config_tests.rs"]
mod tests;
