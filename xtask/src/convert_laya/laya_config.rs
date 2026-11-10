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
pub(crate) use crate::model_files::local_paths::without_local_paths;

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

#[cfg(test)]
#[path = "laya_config_tests.rs"]
mod tests;
