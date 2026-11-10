//! `joint_head_config.json`: the shape of Clef's joint schema head.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;

/// The settings file that, beside `joint_head.safetensors`, marks a folder
/// as a Clef decision model.
pub(crate) const CLEF_HEAD_SETTINGS_FILE_NAME: &str = "joint_head_config.json";
const MAXIMUM_SETTINGS_BYTES: u64 = 64 * 1024;

/// `JointSchemaHead`'s constructor arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClefHeadSettings {
    /// The backbone's hidden width, which the head's input projections take.
    pub(crate) hidden_size: usize,
    /// The head's own width.
    pub(crate) width: usize,
    pub(crate) routing_layer_count: usize,
    pub(crate) decoder_layer_count: usize,
    pub(crate) attention_head_count: usize,
    pub(crate) feedforward_width: usize,
}

impl ClefHeadSettings {
    pub(crate) fn read(model_directory: &Path) -> Result<Self, DecisionFailure> {
        let settings_path = model_directory.join(CLEF_HEAD_SETTINGS_FILE_NAME);
        let settings_size = std::fs::metadata(&settings_path)
            .map_err(|_| unreadable_settings("it is missing"))?
            .len();
        if settings_size > MAXIMUM_SETTINGS_BYTES {
            return Err(unreadable_settings("it is too large"));
        }
        let settings_bytes =
            std::fs::read(&settings_path).map_err(|_| unreadable_settings("it cannot be read"))?;
        Self::from_json_bytes(&settings_bytes)
    }

    pub(crate) fn from_json_bytes(settings_bytes: &[u8]) -> Result<Self, DecisionFailure> {
        let settings: serde_json::Value = serde_json::from_slice(settings_bytes)
            .map_err(|_| unreadable_settings("it is not JSON"))?;
        let mut sizes = [0_usize; 6];
        for (size, setting_name) in sizes.iter_mut().zip([
            "hidden_size",
            "width",
            "routing_layers",
            "layers",
            "heads",
            "feedforward",
        ]) {
            *size = positive_size(&settings, setting_name)?;
        }
        let [
            hidden_size,
            width,
            routing_layer_count,
            decoder_layer_count,
            attention_head_count,
            feedforward_width,
        ] = sizes;
        if !width.is_multiple_of(attention_head_count) {
            return Err(unreadable_settings(
                "its width does not divide into its heads",
            ));
        }
        Ok(Self {
            hidden_size,
            width,
            routing_layer_count,
            decoder_layer_count,
            attention_head_count,
            feedforward_width,
        })
    }
}

fn positive_size(
    settings: &serde_json::Value,
    setting_name: &str,
) -> Result<usize, DecisionFailure> {
    match settings
        .get(setting_name)
        .and_then(serde_json::Value::as_u64)
    {
        Some(size) if size > 0 && size <= u64::from(u32::MAX) => Ok(size as usize),
        _ => Err(unreadable_settings(&format!(
            "its {setting_name} is not a positive whole number"
        ))),
    }
}

fn unreadable_settings(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!(
        "{CLEF_HEAD_SETTINGS_FILE_NAME} cannot be used: {reason}"
    ))
}

#[cfg(test)]
#[path = "head_settings_tests.rs"]
mod tests;
