//! `kev_decision.json`: the settings `convert-kev` carried over from Kev's
//! `head.pt`.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;

/// The settings file that marks a folder as a Kev decision model.
pub(crate) const KEV_SETTINGS_FILE_NAME: &str = "kev_decision.json";
const MAXIMUM_SETTINGS_BYTES: u64 = 64 * 1024;

/// What the engine needs from the settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KevSettings {
    /// What every logit is divided by before the softmax.
    pub(crate) temperature: f64,
    /// The width of the head's query and key projections.
    pub(crate) pointer_dimension: usize,
}

impl KevSettings {
    pub(crate) fn read(model_directory: &Path) -> Result<Self, DecisionFailure> {
        let settings_path = model_directory.join(KEV_SETTINGS_FILE_NAME);
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
        if settings
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            != Some("kev")
        {
            return Err(unreadable_settings("it does not describe a Kev model"));
        }
        let temperature = settings
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .filter(|temperature| temperature.is_finite() && *temperature > 0.0)
            .ok_or_else(|| unreadable_settings("its temperature is not a positive number"))?;
        let pointer_dimension = settings
            .get("pointer_dimension")
            .and_then(serde_json::Value::as_u64)
            .filter(|pointer_dimension| *pointer_dimension > 0)
            .ok_or_else(|| {
                unreadable_settings("its pointer_dimension is not a positive whole number")
            })?;
        Ok(Self {
            temperature,
            pointer_dimension: pointer_dimension as usize,
        })
    }
}

fn unreadable_settings(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!(
        "{KEV_SETTINGS_FILE_NAME} cannot be used: {reason}"
    ))
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
