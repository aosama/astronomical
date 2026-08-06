use std::path::{Path, PathBuf};

use thiserror::Error;

/// Closed model-family classification used at discovery and worker startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Qwen3_5,
    DeepSeekV4,
}

impl ModelFamily {
    /// Classifies a config.json model_type without claiming that the family is executable.
    #[must_use]
    pub fn from_model_type(model_type: Option<&str>) -> Option<Self> {
        match model_type {
            Some("qwen3_5") | Some("qwen3_5_moe") | Some("qwen3_5_moe_vision") => {
                Some(Self::Qwen3_5)
            }
            Some("deepseek_v4") => Some(Self::DeepSeekV4),
            _ => None,
        }
    }
}

/// Failure while reading the family marker from a selected model directory.
#[derive(Debug, Error)]
pub enum ModelFamilyClassificationError {
    #[error("failed to read model config.json: {source}")]
    ReadConfig {
        model_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model config.json: {source}")]
    ParseConfig {
        model_directory: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// Reads only model_type from a selected artifact directory.
pub fn classify_model_directory(
    model_directory: &Path,
) -> Result<Option<ModelFamily>, ModelFamilyClassificationError> {
    let config_path = model_directory.join("config.json");
    let config_bytes = std::fs::read(&config_path).map_err(|source| {
        ModelFamilyClassificationError::ReadConfig {
            model_directory: model_directory.to_path_buf(),
            source,
        }
    })?;
    let config_value: serde_json::Value =
        serde_json::from_slice(&config_bytes).map_err(|source| {
            ModelFamilyClassificationError::ParseConfig {
                model_directory: model_directory.to_path_buf(),
                source,
            }
        })?;
    Ok(ModelFamily::from_model_type(
        config_value
            .get("model_type")
            .and_then(serde_json::Value::as_str),
    ))
}

/// Classifies a model_type without claiming that the family is executable.
pub(super) fn classify_model_family(model_type: Option<&str>) -> Option<ModelFamily> {
    ModelFamily::from_model_type(model_type)
}
