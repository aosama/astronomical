//! Loads and validates one Laya artifact directory before any GPU allocation.
//!
//! Startup rereads the on-disk evidence: a bounded `config.json`, a bounded
//! tokenizer pair and one safetensors payload. Tensor shapes are checked when
//! the engine binds lazy arrays by name.

use std::fs::File;
use std::path::Path;

use thiserror::Error;

use crate::decision_models::laya::configuration::{LayaConfiguration, LayaConfigurationError};

const MAXIMUM_CONFIG_JSON_BYTES: u64 = 1_048_576;
/// The multilingual tokenizer is about 34 MB; this leaves room for a larger vocabulary.
const MAXIMUM_TOKENIZER_JSON_BYTES: u64 = 67_108_864;
const MAXIMUM_TOKENIZER_SETTINGS_BYTES: u64 = 1_048_576;

/// Everything the engine needs to load one Laya checkpoint.
pub(crate) struct LayaArtifact {
    pub(crate) configuration: LayaConfiguration,
    pub(crate) tokenizer_json_bytes: Vec<u8>,
    pub(crate) tokenizer_settings_bytes: Vec<u8>,
    pub(crate) weights_file: File,
}

/// Why a directory is not a servable Laya artifact.
#[derive(Debug, Error)]
pub(crate) enum LayaArtifactError {
    #[error("the Laya artifact file {file_name} is missing or unreadable: {source}")]
    UnreadableFile {
        file_name: &'static str,
        source: std::io::Error,
    },
    #[error(
        "the Laya artifact file {file_name} is {actual_bytes} bytes, over the {maximum_bytes} byte limit"
    )]
    FileTooLarge {
        file_name: &'static str,
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    #[error("the Laya artifact config.json is not valid JSON: {0}")]
    MalformedConfig(#[from] serde_json::Error),
    #[error(transparent)]
    Configuration(#[from] LayaConfigurationError),
}

impl LayaArtifact {
    /// Validates shallow artifact completeness without reading model bytes.
    pub(crate) fn load(model_directory: &Path) -> Result<Self, LayaArtifactError> {
        let config_bytes = bounded_read(model_directory, "config.json", MAXIMUM_CONFIG_JSON_BYTES)?;
        let config_value: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let configuration = LayaConfiguration::from_config_value(&config_value)?;
        let tokenizer_json_bytes = bounded_read(
            model_directory,
            "tokenizer.json",
            MAXIMUM_TOKENIZER_JSON_BYTES,
        )?;
        let tokenizer_settings_bytes = bounded_read(
            model_directory,
            "tokenizer_config.json",
            MAXIMUM_TOKENIZER_SETTINGS_BYTES,
        )?;
        let weights_file = match File::open(model_directory.join("model.safetensors")) {
            Ok(weights_file) => weights_file,
            Err(source) => {
                return Err(LayaArtifactError::UnreadableFile {
                    file_name: "model.safetensors",
                    source,
                });
            }
        };
        Ok(Self {
            configuration,
            tokenizer_json_bytes,
            tokenizer_settings_bytes,
            weights_file,
        })
    }
}

/// Reads one artifact file, refusing it when it is larger than `maximum_bytes`.
///
/// The size is checked before reading so a mislabeled multi-gigabyte file is
/// never pulled into memory as a config.
fn bounded_read(
    model_directory: &Path,
    file_name: &'static str,
    maximum_bytes: u64,
) -> Result<Vec<u8>, LayaArtifactError> {
    let file_path = model_directory.join(file_name);
    let actual_bytes = match std::fs::metadata(&file_path) {
        Ok(file_metadata) => file_metadata.len(),
        Err(source) => return Err(LayaArtifactError::UnreadableFile { file_name, source }),
    };
    if actual_bytes > maximum_bytes {
        return Err(LayaArtifactError::FileTooLarge {
            file_name,
            actual_bytes,
            maximum_bytes,
        });
    }
    match std::fs::read(&file_path) {
        Ok(file_bytes) => Ok(file_bytes),
        Err(source) => Err(LayaArtifactError::UnreadableFile { file_name, source }),
    }
}

#[cfg(test)]
#[path = "artifact_tests.rs"]
mod tests;
