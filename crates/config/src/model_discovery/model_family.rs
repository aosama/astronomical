use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use super::{deepseek_v4, flux2_klein, laguna, qwen3_5};

const MAXIMUM_FAMILY_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_PIPELINE_INDEX_BYTES: u64 = 1024 * 1024;

/// Closed model-family classification used at discovery and worker startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Qwen3_5,
    Laguna,
    DeepSeekV4,
    Flux2Klein,
}

impl ModelFamily {
    /// Classifies a config.json model_type without claiming that the family is executable.
    #[must_use]
    pub fn from_model_type(model_type: Option<&str>) -> Option<Self> {
        if qwen3_5::recognizes_model_type(model_type) {
            Some(Self::Qwen3_5)
        } else if laguna::recognizes_model_type(model_type) {
            Some(Self::Laguna)
        } else if deepseek_v4::recognizes_model_type(model_type) {
            Some(Self::DeepSeekV4)
        } else {
            None
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
    #[error(
        "model config.json has {actual_bytes} bytes, exceeding the {maximum_bytes}-byte classification limit"
    )]
    ConfigTooLarge {
        actual_bytes: u64,
        maximum_bytes: u64,
    },
    #[error("failed to read model model_index.json: {source}")]
    ReadPipelineIndex {
        model_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model model_index.json: {source}")]
    ParsePipelineIndex {
        model_directory: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "model model_index.json has {actual_bytes} bytes, exceeding the {maximum_bytes}-byte classification limit"
    )]
    PipelineIndexTooLarge {
        actual_bytes: u64,
        maximum_bytes: u64,
    },
}

/// Reads only model_type from a selected artifact directory.
pub fn classify_model_directory(
    model_directory: &Path,
) -> Result<Option<ModelFamily>, ModelFamilyClassificationError> {
    if model_directory.join("model_index.json").is_file() {
        return classify_pipeline_directory(model_directory);
    }
    let config_path = model_directory.join("config.json");
    let config_file = std::fs::File::open(&config_path).map_err(|source| {
        ModelFamilyClassificationError::ReadConfig {
            model_directory: model_directory.to_path_buf(),
            source,
        }
    })?;
    let config_size_bytes = config_file
        .metadata()
        .map_err(|source| ModelFamilyClassificationError::ReadConfig {
            model_directory: model_directory.to_path_buf(),
            source,
        })?
        .len();
    if config_size_bytes > MAXIMUM_FAMILY_CONFIG_BYTES {
        return Err(ModelFamilyClassificationError::ConfigTooLarge {
            actual_bytes: config_size_bytes,
            maximum_bytes: MAXIMUM_FAMILY_CONFIG_BYTES,
        });
    }
    let mut config_bytes = Vec::new();
    config_file
        .take(MAXIMUM_FAMILY_CONFIG_BYTES + 1)
        .read_to_end(&mut config_bytes)
        .map_err(|source| ModelFamilyClassificationError::ReadConfig {
            model_directory: model_directory.to_path_buf(),
            source,
        })?;
    if config_bytes.len() as u64 > MAXIMUM_FAMILY_CONFIG_BYTES {
        return Err(ModelFamilyClassificationError::ConfigTooLarge {
            actual_bytes: config_bytes.len() as u64,
            maximum_bytes: MAXIMUM_FAMILY_CONFIG_BYTES,
        });
    }
    let config_document: ModelFamilyConfigDocument = serde_json::from_slice(&config_bytes)
        .map_err(|source| ModelFamilyClassificationError::ParseConfig {
            model_directory: model_directory.to_path_buf(),
            source,
        })?;
    Ok(ModelFamily::from_model_type(
        config_document.model_type.as_deref(),
    ))
}

fn classify_pipeline_directory(
    model_directory: &Path,
) -> Result<Option<ModelFamily>, ModelFamilyClassificationError> {
    let pipeline_index_path = model_directory.join("model_index.json");
    let pipeline_index_file = std::fs::File::open(&pipeline_index_path).map_err(|source| {
        ModelFamilyClassificationError::ReadPipelineIndex {
            model_directory: model_directory.to_path_buf(),
            source,
        }
    })?;
    let pipeline_index_size_bytes = pipeline_index_file
        .metadata()
        .map_err(|source| ModelFamilyClassificationError::ReadPipelineIndex {
            model_directory: model_directory.to_path_buf(),
            source,
        })?
        .len();
    if pipeline_index_size_bytes > MAXIMUM_PIPELINE_INDEX_BYTES {
        return Err(ModelFamilyClassificationError::PipelineIndexTooLarge {
            actual_bytes: pipeline_index_size_bytes,
            maximum_bytes: MAXIMUM_PIPELINE_INDEX_BYTES,
        });
    }
    let mut pipeline_index_bytes = Vec::new();
    pipeline_index_file
        .take(MAXIMUM_PIPELINE_INDEX_BYTES + 1)
        .read_to_end(&mut pipeline_index_bytes)
        .map_err(|source| ModelFamilyClassificationError::ReadPipelineIndex {
            model_directory: model_directory.to_path_buf(),
            source,
        })?;
    if pipeline_index_bytes.len() as u64 > MAXIMUM_PIPELINE_INDEX_BYTES {
        return Err(ModelFamilyClassificationError::PipelineIndexTooLarge {
            actual_bytes: pipeline_index_bytes.len() as u64,
            maximum_bytes: MAXIMUM_PIPELINE_INDEX_BYTES,
        });
    }
    let is_flux2_klein =
        flux2_klein::classifies_pipeline_index(&pipeline_index_bytes).map_err(|source| {
            ModelFamilyClassificationError::ParsePipelineIndex {
                model_directory: model_directory.to_path_buf(),
                source,
            }
        })?;
    Ok(is_flux2_klein.then_some(ModelFamily::Flux2Klein))
}

/// Minimal duplicate-aware projection keeps family dispatch independent from full config shape.
#[derive(Deserialize)]
struct ModelFamilyConfigDocument {
    #[serde(default)]
    model_type: Option<String>,
}
