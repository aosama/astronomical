use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use super::flux2_klein_mlx_4bit;
use super::{flux2_klein, laya, modernbert, prism_hadamard_qwen35, qwen_image_21, qwen3_5};

const MAXIMUM_FAMILY_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_PIPELINE_INDEX_BYTES: u64 = 1024 * 1024;

/// Closed model-family classification used at discovery and worker startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Qwen3_5,
    PrismHadamardQwen35,
    Flux2Klein,
    QwenImage21,
    ModernBert,
    /// A decision model: a ModernBERT encoder scoring one marker per option.
    Laya,
}

impl ModelFamily {
    /// Classifies a config.json model_type without claiming that the family is executable.
    #[must_use]
    pub fn from_model_type(model_type: Option<&str>) -> Option<Self> {
        if qwen3_5::recognizes_model_type(model_type) {
            Some(Self::Qwen3_5)
        } else if prism_hadamard_qwen35::recognizes_model_type(model_type) {
            Some(Self::PrismHadamardQwen35)
        } else {
            Self::encoder_family_from_model_type(model_type)
        }
    }

    /// The families built on a ModernBERT encoder: the embedding model
    /// itself, and the decision model that adds a head to it.
    fn encoder_family_from_model_type(model_type: Option<&str>) -> Option<Self> {
        if modernbert::recognizes_model_type(model_type) {
            Some(Self::ModernBert)
        } else if laya::recognizes_model_type(model_type) {
            Some(Self::Laya)
        } else {
            None
        }
    }

    /// Wire label matching the model_type spelling the discovery layer recognizes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        crate::exhaustive_table!(self, {
            Self::Qwen3_5 => "qwen3_5",
            Self::PrismHadamardQwen35 => "prism_hadamard_qwen35",
            Self::Flux2Klein => "flux2_klein",
            Self::QwenImage21 => "qwen_image_21",
            Self::ModernBert => "modernbert",
            Self::Laya => "laya",
        })
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

/// Classifies a selected artifact directory from the one file its family is
/// known by. The order is the download preflight's too: the pipeline index,
/// then the 4-bit FLUX.2 Klein transformer index, then `config.json`.
pub fn classify_model_directory(
    model_directory: &Path,
) -> Result<Option<ModelFamily>, ModelFamilyClassificationError> {
    if model_directory.join("model_index.json").is_file() {
        return classify_pipeline_directory(model_directory);
    }
    if flux2_klein_mlx_4bit::classifies_directory(model_directory) {
        return Ok(Some(ModelFamily::Flux2Klein));
    }
    classify_config_directory(model_directory)
}

/// Reads `model_type` from the directory's `config.json`.
fn classify_config_directory(
    model_directory: &Path,
) -> Result<Option<ModelFamily>, ModelFamilyClassificationError> {
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
    if is_flux2_klein {
        return Ok(Some(ModelFamily::Flux2Klein));
    }
    let is_qwen_image_21 = qwen_image_21::classifies_pipeline_index(&pipeline_index_bytes)
        .map_err(
            |source| ModelFamilyClassificationError::ParsePipelineIndex {
                model_directory: model_directory.to_path_buf(),
                source,
            },
        )?;
    Ok(is_qwen_image_21.then_some(ModelFamily::QwenImage21))
}

/// Whether `directory` is a Diffusers pipeline root, with or without a
/// pipeline index. Its component folders are never separate models.
pub(super) fn is_pipeline_root(directory: &Path) -> bool {
    directory.join("model_index.json").is_file()
        || flux2_klein_mlx_4bit::has_build_marker(directory)
}

/// Whether `directory` carries the file a model family classifies from, so a
/// scan stops there instead of descending into it.
pub(super) fn has_model_root_marker(directory: &Path) -> bool {
    directory.join("config.json").is_file() || is_pipeline_root(directory)
}

/// Classifies an mflux transformer weight index without a file-system read, so
/// the download preflight applies the same rule as disk discovery.
///
/// # Errors
/// Returns the index document parse failure verbatim.
pub fn classify_mflux_transformer_index_bytes(
    index_bytes: &[u8],
) -> Result<Option<ModelFamily>, serde_json::Error> {
    Ok(
        flux2_klein_mlx_4bit::classifies_transformer_index(index_bytes)?
            .then_some(ModelFamily::Flux2Klein),
    )
}

/// Classifies a Diffusers pipeline index document without a file-system read, reusing the exact
/// production pipeline classifier so download preflight cannot drift from disk discovery.
///
/// # Errors
/// Returns the pipeline document parse failure verbatim.
pub fn classify_pipeline_index_bytes(
    pipeline_index_bytes: &[u8],
) -> Result<Option<ModelFamily>, serde_json::Error> {
    let is_flux2_klein = flux2_klein::classifies_pipeline_index(pipeline_index_bytes)?;
    if is_flux2_klein {
        return Ok(Some(ModelFamily::Flux2Klein));
    }
    let is_qwen_image_21 = qwen_image_21::classifies_pipeline_index(pipeline_index_bytes)?;
    Ok(is_qwen_image_21.then_some(ModelFamily::QwenImage21))
}

/// Minimal duplicate-aware projection keeps family dispatch independent from full config shape.
#[derive(Deserialize)]
struct ModelFamilyConfigDocument {
    #[serde(default)]
    model_type: Option<String>,
}
