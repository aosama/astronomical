//! Classifies a folder as one registered model family, by asking each
//! family's own recognizers in registry order.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use super::flux2_klein_mlx_4bit;
use super::model_family_description::ModelFamily;
use super::model_family_registry::REGISTERED_MODEL_FAMILIES;

const MAXIMUM_FAMILY_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_PIPELINE_INDEX_BYTES: u64 = 1024 * 1024;

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
    if let Some(marked_family) = family_marked_by_file(model_directory) {
        return Ok(Some(marked_family));
    }
    classify_config_directory(model_directory)
}

/// The first family, in registry order, whose marker file the folder
/// carries: the 4-bit FLUX.2 Klein build, whose pipeline index lives under
/// its transformer; Kev and Clef, whose configs are their Qwen3.5
/// backbones'; and Kokoro and Phonon, whose configs have no model type.
fn family_marked_by_file(model_directory: &Path) -> Option<ModelFamily> {
    REGISTERED_MODEL_FAMILIES
        .iter()
        .find(|family_description| family_description.claims_directory(model_directory))
        .map(|family_description| family_description.model_family)
}

/// The first family, in registry order, that claims this `model_type`.
pub(super) fn family_of_model_type(model_type: Option<&str>) -> Option<ModelFamily> {
    REGISTERED_MODEL_FAMILIES
        .iter()
        .find(|family_description| family_description.claims_model_type(model_type))
        .map(|family_description| family_description.model_family)
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
    Ok(family_of_model_type(config_document.model_type.as_deref()))
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
    classify_pipeline_index_bytes(&pipeline_index_bytes).map_err(|source| {
        ModelFamilyClassificationError::ParsePipelineIndex {
            model_directory: model_directory.to_path_buf(),
            source,
        }
    })
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
            .then_some(super::flux2_klein::FLUX2_KLEIN_MODEL_FAMILY),
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
    for family_description in &REGISTERED_MODEL_FAMILIES {
        if family_description.claims_pipeline_index(pipeline_index_bytes)? {
            return Ok(Some(family_description.model_family));
        }
    }
    Ok(None)
}

/// Minimal duplicate-aware projection keeps family dispatch independent from full config shape.
#[derive(Deserialize)]
struct ModelFamilyConfigDocument {
    #[serde(default)]
    model_type: Option<String>,
}
