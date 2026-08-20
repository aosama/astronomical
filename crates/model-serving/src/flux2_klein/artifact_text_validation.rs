//! Strict text-encoder sidecar and shard validation for the official artifact.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use ::safetensors::Dtype;
use serde::Deserialize;

use crate::artifact_validation::{
    RawSafetensorsInventory, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    read_bounded_required_file_bytes, validate_required_file,
};
use crate::strict_json::DuplicateAwareJsonValue;

use super::Flux2KleinTensorInventory;
use super::artifact::Flux2KleinArtifactError;
use super::inventory::public_descriptor;

pub(super) const TEXT_INDEX_FILE_NAME: &str = "text_encoder/model.safetensors.index.json";
pub(super) const TEXT_GENERATION_CONFIG_FILE_NAME: &str = "text_encoder/generation_config.json";
const MAXIMUM_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextShardIndexDocument {
    metadata: TextShardIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextShardIndexMetadata {
    total_parameters: u64,
    total_size: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextGenerationConfig {
    bos_token_id: u32,
    do_sample: bool,
    eos_token_id: [u32; 2],
    pad_token_id: u32,
    temperature: f64,
    top_k: u32,
    top_p: f64,
    transformers_version: String,
}

struct TextShardIndex {
    total_parameters: u64,
    total_size: u64,
    weight_map: BTreeMap<String, String>,
    shard_file_names: Vec<String>,
}

impl TextShardIndex {
    fn parse(bytes: &[u8]) -> Result<Self, Flux2KleinArtifactError> {
        let duplicate_aware = serde_json::from_slice::<DuplicateAwareJsonValue>(bytes)
            .map_err(Flux2KleinArtifactError::MalformedTextShardIndex)?;
        let document = serde_json::from_value::<TextShardIndexDocument>(duplicate_aware.0)
            .map_err(Flux2KleinArtifactError::MalformedTextShardIndex)?;
        let mut supported_names = BTreeSet::new();
        for shard_file_name in document.weight_map.values() {
            if !is_safe_text_shard_path(shard_file_name) {
                return Err(Flux2KleinArtifactError::UnsupportedTextShardName {
                    shard_file_name: shard_file_name.clone(),
                });
            }
            supported_names.insert(shard_file_name.clone());
        }
        if supported_names.is_empty() {
            return Err(Flux2KleinArtifactError::ArtifactFile {
                file_name: TEXT_INDEX_FILE_NAME.to_owned(),
            });
        }
        Ok(Self {
            total_parameters: document.metadata.total_parameters,
            total_size: document.metadata.total_size,
            weight_map: document.weight_map,
            shard_file_names: supported_names.into_iter().collect(),
        })
    }
}

pub(super) fn validate_text_artifacts(
    model_directory: &Path,
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
) -> Result<
    (
        BTreeMap<String, ValidatedWeightsFile>,
        Flux2KleinTensorInventory,
    ),
    Flux2KleinArtifactError,
> {
    let index = TextShardIndex::parse(&read_document(document_files, TEXT_INDEX_FILE_NAME)?)?;
    validate_text_generation_config(&read_document(
        document_files,
        TEXT_GENERATION_CONFIG_FILE_NAME,
    )?)?;
    validate_text_shards(model_directory, &index)
}

fn read_document(
    files: &BTreeMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<Vec<u8>, Flux2KleinArtifactError> {
    let file = files
        .get(file_name)
        .ok_or_else(|| Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        })?;
    read_bounded_required_file_bytes(file, MAXIMUM_DOCUMENT_BYTES).map_err(|_| {
        Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        }
    })
}

fn validate_text_generation_config(bytes: &[u8]) -> Result<(), Flux2KleinArtifactError> {
    let duplicate_aware =
        serde_json::from_slice::<DuplicateAwareJsonValue>(bytes).map_err(|_| {
            Flux2KleinArtifactError::ArtifactFile {
                file_name: TEXT_GENERATION_CONFIG_FILE_NAME.to_owned(),
            }
        })?;
    let config =
        serde_json::from_value::<TextGenerationConfig>(duplicate_aware.0).map_err(|_| {
            Flux2KleinArtifactError::ArtifactFile {
                file_name: TEXT_GENERATION_CONFIG_FILE_NAME.to_owned(),
            }
        })?;
    if config.bos_token_id == 151_643
        && config.do_sample
        && config.eos_token_id == [151_645, 151_643]
        && config.pad_token_id == 151_643
        && config.temperature == 0.6
        && config.top_k == 20
        && config.top_p == 0.95
        && config.transformers_version == "4.56.1"
    {
        Ok(())
    } else {
        Err(Flux2KleinArtifactError::ArtifactFile {
            file_name: TEXT_GENERATION_CONFIG_FILE_NAME.to_owned(),
        })
    }
}

fn validate_text_shards(
    model_directory: &Path,
    index: &TextShardIndex,
) -> Result<
    (
        BTreeMap<String, ValidatedWeightsFile>,
        Flux2KleinTensorInventory,
    ),
    Flux2KleinArtifactError,
> {
    let mut files = BTreeMap::new();
    let mut descriptors = Vec::new();
    let mut payload_bytes = 0_u64;
    let mut parameter_count = 0_u64;
    let mut physical_names = BTreeSet::new();
    for index_file_name in &index.shard_file_names {
        let relative_file_name = format!("text_encoder/{index_file_name}");
        let (weights_file, inventory) = open_weights(model_directory, &relative_file_name)?;
        payload_bytes = payload_bytes
            .checked_add(inventory.shard_payload_bytes)
            .ok_or(Flux2KleinArtifactError::PayloadAccountingOverflow)?;
        for tensor in inventory.tensor_descriptors {
            let tensor_parameter_count =
                tensor.shape.iter().try_fold(1_u64, |count, dimension| {
                    let dimension = u64::try_from(*dimension)
                        .map_err(|_| Flux2KleinArtifactError::PayloadAccountingOverflow)?;
                    count
                        .checked_mul(dimension)
                        .ok_or(Flux2KleinArtifactError::PayloadAccountingOverflow)
                })?;
            parameter_count = parameter_count
                .checked_add(tensor_parameter_count)
                .ok_or(Flux2KleinArtifactError::PayloadAccountingOverflow)?;
            if tensor.dtype != Dtype::BF16 {
                return Err(Flux2KleinArtifactError::TensorDtype {
                    component: "text encoder",
                    tensor_name: tensor.tensor_name,
                });
            }
            if index
                .weight_map
                .get(&tensor.tensor_name)
                .map(String::as_str)
                != Some(index_file_name.as_str())
                || !physical_names.insert(tensor.tensor_name.clone())
            {
                return Err(Flux2KleinArtifactError::TextShardIndexDisagreement {
                    tensor_name: tensor.tensor_name,
                });
            }
            descriptors.push(public_descriptor(&relative_file_name, tensor));
        }
        files.insert(relative_file_name, weights_file);
    }
    if let Some(missing_name) = index
        .weight_map
        .keys()
        .find(|name| !physical_names.contains(*name))
    {
        return Err(Flux2KleinArtifactError::TextShardIndexDisagreement {
            tensor_name: missing_name.clone(),
        });
    }
    if index.total_size != payload_bytes {
        return Err(Flux2KleinArtifactError::TextShardIndexTotalSizeMismatch {
            declared_bytes: index.total_size,
            actual_bytes: payload_bytes,
        });
    }
    if index.total_parameters != parameter_count {
        return Err(
            Flux2KleinArtifactError::TextShardIndexTotalParameterMismatch {
                declared_parameters: index.total_parameters,
                actual_parameters: parameter_count,
            },
        );
    }
    Ok((
        files,
        Flux2KleinTensorInventory {
            descriptors,
            payload_bytes,
            double_stream_block_count: 0,
            single_stream_block_count: 0,
            up_block_count: 0,
        },
    ))
}

fn is_safe_text_shard_path(shard_file_name: &str) -> bool {
    let shard_path = Path::new(shard_file_name);
    !shard_file_name.is_empty()
        && !shard_file_name.contains('\\')
        && !shard_path.is_absolute()
        && shard_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && shard_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
}

fn open_weights(
    model_directory: &Path,
    file_name: &str,
) -> Result<(ValidatedWeightsFile, RawSafetensorsInventory), Flux2KleinArtifactError> {
    let weights = validate_required_file(
        model_directory,
        &RequiredFileProfile {
            file_name: file_name.to_owned(),
            size_bytes: 0,
        },
    )
    .map_err(|_| Flux2KleinArtifactError::ArtifactFile {
        file_name: file_name.to_owned(),
    })?
    .into_validated_weights_file()
    .map_err(|_| Flux2KleinArtifactError::ArtifactFile {
        file_name: file_name.to_owned(),
    })?;
    let inventory = weights.read_raw_safetensors_inventory().map_err(|_| {
        Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        }
    })?;
    Ok((weights, inventory))
}
