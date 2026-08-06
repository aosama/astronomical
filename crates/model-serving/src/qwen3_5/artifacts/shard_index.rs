use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use thiserror::Error;

use crate::artifact_validation::TensorProfile;

use super::tensor_spec::validate_language_tensor_names;

pub const MAXIMUM_INDEX_BYTES: usize = 1024 * 1024;
const MAXIMUM_TENSOR_NAME_BYTES: usize = 512;

/// The bounded executable tensor-to-shard inventory for a Qwen3.5 artifact.
///
/// Tracks both language model tensors (`language_model.*`) and vision tower
/// tensors (`vision_tower.*`) and their shard file locations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5ShardIndex {
    tensor_name_to_shard_file_name: BTreeMap<String, String>,
    mtp_tensor_name_to_shard_file_name: BTreeMap<String, String>,
    /// Vision tensor name to shard file name mapping for embedded or sidecar storage.
    vision_tensor_name_to_shard_file_name: BTreeMap<String, String>,
    total_payload_bytes: u64,
    model_shard_file_names: Vec<String>,
    vision_sidecar_file_names: Vec<String>,
    mtp_only_shard_file_names: Vec<String>,
}

impl Qwen3_5ShardIndex {
    /// Parses and independently validates the executable language inventory.
    /// Also collects vision tower tensor mappings for embedded and sidecar storage.
    pub fn from_json_bytes(
        index_bytes: &[u8],
        language_tensor_profiles: &[TensorProfile],
    ) -> Result<Self, Qwen3_5ArtifactError> {
        if index_bytes.len() > MAXIMUM_INDEX_BYTES {
            return Err(Qwen3_5ArtifactError::IndexTooLarge {
                actual_index_bytes: index_bytes.len(),
                maximum_index_bytes: MAXIMUM_INDEX_BYTES,
            });
        }
        let index_document = serde_json::from_slice::<Qwen3_5ShardIndexDocument>(index_bytes)
            .map_err(Qwen3_5ArtifactError::DeserializeIndex)?;
        let total_payload_bytes = index_document.metadata.total_size;
        // Collect actual model shard file names from the index rather than
        // hardcoding names. The index remains authoritative for logical tensor
        // ownership; physical duplicate tensors are handled during validation.
        let mut language_tensor_names = BTreeSet::new();
        let mut language_tensor_name_to_shard_file_name = BTreeMap::new();
        let mut mtp_tensor_name_to_shard_file_name = BTreeMap::new();
        let mut vision_tensor_name_to_shard_file_name = BTreeMap::new();
        let mut language_shard_file_names = BTreeSet::new();
        let mut mtp_shard_file_names = BTreeSet::new();
        let mut language_or_mtp_shard_file_names = BTreeSet::new();
        let mut vision_shard_file_names = BTreeSet::new();
        for (tensor_name, shard_file_name) in &index_document.weight_map {
            validate_tensor_name(tensor_name)?;
            if tensor_name.starts_with("language_model.") {
                if contains_mtp_component(tensor_name) {
                    // Qwen3.6 oQ artifacts embed the optional MTP head in the
                    // same shards as the autoregressive trunk. The head has its
                    // own strict profile and is not part of the trunk inventory.
                    mtp_shard_file_names.insert(shard_file_name.clone());
                    language_or_mtp_shard_file_names.insert(shard_file_name.clone());
                    mtp_tensor_name_to_shard_file_name
                        .insert(tensor_name.clone(), shard_file_name.clone());
                    continue;
                }
                language_shard_file_names.insert(shard_file_name.clone());
                language_or_mtp_shard_file_names.insert(shard_file_name.clone());
                language_tensor_names.insert(tensor_name.as_str());
                language_tensor_name_to_shard_file_name
                    .insert(tensor_name.clone(), shard_file_name.clone());
            } else if tensor_name.starts_with("vision_tower.") {
                // Embedded vision tensors may share language shards or occupy
                // dedicated vision files. A file containing only vision tensors
                // is loaded through the separate vision-tower path, regardless
                // of its filename.
                vision_shard_file_names.insert(shard_file_name.clone());
                vision_tensor_name_to_shard_file_name
                    .insert(tensor_name.clone(), shard_file_name.clone());
            }
            // Other tensor prefixes (e.g., "mtp.") are silently skipped.
        }
        validate_language_tensor_names(&language_tensor_names, language_tensor_profiles)?;

        let vision_only_shard_file_names = vision_shard_file_names
            .difference(&language_or_mtp_shard_file_names)
            .cloned()
            .collect::<BTreeSet<_>>();
        let vision_sidecar_file_names =
            vision_only_shard_file_names.into_iter().collect::<Vec<_>>();
        let mtp_only_shard_file_names = mtp_shard_file_names
            .difference(&language_shard_file_names)
            .cloned()
            .collect::<Vec<_>>();
        let model_shard_file_names = language_shard_file_names.into_iter().collect::<Vec<_>>();
        Ok(Self {
            tensor_name_to_shard_file_name: language_tensor_name_to_shard_file_name,
            mtp_tensor_name_to_shard_file_name,
            vision_tensor_name_to_shard_file_name,
            total_payload_bytes,
            model_shard_file_names,
            vision_sidecar_file_names,
            mtp_only_shard_file_names,
        })
    }

    /// Returns the total payload bytes declared by the index.
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    /// Returns the executable text-model tensor count.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_name_to_shard_file_name.len()
    }

    /// Returns the count of executable text-model tensors.
    #[must_use]
    pub fn language_tensor_count(&self) -> usize {
        self.tensor_name_to_shard_file_name.len()
    }

    /// Returns the count of optional MTP-head tensors recorded in language shards.
    #[must_use]
    pub fn mtp_tensor_count(&self) -> usize {
        self.mtp_tensor_name_to_shard_file_name.len()
    }

    /// Returns the optional MTP-head tensor location inventory.
    #[must_use]
    pub fn mtp_tensor_name_to_shard_file_name(&self) -> &BTreeMap<String, String> {
        &self.mtp_tensor_name_to_shard_file_name
    }

    /// Returns the vision tower tensor name to shard file name mapping.
    #[must_use]
    pub fn vision_tensor_name_to_shard_file_name(&self) -> &BTreeMap<String, String> {
        &self.vision_tensor_name_to_shard_file_name
    }

    /// Returns vision tensor names that belong to one indexed file.
    #[must_use]
    pub fn vision_tensor_names_for_shard(&self, shard_file_name: &str) -> Vec<&str> {
        self.vision_tensor_name_to_shard_file_name
            .iter()
            .filter_map(|(tensor_name, tensor_shard_file_name)| {
                (tensor_shard_file_name == shard_file_name).then_some(tensor_name.as_str())
            })
            .collect()
    }

    /// Returns executable model shard file names in sorted order.
    ///
    /// Includes target-language files and language files with embedded vision.
    /// MTP-only files are retained separately so target-only loading does not map them.
    #[must_use]
    pub fn model_shard_file_names(&self) -> &[String] {
        &self.model_shard_file_names
    }

    /// Returns vision-only files that are loaded separately from language shards.
    #[must_use]
    pub fn vision_sidecar_file_names(&self) -> &[String] {
        &self.vision_sidecar_file_names
    }

    /// Returns MTP-only files that may be absent without blocking target serving.
    #[must_use]
    pub fn mtp_only_shard_file_names(&self) -> &[String] {
        &self.mtp_only_shard_file_names
    }

    /// Returns whether the indexed file contains only optional MTP tensors.
    #[must_use]
    pub fn is_mtp_only_shard_file(&self, shard_file_name: &str) -> bool {
        self.mtp_only_shard_file_names
            .iter()
            .any(|indexed_file_name| indexed_file_name == shard_file_name)
    }

    /// Removes an absent optional MTP file and its logical tensor ownership.
    pub fn omit_optional_mtp_shard_file(&mut self, shard_file_name: &str) -> bool {
        if !self.is_mtp_only_shard_file(shard_file_name) {
            return false;
        }
        self.mtp_tensor_name_to_shard_file_name
            .retain(|_, indexed_shard_file_name| indexed_shard_file_name != shard_file_name);
        self.mtp_only_shard_file_names
            .retain(|indexed_file_name| indexed_file_name != shard_file_name);
        true
    }

    /// Returns whether the indexed file is a separately loaded vision file.
    #[must_use]
    pub fn is_vision_sidecar_file(&self, shard_file_name: &str) -> bool {
        self.vision_sidecar_file_names
            .iter()
            .any(|indexed_file_name| indexed_file_name == shard_file_name)
    }

    /// Returns the mapping from language tensor names to their containing shard
    /// file names. Used by expert paging to locate weight tensors in shard files.
    #[must_use]
    pub fn language_tensor_name_to_shard_file_name(&self) -> &BTreeMap<String, String> {
        &self.tensor_name_to_shard_file_name
    }

    /// Returns the number of executable model shards.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.model_shard_file_names.len()
    }

    /// Resolves one exact tensor name to its shard file.
    #[must_use]
    pub fn shard_file_name_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.tensor_name_to_shard_file_name
            .get(tensor_name)
            .map(String::as_str)
    }

    /// Resolves one exact MTP-head tensor name to its shard file.
    #[must_use]
    pub fn shard_file_name_for_mtp_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.mtp_tensor_name_to_shard_file_name
            .get(tensor_name)
            .map(String::as_str)
    }

    /// Returns language tensor names that belong to one shard.
    #[must_use]
    pub fn language_tensor_names_for_shard(&self, shard_file_name: &str) -> Vec<&str> {
        self.tensor_name_to_shard_file_name
            .iter()
            .filter_map(|(tensor_name, tensor_shard_file_name)| {
                if tensor_shard_file_name == shard_file_name
                    && tensor_name.starts_with("language_model.")
                {
                    Some(tensor_name.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns MTP-head tensor names that belong to one language shard.
    #[must_use]
    pub fn mtp_tensor_names_for_shard(&self, shard_file_name: &str) -> Vec<&str> {
        self.mtp_tensor_name_to_shard_file_name
            .iter()
            .filter_map(|(tensor_name, tensor_shard_file_name)| {
                (tensor_shard_file_name == shard_file_name).then_some(tensor_name.as_str())
            })
            .collect()
    }

    /// Extracts the set of language tensor names from the safetensors index JSON
    /// without performing any validation against tensor profiles.
    ///
    /// This is used to determine which target and MTP modules are quantized vs.
    /// unquantized by checking for affine companion tensors before the full
    /// validation pass that requires complete tensor profiles.
    pub fn extract_language_tensor_names_from_json(
        index_bytes: &[u8],
    ) -> Result<BTreeSet<String>, Qwen3_5ArtifactError> {
        if index_bytes.len() > MAXIMUM_INDEX_BYTES {
            return Err(Qwen3_5ArtifactError::IndexTooLarge {
                actual_index_bytes: index_bytes.len(),
                maximum_index_bytes: MAXIMUM_INDEX_BYTES,
            });
        }
        let index_document = serde_json::from_slice::<Qwen3_5ShardIndexDocument>(index_bytes)
            .map_err(Qwen3_5ArtifactError::DeserializeIndex)?;
        Ok(index_document
            .weight_map
            .keys()
            .filter(|tensor_name| tensor_name.starts_with("language_model."))
            .cloned()
            .collect())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen3_5ShardIndexDocument {
    metadata: Qwen3_5ShardIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qwen3_5ShardIndexMetadata {
    total_size: u64,
    #[allow(dead_code)]
    total_parameters: Option<u64>,
}

fn validate_tensor_name(tensor_name: &str) -> Result<(), Qwen3_5ArtifactError> {
    if tensor_name.is_empty() || tensor_name.len() > MAXIMUM_TENSOR_NAME_BYTES {
        return Err(Qwen3_5ArtifactError::InvalidTensorNameLength {
            tensor_name: tensor_name.to_owned(),
            maximum_tensor_name_bytes: MAXIMUM_TENSOR_NAME_BYTES,
        });
    }
    Ok(())
}

fn contains_mtp_component(tensor_name: &str) -> bool {
    tensor_name
        .split('.')
        .any(|component| component == "mtp" || component.starts_with("mtp_"))
}

/// A bounded structural mismatch in the Qwen3.5 shard index.
#[derive(Debug, Error)]
pub enum Qwen3_5ArtifactError {
    #[error("Qwen3.5 shard index is {actual_index_bytes} bytes, exceeding {maximum_index_bytes}")]
    IndexTooLarge {
        actual_index_bytes: usize,
        maximum_index_bytes: usize,
    },
    #[error("failed to decode the Qwen3.5 shard index")]
    DeserializeIndex(#[source] serde_json::Error),
    #[error(
        "invalid Qwen3.5 tensor name length for '{tensor_name}' (maximum {maximum_tensor_name_bytes} bytes)"
    )]
    InvalidTensorNameLength {
        tensor_name: String,
        maximum_tensor_name_bytes: usize,
    },
    #[error("Qwen3.5 index contains unexpected executable language tensor '{tensor_name}'")]
    UnexpectedLanguageTensor { tensor_name: String },
    #[error("Qwen3.5 index is missing executable language tensor '{tensor_name}'")]
    MissingLanguageTensor { tensor_name: String },
    #[error("Qwen3.5 index contains unexpected MTP tensor '{tensor_name}'")]
    UnexpectedMtpTensor { tensor_name: String },
    #[error("Qwen3.5 index is missing MTP tensor '{tensor_name}'")]
    MissingMtpTensor { tensor_name: String },
    #[error("Qwen3.5 index contains unexpected vision tensor '{tensor_name}'")]
    UnexpectedVisionTensor { tensor_name: String },
    #[error("Qwen3.5 index is missing vision tensor '{tensor_name}'")]
    MissingVisionTensor { tensor_name: String },
    #[error("Qwen3.5 index contains visual tensors but config.json has no vision_config")]
    MissingVisionConfig,
    #[error(
        "Qwen3.5 vision tensor '{tensor_name}' mixes sidecar and embedded storage through '{shard_file_name}'"
    )]
    MixedVisionTensorStorage {
        tensor_name: String,
        shard_file_name: String,
    },
    #[error(
        "Qwen3.5 vision tensor '{tensor_name}' resolves outside loaded model shards through '{shard_file_name}'"
    )]
    VisionTensorOutsideModelShards {
        tensor_name: String,
        shard_file_name: String,
    },
}
