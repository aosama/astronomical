use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use thiserror::Error;

#[cfg(test)]
mod tests;

use super::tensor_spec;
use crate::artifact_validation::TensorProfile;

pub const MAXIMUM_INDEX_BYTES: usize = 1024 * 1024;
const MAXIMUM_TENSOR_NAME_BYTES: usize = 512;
/// The bounded executable tensor-to-shard inventory for a Qwen3.5 artifact.
///
/// Tracks both language model tensors (`language_model.*`) and vision tower
/// tensors (`vision_tower.*`) declared only by the main shard index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5ShardIndex {
    tensor_name_to_shard_file_name: BTreeMap<String, String>,
    /// Vision tensor name to shard file name mapping for embedded or sidecar storage.
    vision_tensor_name_to_shard_file_name: BTreeMap<String, String>,
    /// Auxiliary tensors retained in the artifact but not loaded by this runtime.
    auxiliary_tensor_name_to_shard_file_name: BTreeMap<String, String>,
    /// Stored shard spelling for every canonical tensor name the index
    /// carries. Unified-naming checkpoints (`model.language_model.*`,
    /// `model.visual.*`) map to their stored unified spelling;
    /// canonical-named tensors map to themselves. Both resolve identically
    /// through [`Self::stored_tensor_name_for`].
    stored_tensor_name_by_canonical_name: BTreeMap<String, String>,
    total_payload_bytes: u64,
    model_shard_file_names: Vec<String>,
    vision_sidecar_file_names: Vec<String>,
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
        let mut vision_tensor_name_to_shard_file_name = BTreeMap::new();
        let mut auxiliary_tensor_name_to_shard_file_name = BTreeMap::new();
        let mut stored_tensor_name_by_canonical_name = BTreeMap::new();
        let mut language_shard_file_names = BTreeSet::new();
        let mut vision_shard_file_names = BTreeSet::new();
        for (index_tensor_name, shard_file_name) in &index_document.weight_map {
            validate_tensor_name(index_tensor_name)?;
            // Tensor names outside the executable model prefixes are silently
            // skipped, exactly as before unified-naming support existed.
            let Some(canonical_tensor_name) = canonical_model_tensor_name(index_tensor_name) else {
                continue;
            };
            if canonical_tensor_name.starts_with("language_model.") {
                language_shard_file_names.insert(shard_file_name.clone());
                language_tensor_names.insert(canonical_tensor_name.clone());
                language_tensor_name_to_shard_file_name
                    .insert(canonical_tensor_name.clone(), shard_file_name.clone());
            } else if canonical_tensor_name.starts_with("vision_tower.") {
                // Embedded vision tensors may share language shards or occupy
                // dedicated vision files. A file containing only vision tensors
                // is loaded through the separate vision-tower path, regardless
                // of its filename.
                vision_shard_file_names.insert(shard_file_name.clone());
                vision_tensor_name_to_shard_file_name
                    .insert(canonical_tensor_name.clone(), shard_file_name.clone());
            } else {
                auxiliary_tensor_name_to_shard_file_name
                    .insert(canonical_tensor_name.clone(), shard_file_name.clone());
            }
            // Every canonical tensor name claims exactly one stored spelling.
            // Inserting unconditionally (canonical spellings included) lets a
            // second spelling conflict in either iteration order, because the
            // index weight-map iteration order is not controlled by us.
            let existing_stored_tensor_name = stored_tensor_name_by_canonical_name
                .insert(canonical_tensor_name.clone(), index_tensor_name.clone());
            if let Some(existing_stored_tensor_name) = existing_stored_tensor_name {
                return Err(Qwen3_5ArtifactError::ConflictingCanonicalTensorName {
                    canonical_tensor_name,
                    existing_stored_tensor_name,
                    conflicting_stored_tensor_name: index_tensor_name.clone(),
                });
            }
        }
        let language_tensor_name_references = language_tensor_names
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        tensor_spec::validate_language_tensor_names(
            &language_tensor_name_references,
            language_tensor_profiles,
        )?;

        let vision_only_shard_file_names = vision_shard_file_names
            .difference(&language_shard_file_names)
            .cloned()
            .collect::<BTreeSet<_>>();
        let vision_sidecar_file_names =
            vision_only_shard_file_names.into_iter().collect::<Vec<_>>();
        let model_shard_file_names = language_shard_file_names.into_iter().collect::<Vec<_>>();
        Ok(Self {
            tensor_name_to_shard_file_name: language_tensor_name_to_shard_file_name,
            vision_tensor_name_to_shard_file_name,
            auxiliary_tensor_name_to_shard_file_name,
            stored_tensor_name_by_canonical_name,
            total_payload_bytes,
            model_shard_file_names,
            vision_sidecar_file_names,
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

    /// Returns the vision tower tensor name to shard file name mapping.
    #[must_use]
    pub fn vision_tensor_name_to_shard_file_name(&self) -> &BTreeMap<String, String> {
        &self.vision_tensor_name_to_shard_file_name
    }

    /// Returns the MTP tensors preserved by the artifact but not loaded by this runtime.
    #[must_use]
    pub fn auxiliary_tensor_name_to_shard_file_name(&self) -> &BTreeMap<String, String> {
        &self.auxiliary_tensor_name_to_shard_file_name
    }

    /// Returns executable model shard file names in sorted order.
    ///
    /// Includes target-language files and language files with embedded vision.
    #[must_use]
    pub fn model_shard_file_names(&self) -> &[String] {
        &self.model_shard_file_names
    }

    /// Returns vision-only files that are loaded separately from language shards.
    #[must_use]
    pub fn vision_sidecar_file_names(&self) -> &[String] {
        &self.vision_sidecar_file_names
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

    /// Returns the stored shard spelling for a canonical tensor name.
    ///
    /// Every physical shard read must go through this lookup: unified-naming
    /// checkpoints store `model.language_model.*` / `model.visual.*` bytes
    /// under those exact names, while the runtime addresses tensors by their
    /// canonical `language_model.model.*` / `vision_tower.*` names.
    /// Canonical-named tensors resolve to themselves.
    #[must_use]
    pub fn stored_tensor_name_for(&self, canonical_tensor_name: &str) -> String {
        match self
            .stored_tensor_name_by_canonical_name
            .get(canonical_tensor_name)
        {
            Some(stored_tensor_name) => stored_tensor_name.clone(),
            None => canonical_tensor_name.to_owned(),
        }
    }

    /// Extracts the set of language tensor names from the safetensors index JSON
    /// without performing any validation against tensor profiles.
    ///
    /// This is used to determine which target modules are quantized vs.
    /// unquantized by checking for affine companion tensors before the full
    /// validation pass that requires complete tensor profiles.
    ///
    /// Returned names are canonical: unified-naming checkpoints contribute
    /// their normalized spellings.
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
        let mut canonical_language_tensor_names = BTreeSet::new();
        for index_tensor_name in index_document.weight_map.keys() {
            let Some(canonical_tensor_name) = canonical_model_tensor_name(index_tensor_name) else {
                continue;
            };
            if canonical_tensor_name.starts_with("language_model.") {
                canonical_language_tensor_names.insert(canonical_tensor_name);
            }
        }
        Ok(canonical_language_tensor_names)
    }

    /// Extracts the language and vision tensor names needed to resolve native
    /// modules before profile validation, without validating their profiles.
    ///
    /// Returned names are canonical: unified-naming checkpoints contribute
    /// their normalized spellings.
    pub fn extract_language_and_vision_tensor_names_from_json(
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
        let mut canonical_model_tensor_names = BTreeSet::new();
        for index_tensor_name in index_document.weight_map.keys() {
            let Some(canonical_tensor_name) = canonical_model_tensor_name(index_tensor_name) else {
                continue;
            };
            let is_model_tensor = canonical_tensor_name.starts_with("language_model.")
                || canonical_tensor_name.starts_with("vision_tower.");
            if is_model_tensor {
                canonical_model_tensor_names.insert(canonical_tensor_name);
            }
        }
        Ok(canonical_model_tensor_names)
    }
}

/// Returns the canonical executable tensor name for a shard-index tensor name.
///
/// Newer unified Hugging Face checkpoints name the language tower
/// `model.language_model.*` and the vision tower `model.visual.*`, while the
/// canonical executable layout is `language_model.model.*` and
/// `vision_tower.*`. Both spellings describe byte-identical tensors, so the
/// shard index normalizes to canonical names at parse time and
/// [`Qwen3_5ShardIndex::stored_tensor_name_for`] recovers the stored spelling
/// for physical shard reads. `None` marks a tensor name outside the
/// executable model prefixes (`language_model.*`, `vision_tower.*`, `mtp.*`),
/// which the index silently skips.
///
/// The two prefix swaps are length-neutral, which keeps every downstream
/// tensor-name length bound intact.
fn canonical_model_tensor_name(index_tensor_name: &str) -> Option<String> {
    if let Some(tower_suffix) = index_tensor_name.strip_prefix("model.language_model.") {
        return Some(format!("language_model.model.{tower_suffix}"));
    }
    if let Some(tower_suffix) = index_tensor_name.strip_prefix("model.visual.") {
        return Some(format!("vision_tower.{tower_suffix}"));
    }
    let uses_canonical_prefix = index_tensor_name.starts_with("language_model.")
        || index_tensor_name.starts_with("vision_tower.")
        || index_tensor_name.starts_with("mtp.");
    if uses_canonical_prefix {
        return Some(index_tensor_name.to_owned());
    }
    None
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
    #[error(
        "Qwen3.5 index maps stored tensors '{existing_stored_tensor_name}' and '{conflicting_stored_tensor_name}' to the same canonical tensor '{canonical_tensor_name}'"
    )]
    ConflictingCanonicalTensorName {
        canonical_tensor_name: String,
        existing_stored_tensor_name: String,
        conflicting_stored_tensor_name: String,
    },
    #[error("Qwen3.5 index is missing executable language tensor '{tensor_name}'")]
    MissingLanguageTensor { tensor_name: String },
    #[error("Qwen3.5 index contains unexpected vision tensor '{tensor_name}'")]
    UnexpectedVisionTensor { tensor_name: String },
    #[error(
        "Qwen3.5 vision module '{module_name}' has an unsupported quantization profile: {description}"
    )]
    InvalidVisionQuantizationProfile {
        module_name: String,
        description: String,
    },
    #[error("the jangtq2 tensor contract rejected module '{module_name}': {source}")]
    Jangtq2TensorContract {
        module_name: String,
        source: astronomical_jangtq2_qwen35::configuration::Jangtq2ConfigurationError,
    },
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
