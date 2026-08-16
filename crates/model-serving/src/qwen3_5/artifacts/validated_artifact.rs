use std::collections::{BTreeMap, HashMap};

use crate::artifact_validation::{
    ArtifactValidationError, TensorInventory, TensorSourceId, ValidatedRequiredFile,
    ValidatedSafetensorsSource, ValidatedWeightsFile,
};

use super::{Qwen3_5Config, Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex, Qwen3_5VisionConfig};

/// Descriptor-backed validated ownership of the complete Qwen3.5 artifact.
#[derive(Debug)]
pub struct ValidatedQwen3_5Artifact {
    pub(super) config: Qwen3_5Config,
    pub(super) vision_config: Option<Qwen3_5VisionConfig>,
    pub(super) required_files: HashMap<String, ValidatedRequiredFile>,
    pub(super) shard_index: Qwen3_5ShardIndex,
    pub(super) total_payload_bytes: u64,
    pub(super) has_separate_vision_sidecar: bool,
    pub(super) has_validated_vision_tower: bool,
    pub(super) mtp_artifact_capability: Qwen3_5MtpArtifactCapability,
    pub(super) tensor_inventory: TensorInventory,
    pub(super) safetensors_sources: HashMap<TensorSourceId, ValidatedSafetensorsSource>,
    pub(super) source_id_by_file_name: BTreeMap<String, TensorSourceId>,
    pub(super) mtp_sidecar_file_name: Option<String>,
    pub(super) model_id: String,
    pub(super) revision: String,
    pub(super) max_output_tokens: u32,
}

impl ValidatedQwen3_5Artifact {
    #[must_use]
    pub const fn config(&self) -> &Qwen3_5Config {
        &self.config
    }
    #[must_use]
    pub const fn vision_config(&self) -> Option<&Qwen3_5VisionConfig> {
        self.vision_config.as_ref()
    }
    #[must_use]
    pub const fn supports_image_input(&self) -> bool {
        self.has_validated_vision_tower
    }
    #[must_use]
    pub const fn shard_index(&self) -> &Qwen3_5ShardIndex {
        &self.shard_index
    }
    #[must_use]
    pub const fn has_separate_vision_sidecar(&self) -> bool {
        self.has_separate_vision_sidecar
    }
    #[must_use]
    pub const fn mtp_artifact_capability(&self) -> &Qwen3_5MtpArtifactCapability {
        &self.mtp_artifact_capability
    }
    #[must_use]
    pub const fn tensor_inventory(&self) -> &TensorInventory {
        &self.tensor_inventory
    }
    #[must_use]
    pub fn mtp_sidecar_file_name(&self) -> Option<&str> {
        self.mtp_sidecar_file_name.as_deref()
    }
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }
    #[must_use]
    pub const fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_index.shard_count()
    }
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }
    #[must_use]
    pub fn tokenizer_bytes(&self) -> Option<&[u8]> {
        self.required_files
            .get("tokenizer.json")
            .and_then(ValidatedRequiredFile::captured_bytes)
    }
    #[must_use]
    pub fn generation_config_bytes(&self) -> Option<&[u8]> {
        self.required_files
            .get("generation_config.json")
            .and_then(ValidatedRequiredFile::captured_bytes)
    }
    /// Resolves validated source IDs once while architecture-specific file groupings are intact.
    #[doc(hidden)]
    pub fn source_ids_for_file_names(
        &self,
        file_names: &[String],
    ) -> Result<Vec<TensorSourceId>, ArtifactValidationError> {
        file_names
            .iter()
            .map(|file_name| self.source_id_for_file_name(file_name))
            .collect()
    }

    /// Resolves one architecture declaration to the opaque source used by all later transfers.
    #[doc(hidden)]
    pub fn source_id_for_file_name(
        &self,
        file_name: &str,
    ) -> Result<TensorSourceId, ArtifactValidationError> {
        self.source_id_by_file_name
            .get(file_name)
            .copied()
            .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                file_name: file_name.to_owned(),
            })
    }

    /// Transfers each already-open SafeTensors owner exactly once by opaque validated identity.
    #[doc(hidden)]
    pub fn take_safetensors_sources(
        &mut self,
        source_ids: &[TensorSourceId],
    ) -> Result<Vec<ValidatedWeightsFile>, ArtifactValidationError> {
        source_ids
            .iter()
            .map(|source_id| self.take_safetensors_source(*source_id))
            .collect()
    }

    /// Transfers one source without reopening a model-directory path or reparsing its header.
    #[doc(hidden)]
    pub fn take_safetensors_source(
        &mut self,
        source_id: TensorSourceId,
    ) -> Result<ValidatedWeightsFile, ArtifactValidationError> {
        let source = self.safetensors_sources.remove(&source_id).ok_or_else(|| {
            let file_name = self
                .source_id_by_file_name
                .iter()
                .find_map(|(file_name, candidate_source_id)| {
                    (*candidate_source_id == source_id).then(|| file_name.clone())
                })
                .unwrap_or_else(|| "unresolved SafeTensors source".to_owned());
            ArtifactValidationError::ProfileMissingRequiredFile { file_name }
        })?;
        source.into_validated_weights_file()
    }
}
