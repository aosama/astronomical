use std::collections::{BTreeMap, HashMap};

use crate::artifact_validation::{
    ArtifactValidationError, TensorInventory, TensorSemanticRole, TensorSourceId,
    ValidatedRequiredFile, ValidatedSafetensorsSource, ValidatedWeightsFile,
};

use super::{Qwen3_5Config, Qwen3_5ShardIndex, Qwen3_5VisionConfig};

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
    pub(super) tensor_inventory: TensorInventory,
    pub(super) safetensors_sources: HashMap<TensorSourceId, ValidatedSafetensorsSource>,
    pub(super) source_id_by_file_name: BTreeMap<String, TensorSourceId>,
    pub(super) model_id: String,
    pub(super) revision: String,
    pub(super) max_output_tokens: u32,
    /// Where an Expert Atlas model keeps its routed experts; `None` for a
    /// stock safetensors checkpoint.
    pub(super) expert_atlas: Option<astronomical_expert_atlas::AtlasManifest>,
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
    /// The Atlas manifest when this model keeps its routed experts in Atlas
    /// layer files rather than in its shards.
    #[must_use]
    pub const fn expert_atlas(&self) -> Option<&astronomical_expert_atlas::AtlasManifest> {
        self.expert_atlas.as_ref()
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
    pub const fn tensor_inventory(&self) -> &TensorInventory {
        &self.tensor_inventory
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
    /// Bytes of every tensor the text model binds: what loading the whole
    /// text model places in memory, and what startup placement compares with
    /// the limit.
    ///
    /// The vision tower is left out because it is bound lazily and holds
    /// memory only while an image request encodes, and auxiliary tensors
    /// because the runtime never binds them. Reads the validated headers, so
    /// it answers `None`, never an undercount, when the engine has already
    /// taken a source or a header lacks a declared tensor.
    #[must_use]
    pub fn text_weight_payload_bytes(&self) -> Option<u64> {
        self.payload_bytes_for_semantic_role(TensorSemanticRole::Target)
    }

    /// Bytes of every validated tensor with `semantic_role`, or `None` when a
    /// source was already taken or a header lacks a declared tensor.
    #[must_use]
    pub fn payload_bytes_for_semantic_role(
        &self,
        semantic_role: TensorSemanticRole,
    ) -> Option<u64> {
        let mut role_payload_bytes = 0_u64;
        for tensor_location in self.tensor_inventory.locations() {
            if tensor_location.semantic_role() != semantic_role {
                continue;
            }
            let tensor_payload_bytes = self
                .safetensors_sources
                .get(&tensor_location.source_id())?
                .stored_tensor_payload_bytes(tensor_location.stored_name())?;
            role_payload_bytes = role_payload_bytes.saturating_add(tensor_payload_bytes);
        }
        Some(role_payload_bytes)
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
