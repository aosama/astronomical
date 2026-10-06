use std::collections::HashMap;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedSafetensorsSource,
    hugging_face_snapshot_model_id, validate_required_file, validate_required_files,
};

use super::artifact_helpers::{
    captured_required_file_bytes, read_required_file_bytes, required_file,
};
use super::artifact_inventory;
use super::tensor_spec;
use super::validated_artifact::ValidatedQwen3_5Artifact;
use super::vision_tensor_spec::qwen3_5_vision_tensor_profiles;
use super::vision_validation;
use super::{
    OptiQMetadata, OptiQMetadataError, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5VisionConfig,
};
use super::{Qwen3_5ArtifactError, Qwen3_5ShardIndex};

/// Validates the complete Qwen3.5 artifact before any native allocation.
///
/// Everything is discovered from the model directory. The config, shard index, and tokenizer are the sole sources of truth.
#[derive(Debug, Default)]
pub struct Qwen3_5ArtifactValidator;

impl Qwen3_5ArtifactValidator {
    /// Creates the validator for Qwen3.5 model artifacts.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validates required file structure, config, index, and bounded model shard headers.
    ///
    /// Discovers everything from the model directory:
    /// - Required files (`config.json`, `tokenizer.json`, `model.safetensors.index.json`, shards)
    /// - Shard names from the safetensors index
    /// - Vision sidecar presence from the index
    /// - Model ID from the leaf directory name
    /// - Revision from a SHA-256 hash of `config.json` bytes
    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
        max_output_tokens: u32,
    ) -> Result<ValidatedQwen3_5Artifact, Qwen3_5ArtifactValidationError> {
        let model_directory = model_directory.as_ref();
        if !model_directory.is_dir() {
            return Err(ArtifactValidationError::ModelDirectoryNotFound {
                model_directory: model_directory.to_path_buf(),
            }
            .into());
        }

        // A converted per-expert streaming revision declares itself with
        // manifest.json (format version 3) and carries no shard index; it
        // validates against its own manifest and resident weight bundle.
        if model_directory.join("manifest.json").is_file() {
            return Self::validate_streaming_revision(model_directory, max_output_tokens);
        }

        // Build required file profiles for the core config files.
        // Shard files are discovered later from the safetensors index.
        let mut required_file_profiles = vec![
            required_file("config.json"),
            required_file("model.safetensors.index.json"),
            required_file("tokenizer.json"),
        ];

        // optiq_metadata.json is optional — validate it if present.
        let optiq_metadata_path = model_directory.join("optiq_metadata.json");
        if optiq_metadata_path.is_file() {
            required_file_profiles.push(required_file("optiq_metadata.json"));
        }
        if model_directory.join("generation_config.json").is_file() {
            required_file_profiles.push(required_file("generation_config.json"));
        }

        let required_files = validate_required_files(model_directory, &required_file_profiles)?;

        // Read config.json and derive the revision hash from its bytes.
        let config_bytes = captured_required_file_bytes(&required_files, "config.json")?;
        let revision = derive_revision_from_config_bytes(config_bytes);

        let mut config = Qwen3_5Config::from_json_bytes(config_bytes)?;
        let vision_config = Qwen3_5VisionConfig::from_optional_json_bytes(config_bytes)?;

        // Validate optiq_metadata.json if present.
        if let Some(optiq_metadata_required_file) = required_files.get("optiq_metadata.json") {
            let optiq_metadata_bytes = read_required_file_bytes(optiq_metadata_required_file)?;
            OptiQMetadata::from_json_bytes(&optiq_metadata_bytes)?
                .validate_against_config(&config)?;
        }

        // Read the shard index to discover shard names, tensor names, and
        // resolve which modules are quantized vs. stored as bfloat16.
        let shard_index_bytes = read_required_file_bytes(
            required_files
                .get("model.safetensors.index.json")
                .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: "model.safetensors.index.json".to_owned(),
                })?,
        )?;
        let canonical_tensor_names =
            Qwen3_5ShardIndex::extract_language_tensor_names_from_json(&shard_index_bytes)?;
        config.resolve_unquantized_modules_from_shard_index(&canonical_tensor_names);
        let language_tensor_profiles = tensor_spec::qwen3_5_language_tensor_profiles(&config);
        let shard_index =
            Qwen3_5ShardIndex::from_json_bytes(&shard_index_bytes, &language_tensor_profiles)?;
        let validated_vision_tower_storage = vision_validation::validate_vision_tower_inventory(
            &shard_index,
            vision_config.as_ref(),
        )?;
        let has_separate_vision_sidecar = validated_vision_tower_storage.has_separate_sidecar();
        let vision_tensor_profiles = vision_config
            .as_ref()
            .map(qwen3_5_vision_tensor_profiles)
            .unwrap_or_default();
        let tensor_inventory = artifact_inventory::build_index_tensor_inventory(&shard_index)?;
        let mut recognized_tensor_profiles = language_tensor_profiles.clone();
        recognized_tensor_profiles.extend(vision_tensor_profiles.clone());
        let source_id_by_file_name = artifact_inventory::source_id_by_file_name(&shard_index)?;
        let mut safetensors_sources = HashMap::new();
        let mut total_payload_bytes = 0_u64;
        for (file_name, source_id) in &source_id_by_file_name {
            let required_file = validate_required_file(
                model_directory,
                &RequiredFileProfile {
                    file_name: file_name.clone(),
                    size_bytes: 0,
                },
            )?;
            let source = ValidatedSafetensorsSource::parse(*source_id, required_file)?;
            source.validate_required_inventory_profiles(
                &tensor_inventory,
                &recognized_tensor_profiles,
            )?;
            total_payload_bytes = total_payload_bytes
                .checked_add(source.payload_bytes())
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
            safetensors_sources.insert(*source_id, source);
        }

        // Derive model_id from the leaf directory name.
        let model_id = hugging_face_snapshot_model_id(model_directory).unwrap_or_else(|| {
            model_directory
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_owned())
        });

        Ok(ValidatedQwen3_5Artifact {
            config,
            vision_config,
            required_files,
            shard_index,
            total_payload_bytes,
            has_separate_vision_sidecar,
            has_validated_vision_tower: validated_vision_tower_storage.has_validated_vision_tower(),
            tensor_inventory,
            safetensors_sources,
            source_id_by_file_name,
            model_id,
            revision,
            max_output_tokens,
        })
    }
}

/// Derives a 12-character hex revision string from the SHA-256 hash of config.json bytes.
/// This ensures prompt cache blocks are invalidated when the model config changes.
pub(super) fn derive_revision_from_config_bytes(config_bytes: &[u8]) -> String {
    let mut sha256_hasher = Sha256::new();
    sha256_hasher.update(config_bytes);
    let config_hash = sha256_hasher.finalize();
    format!(
        "{:012x}",
        u64::from_be_bytes(config_hash[..8].try_into().unwrap_or([0u8; 8]))
    )
}

/// A cause-preserving failure while validating the complete Qwen3.5 artifact.
#[derive(Debug, Error)]
pub enum Qwen3_5ArtifactValidationError {
    #[error("Qwen3.5 file or safetensors validation failed: {0}")]
    Artifact(#[from] ArtifactValidationError),
    #[error("Qwen3.5 config validation failed")]
    Config(#[from] Qwen3_5ConfigError),
    #[error("Qwen3.5 OptiQ metadata validation failed")]
    OptiQMetadata(#[from] OptiQMetadataError),
    #[error("Qwen3.5 shard-index validation failed")]
    Qwen3_5ShardIndex(#[from] Qwen3_5ArtifactError),
}

impl Qwen3_5ArtifactValidationError {
    /// Returns a bounded explanation suitable for a public model-load error.
    #[must_use]
    pub fn public_failure_reason(&self) -> String {
        match self {
            Self::Artifact(validation_error) => bound_public_qwen3_5_failure_reason(format!(
                "Qwen3.5 artifact validation failed: {}",
                validation_error.public_failure_reason()
            )),
            Self::Config(config_error) => {
                format!("Qwen3.5 config validation failed: {config_error}")
            }
            Self::OptiQMetadata(metadata_error) => {
                format!("Qwen3.5 OptiQ metadata validation failed: {metadata_error}")
            }
            Self::Qwen3_5ShardIndex(shard_index_error) => {
                format!("Qwen3.5 shard-index validation failed: {shard_index_error}")
            }
        }
    }
}

fn bound_public_qwen3_5_failure_reason(unbounded_reason: String) -> String {
    const MAX_PUBLIC_REASON_CHARACTERS: usize = 512;
    let mut bounded_reason = unbounded_reason
        .replace('/', "_")
        .replace('\\', "_")
        .chars()
        .take(MAX_PUBLIC_REASON_CHARACTERS)
        .collect::<String>();
    if unbounded_reason.chars().count() > MAX_PUBLIC_REASON_CHARACTERS {
        bounded_reason.pop();
        bounded_reason.push('…');
    }
    bounded_reason
}
