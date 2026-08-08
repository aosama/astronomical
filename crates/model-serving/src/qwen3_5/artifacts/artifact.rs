use std::collections::HashMap;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorProfile, hugging_face_snapshot_model_id,
    validate_required_file, validate_required_files,
};
use crate::artifact_validation::{ValidatedRequiredFile, ValidatedWeightsFile};

use super::Qwen3_5MtpArtifactCapability;
use super::artifact_helpers::{
    captured_required_file_bytes, read_required_file_bytes, recognized_tensor_names, required_file,
};
use super::tensor_spec::qwen3_5_language_tensor_profiles;
use super::vision_tensor_spec::qwen3_5_vision_tensor_profiles;
use super::vision_validation::{
    embedded_vision_tensor_profiles_for_shard, validate_vision_sidecars,
    validate_vision_tower_inventory,
};
use super::{
    OptiQMetadata, OptiQMetadataError, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5VisionConfig,
};
use super::{Qwen3_5ArtifactError, Qwen3_5ShardIndex};
use crate::qwen3_5::multi_token_prediction::qwen3_5_mtp_tensor_profiles;

/// Validates the complete Qwen3.5 artifact before any native allocation.
///
/// Everything is discovered from the model directory. No hardcoded model profiles
/// or certification checks — the config, shard index, and tokenizer are the
/// sole sources of truth.
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

        let mut required_files = validate_required_files(model_directory, &required_file_profiles)?;

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
        let shard_tensor_names =
            Qwen3_5ShardIndex::extract_language_tensor_names_from_json(&shard_index_bytes)?;
        config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);
        let language_tensor_profiles = qwen3_5_language_tensor_profiles(&config);
        let mut shard_index =
            Qwen3_5ShardIndex::from_json_bytes(&shard_index_bytes, &language_tensor_profiles)?;
        let absent_optional_mtp_shard_file_names = shard_index
            .mtp_only_shard_file_names()
            .iter()
            .filter(|shard_file_name| !model_directory.join(shard_file_name).is_file())
            .cloned()
            .collect::<Vec<_>>();
        for absent_optional_mtp_shard_file_name in absent_optional_mtp_shard_file_names {
            shard_index.omit_optional_mtp_shard_file(&absent_optional_mtp_shard_file_name);
        }
        let mtp_artifact_capability =
            Qwen3_5MtpArtifactCapability::from_shard_index(&config, &shard_index);
        let validated_vision_tower_storage =
            validate_vision_tower_inventory(&shard_index, vision_config.as_ref())?;
        let has_separate_vision_sidecar = validated_vision_tower_storage.has_separate_sidecar();
        let vision_tensor_profiles = vision_config
            .as_ref()
            .map(qwen3_5_vision_tensor_profiles)
            .unwrap_or_default();
        let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&config);
        let mut recognized_tensor_profiles = language_tensor_profiles.clone();
        recognized_tensor_profiles.extend(mtp_tensor_profiles.clone());
        recognized_tensor_profiles.extend(vision_tensor_profiles.clone());
        let recognized_tensor_names = recognized_tensor_names(&shard_index);

        // Validate and register the shard files discovered from the index.
        for shard_file_name in shard_index
            .model_shard_file_names()
            .iter()
            .chain(shard_index.mtp_only_shard_file_names())
        {
            let shard_path = model_directory.join(shard_file_name);
            if !shard_path.is_file() {
                return Err(ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: shard_file_name.clone(),
                }
                .into());
            }
            let shard_profile = RequiredFileProfile {
                file_name: shard_file_name.clone(),
                size_bytes: 0,
            };
            let validated_shard = validate_required_file(model_directory, &shard_profile)?;
            required_files.insert(shard_file_name.clone(), validated_shard);
        }

        // If the model has a separate vision sidecar, validate and register it.
        for vision_sidecar_file_name in shard_index.vision_sidecar_file_names() {
            let vision_sidecar_path = model_directory.join(vision_sidecar_file_name);
            if !vision_sidecar_path.is_file() {
                return Err(ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: vision_sidecar_file_name.clone(),
                }
                .into());
            }
            let vision_required_file_profile = RequiredFileProfile {
                file_name: vision_sidecar_file_name.clone(),
                size_bytes: 0,
            };
            let vision_validated_file =
                validate_required_file(model_directory, &vision_required_file_profile)?;
            required_files.insert(vision_sidecar_file_name.clone(), vision_validated_file);
        }

        // Validate each executable model shard's safetensors headers.
        let mut total_payload_bytes = 0_u64;
        for shard_file_name in shard_index
            .model_shard_file_names()
            .iter()
            .chain(shard_index.mtp_only_shard_file_names())
        {
            let shard_file = required_files
                .get(shard_file_name.as_str())
                .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: shard_file_name.clone(),
                })?;
            let shard_language_tensor_names =
                shard_index.language_tensor_names_for_shard(shard_file_name);
            let mut profiled_tensors_for_shard: Vec<TensorProfile> = language_tensor_profiles
                .iter()
                .filter(|tensor_profile| {
                    shard_language_tensor_names.contains(&tensor_profile.name.as_str())
                })
                .cloned()
                .collect();
            profiled_tensors_for_shard.extend(embedded_vision_tensor_profiles_for_shard(
                &vision_tensor_profiles,
                &shard_index,
                shard_file_name,
            ));
            if mtp_artifact_capability.is_mtp_capable() {
                profiled_tensors_for_shard.extend(
                    mtp_tensor_profiles
                        .iter()
                        .filter(|tensor_profile| {
                            shard_index.shard_file_name_for_mtp_tensor(&tensor_profile.name)
                                == Some(shard_file_name)
                        })
                        .cloned(),
                );
            }
            let shard_metadata =
                crate::artifact_validation::validate_bounded_safetensors_with_indexed_profiles(
                    shard_file.file(),
                    shard_file.size_bytes(),
                    shard_file_name,
                    &profiled_tensors_for_shard,
                    &recognized_tensor_profiles,
                    &recognized_tensor_names,
                )?;
            total_payload_bytes = total_payload_bytes
                .checked_add(shard_metadata.total_payload_bytes)
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
        }

        if has_separate_vision_sidecar {
            validate_vision_sidecars(
                &required_files,
                &shard_index,
                &vision_tensor_profiles,
                &recognized_tensor_profiles,
                &recognized_tensor_names,
            )?;
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
            mtp_artifact_capability,
            model_id,
            revision,
            max_output_tokens,
        })
    }
}

/// Derives a 12-character hex revision string from the SHA-256 hash of config.json bytes.
/// This ensures prompt cache blocks are invalidated when the model config changes.
fn derive_revision_from_config_bytes(config_bytes: &[u8]) -> String {
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
    #[error("Qwen3.5 file or safetensors validation failed")]
    Artifact(#[from] ArtifactValidationError),
    #[error("Qwen3.5 config validation failed")]
    Config(#[from] Qwen3_5ConfigError),
    #[error("Qwen3.5 OptiQ metadata validation failed")]
    OptiQMetadata(#[from] OptiQMetadataError),
    #[error("Qwen3.5 shard-index validation failed")]
    Qwen3_5ShardIndex(#[from] Qwen3_5ArtifactError),
}

/// Descriptor-backed validated ownership of the complete Qwen3.5 artifact.
#[derive(Debug)]
pub struct ValidatedQwen3_5Artifact {
    config: Qwen3_5Config,
    vision_config: Option<Qwen3_5VisionConfig>,
    required_files: HashMap<String, ValidatedRequiredFile>,
    shard_index: Qwen3_5ShardIndex,
    total_payload_bytes: u64,
    /// Whether this model has a complete separate vision sidecar file.
    /// When false, the validated visual tower is embedded or absent.
    has_separate_vision_sidecar: bool,
    /// Whether the complete visual tower has validated physical weights.
    has_validated_vision_tower: bool,
    /// MTP capability discovered from validated tensor inventory.
    mtp_artifact_capability: Qwen3_5MtpArtifactCapability,
    /// Discovered model identity from the leaf directory name.
    model_id: String,
    /// SHA-256 hash of config.json bytes (12 hex chars).
    revision: String,
    /// Per-request output-token ceiling from user config or default.
    max_output_tokens: u32,
}

impl ValidatedQwen3_5Artifact {
    #[must_use]
    pub const fn config(&self) -> &Qwen3_5Config {
        &self.config
    }

    /// Returns the validated Qwen3.5 vision configuration.
    #[must_use]
    pub const fn vision_config(&self) -> Option<&Qwen3_5VisionConfig> {
        self.vision_config.as_ref()
    }

    /// Returns whether this validated artifact accepts image input.
    ///
    /// Image capability requires complete validated physical visual weights, not metadata alone.
    #[must_use]
    pub const fn supports_image_input(&self) -> bool {
        self.has_validated_vision_tower
    }

    #[must_use]
    pub const fn shard_index(&self) -> &Qwen3_5ShardIndex {
        &self.shard_index
    }

    /// Returns whether this model has a separate vision sidecar file.
    /// When false, vision tensors are embedded in model shards or absent.
    #[must_use]
    pub const fn has_separate_vision_sidecar(&self) -> bool {
        self.has_separate_vision_sidecar
    }

    /// Returns the MTP capability discovered from artifact tensor inventory.
    #[must_use]
    pub const fn mtp_artifact_capability(&self) -> &Qwen3_5MtpArtifactCapability {
        &self.mtp_artifact_capability
    }

    /// Returns the discovered model identity from the leaf directory name.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the SHA-256 hash of config.json bytes (12 hex chars).
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns the per-request output-token ceiling.
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

    /// Consumes the artifact and transfers all validated shard descriptors in load order.
    pub fn into_shard_files(
        mut self,
    ) -> Result<Vec<ValidatedWeightsFile>, ArtifactValidationError> {
        let shard_file_names = self.shard_index.model_shard_file_names().to_vec();
        let mut shard_files = Vec::with_capacity(shard_file_names.len());
        for shard_file_name in &shard_file_names {
            let required_file = self.required_files.remove(shard_file_name).ok_or_else(|| {
                ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: shard_file_name.clone(),
                }
            })?;
            shard_files.push(required_file.into_validated_weights_file()?);
        }
        Ok(shard_files)
    }

    /// Transfers the validated vision sidecars while retaining language shard descriptors.
    pub fn take_vision_sidecar_files(
        &mut self,
    ) -> Result<Vec<ValidatedWeightsFile>, ArtifactValidationError> {
        let vision_sidecar_file_names = self.shard_index.vision_sidecar_file_names().to_vec();
        let mut vision_sidecar_files = Vec::with_capacity(vision_sidecar_file_names.len());
        for vision_sidecar_file_name in vision_sidecar_file_names {
            let required_file = self
                .required_files
                .remove(&vision_sidecar_file_name)
                .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: vision_sidecar_file_name,
                })?;
            vision_sidecar_files.push(required_file.into_validated_weights_file()?);
        }
        Ok(vision_sidecar_files)
    }

    /// Transfers optional MTP-only files without consuming target shard descriptors.
    pub fn take_mtp_only_shard_files(
        &mut self,
    ) -> Result<Vec<ValidatedWeightsFile>, ArtifactValidationError> {
        let mtp_only_shard_file_names = self.shard_index.mtp_only_shard_file_names().to_vec();
        let mut mtp_only_shard_files = Vec::with_capacity(mtp_only_shard_file_names.len());
        for mtp_only_shard_file_name in mtp_only_shard_file_names {
            let required_file = self
                .required_files
                .remove(&mtp_only_shard_file_name)
                .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: mtp_only_shard_file_name,
                })?;
            mtp_only_shard_files.push(required_file.into_validated_weights_file()?);
        }
        Ok(mtp_only_shard_files)
    }
}
