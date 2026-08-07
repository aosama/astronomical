use std::collections::{HashMap, HashSet};
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    hugging_face_snapshot_model_id, read_validated_required_file_bytes,
    validate_bounded_safetensors_with_exact_tensor_names, validate_required_file,
    validate_required_files,
};

use super::{
    DeepSeekV4ConfigError, DeepSeekV4DsparkArtifactCapability, DeepSeekV4Flash0731Config,
    DeepSeekV4ShardIndex, DeepSeekV4ShardIndexError,
};

const MAXIMUM_INDEX_BYTES: u64 = 1024 * 1024;

/// Validates the selected DeepSeek-V4 Flash artifact before a future MLX load.
#[derive(Debug, Default)]
pub struct DeepSeekV4ArtifactValidator;

impl DeepSeekV4ArtifactValidator {
    /// Creates a DeepSeek-V4 Flash structural artifact validator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validates retained core files, index ownership, and bounded shard headers.
    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
    ) -> Result<ValidatedDeepSeekV4Artifact, DeepSeekV4ArtifactValidationError> {
        let model_directory = model_directory.as_ref();
        if !model_directory.is_dir() {
            return Err(ArtifactValidationError::ModelDirectoryNotFound {
                model_directory: model_directory.to_path_buf(),
            }
            .into());
        }
        let mut required_files = validate_required_files(
            model_directory,
            &[
                required_file("config.json"),
                required_file("tokenizer.json"),
                required_file("model.safetensors.index.json"),
            ],
        )?;
        let config_bytes = captured_required_file_bytes(&required_files, "config.json")?.to_vec();
        let config = DeepSeekV4Flash0731Config::from_json_bytes(&config_bytes)?;
        let revision = derive_revision_from_config_bytes(&config_bytes);
        let index_required_file = required_files
            .get("model.safetensors.index.json")
            .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                file_name: "model.safetensors.index.json".to_owned(),
            })?;
        let index_bytes =
            read_validated_required_file_bytes(index_required_file, MAXIMUM_INDEX_BYTES)?;
        let shard_index = DeepSeekV4ShardIndex::from_json_bytes(
            &index_bytes,
            config.dspark_artifact_capability(),
        )?;
        for shard_file_name in shard_index.shard_file_names() {
            let validated_shard =
                validate_required_file(model_directory, &required_file(shard_file_name))?;
            required_files.insert(shard_file_name.clone(), validated_shard);
        }
        for shard_file_name in shard_index.shard_file_names() {
            let validated_shard = required_files.get(shard_file_name).ok_or_else(|| {
                ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: shard_file_name.clone(),
                }
            })?;
            let shard_tensor_names = shard_index
                .tensor_names_for_shard(shard_file_name)
                .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: shard_file_name.clone(),
                })?;
            let accepted_tensor_names = shard_tensor_names
                .iter()
                .map(String::as_str)
                .collect::<HashSet<_>>();
            validate_bounded_safetensors_with_exact_tensor_names(
                validated_shard.file(),
                validated_shard.size_bytes(),
                shard_file_name,
                &accepted_tensor_names,
            )?;
        }
        let model_id = hugging_face_snapshot_model_id(model_directory).unwrap_or_else(|| {
            model_directory
                .file_name()
                .map(|directory_name| directory_name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_owned())
        });
        Ok(ValidatedDeepSeekV4Artifact {
            config,
            shard_index,
            required_files,
            model_id,
            revision,
        })
    }
}

/// Descriptor-backed structural ownership of a DeepSeek-V4 Flash artifact.
#[derive(Debug)]
pub struct ValidatedDeepSeekV4Artifact {
    config: DeepSeekV4Flash0731Config,
    shard_index: DeepSeekV4ShardIndex,
    required_files: HashMap<String, ValidatedRequiredFile>,
    model_id: String,
    revision: String,
}

impl ValidatedDeepSeekV4Artifact {
    /// Returns the validated selected-architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &DeepSeekV4Flash0731Config {
        &self.config
    }

    /// Returns the artifact’s declared target-only or DSpark capability.
    #[must_use]
    pub const fn dspark_artifact_capability(&self) -> &DeepSeekV4DsparkArtifactCapability {
        self.config.dspark_artifact_capability()
    }

    /// Returns the discovered model identity.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the config-derived revision identity.
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns retained tokenizer bytes for a later tokenizer owner.
    #[must_use]
    pub fn tokenizer_bytes(&self) -> Option<&[u8]> {
        self.required_files
            .get("tokenizer.json")
            .and_then(ValidatedRequiredFile::captured_bytes)
    }

    /// Returns the index-declared safetensors payload size.
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.shard_index.total_payload_bytes()
    }

    /// Transfers all validated target shard descriptors in index order.
    pub fn into_shard_files(
        mut self,
    ) -> Result<Vec<ValidatedWeightsFile>, ArtifactValidationError> {
        let mut shard_files = Vec::with_capacity(self.shard_index.shard_file_names().len());
        for shard_file_name in self.shard_index.shard_file_names() {
            let required_file = self.required_files.remove(shard_file_name).ok_or_else(|| {
                ArtifactValidationError::ProfileMissingRequiredFile {
                    file_name: shard_file_name.clone(),
                }
            })?;
            shard_files.push(required_file.into_validated_weights_file()?);
        }
        Ok(shard_files)
    }
}

fn required_file(file_name: &str) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: file_name.to_owned(),
        size_bytes: 0,
    }
}

fn captured_required_file_bytes<'a>(
    required_files: &'a HashMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<&'a [u8], ArtifactValidationError> {
    required_files
        .get(file_name)
        .and_then(ValidatedRequiredFile::captured_bytes)
        .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
            file_name: file_name.to_owned(),
        })
}

fn derive_revision_from_config_bytes(config_bytes: &[u8]) -> String {
    let mut sha256_hasher = Sha256::new();
    sha256_hasher.update(config_bytes);
    let config_hash = sha256_hasher.finalize();
    format!(
        "{:012x}",
        u64::from_be_bytes(config_hash[..8].try_into().unwrap_or([0_u8; 8]))
    )
}

/// A cause-preserving DeepSeek-V4 Flash structural validation failure.
#[derive(Debug, Error)]
pub enum DeepSeekV4ArtifactValidationError {
    /// Required files or bounded safetensors validation failed.
    #[error("DeepSeek-V4 file or safetensors validation failed")]
    Artifact(#[from] ArtifactValidationError),
    /// The selected DeepSeek-V4 configuration was incompatible.
    #[error("DeepSeek-V4 config validation failed")]
    Config(#[from] DeepSeekV4ConfigError),
    /// The selected DeepSeek-V4 index was incompatible.
    #[error("DeepSeek-V4 shard-index validation failed")]
    ShardIndex(#[from] DeepSeekV4ShardIndexError),
}
