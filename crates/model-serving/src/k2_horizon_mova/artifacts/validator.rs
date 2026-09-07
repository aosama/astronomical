//! Strict stacked-affine validation without executing checkpoint Python.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::artifact_validation::{
    RequiredFileProfile, read_bounded_required_file_bytes, validate_required_files,
};
use crate::k2_horizon_mova::configuration::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};

use super::dialect::K2HorizonMoVAWeightDialect;
use super::error::K2HorizonMoVAArtifactValidationError;
use super::expected_tensors::expected_stacked_affine_tensor_names;
use super::shard_index::K2HorizonMoVAShardIndex;

const MAXIMUM_INDEX_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_CHAT_TEMPLATE_BYTES: u64 = 512 * 1024;

/// Descriptor-backed ownership of a validated stacked affine family member.
#[derive(Debug)]
pub struct ValidatedK2HorizonMoVAArtifact {
    config: K2HorizonMoVAConfig,
    shard_index: K2HorizonMoVAShardIndex,
    model_directory: PathBuf,
    model_id: String,
    revision: String,
    tokenizer_bytes: Vec<u8>,
    chat_template_bytes: Vec<u8>,
    total_payload_bytes: u64,
}

impl ValidatedK2HorizonMoVAArtifact {
    #[must_use]
    pub const fn config(&self) -> &K2HorizonMoVAConfig {
        &self.config
    }
    #[must_use]
    pub const fn shard_index(&self) -> &K2HorizonMoVAShardIndex {
        &self.shard_index
    }
    #[must_use]
    pub fn model_directory(&self) -> &Path {
        &self.model_directory
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
    pub fn tokenizer_bytes(&self) -> &[u8] {
        &self.tokenizer_bytes
    }
    #[must_use]
    pub fn chat_template_bytes(&self) -> &[u8] {
        &self.chat_template_bytes
    }
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_index.shard_count()
    }
}

/// Validates stacked MLX affine K2 Horizon MoVA directories.
#[derive(Debug, Default)]
pub struct K2HorizonMoVAArtifactValidator;

impl K2HorizonMoVAArtifactValidator {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validates required files, family knobs, dialect, and shard presence.
    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
    ) -> Result<ValidatedK2HorizonMoVAArtifact, K2HorizonMoVAArtifactValidationError> {
        let model_directory = model_directory.as_ref();
        if !model_directory.is_dir() {
            return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "selected K2 Horizon MoVA directory is not a folder".to_owned(),
            });
        }
        let required_files = validate_required_files(
            model_directory,
            &[
                required_file("config.json"),
                required_file("model.safetensors.index.json"),
                required_file("tokenizer.json"),
                required_file("chat_template.jinja"),
            ],
        )?;
        let config_bytes = required_files
            .get("config.json")
            .and_then(|required_file| required_file.captured_bytes())
            .ok_or(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "config.json is required".to_owned(),
            })?;
        let config = K2HorizonMoVAConfig::from_json_bytes(config_bytes)?;
        let index_file = required_files.get("model.safetensors.index.json").ok_or(
            K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "model.safetensors.index.json is required".to_owned(),
            },
        )?;
        let index_bytes = read_bounded_required_file_bytes(index_file, MAXIMUM_INDEX_BYTES)?;
        let shard_index = K2HorizonMoVAShardIndex::from_json_bytes(&index_bytes)?;
        let dialect = K2HorizonMoVAWeightDialect::from_tensor_names(shard_index.tensor_names());
        let has_sparse_layer = config
            .layer_kinds()
            .iter()
            .any(|layer_kind| *layer_kind != K2HorizonMoVALayerKind::Dense);
        match dialect {
            K2HorizonMoVAWeightDialect::UnstackedPerExpert => {
                return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: "unstacked per-expert tensors are not an executable dialect"
                        .to_owned(),
                });
            }
            K2HorizonMoVAWeightDialect::Unknown if has_sparse_layer => {
                return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: "sparse K2 Horizon MoVA artifacts must use stacked affine experts"
                        .to_owned(),
                });
            }
            K2HorizonMoVAWeightDialect::StackedAffine | K2HorizonMoVAWeightDialect::Unknown => {}
        }
        let present_names = shard_index.tensor_names().collect::<HashSet<_>>();
        for expected_name in expected_stacked_affine_tensor_names(&config) {
            if !present_names.contains(expected_name.as_str()) {
                return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: format!("stacked affine artifact is missing {expected_name}"),
                });
            }
        }
        if shard_index.shard_count() == 0 {
            return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "stacked affine artifact must declare at least one shard".to_owned(),
            });
        }
        let mut total_payload_bytes = 0_u64;
        for shard_file_name in shard_index.shard_file_names() {
            let shard_path = model_directory.join(shard_file_name);
            let shard_metadata = fs::metadata(&shard_path).map_err(|_| {
                K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: "indexed safetensors shard is missing".to_owned(),
                }
            })?;
            if !shard_metadata.is_file() || shard_metadata.len() == 0 {
                return Err(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: "indexed safetensors shard is empty or not a file".to_owned(),
                });
            }
            total_payload_bytes = total_payload_bytes
                .checked_add(shard_metadata.len())
                .ok_or(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                    description: "shard payload byte count overflowed".to_owned(),
                })?;
        }
        let tokenizer_bytes = required_files
            .get("tokenizer.json")
            .and_then(|required_file| required_file.captured_bytes())
            .ok_or(K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "tokenizer.json is required".to_owned(),
            })?
            .to_vec();
        let chat_template_file = required_files.get("chat_template.jinja").ok_or(
            K2HorizonMoVAArtifactValidationError::InvalidArtifact {
                description: "chat_template.jinja is required".to_owned(),
            },
        )?;
        let chat_template_bytes =
            read_bounded_required_file_bytes(chat_template_file, MAXIMUM_CHAT_TEMPLATE_BYTES)?;
        let model_id = model_directory
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "k2_horizon_mova".to_owned());
        Ok(ValidatedK2HorizonMoVAArtifact {
            config,
            shard_index,
            model_directory: model_directory.to_path_buf(),
            model_id,
            revision: derive_revision_from_config_bytes(config_bytes),
            tokenizer_bytes,
            chat_template_bytes,
            total_payload_bytes,
        })
    }
}

fn required_file(file_name: &str) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: file_name.to_owned(),
        size_bytes: 0,
    }
}

fn derive_revision_from_config_bytes(config_bytes: &[u8]) -> String {
    let mut sha256_hasher = Sha256::new();
    sha256_hasher.update(config_bytes);
    let config_hash = sha256_hasher.finalize();
    format!(
        "{:012x}",
        u64::from_be_bytes(config_hash[..8].try_into().unwrap_or([0u8; 8]))
    )
}
