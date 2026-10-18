//! Strict pack validation for folded ternary Prism artifacts.
//!
//! The validator classifies every tensor in the single-file pack against the
//! family's shape contract, cross-checks the packed-module records against
//! the Hadamard manifest, and refuses anything it cannot serve exactly.

use std::collections::BTreeSet;
#[cfg(feature = "direct-mlx")]
use std::collections::HashMap;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(feature = "direct-mlx")]
use crate::ValidatedWeightsFile;
use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedRequiredFile,
    hugging_face_snapshot_model_id, read_bounded_required_file_bytes,
    read_raw_safetensors_inventory, validate_required_files,
};

use super::configuration::{FoldedPackConfig, FoldedPackConfigError};
use super::hadamard::{HadamardContract, HadamardContractError};

mod module_shape_contract;

#[cfg(test)]
pub(crate) mod tests;

/// The single weights file the folded pack ships instead of a shard index.
pub(crate) const PACK_WEIGHTS_FILE_NAME: &str = "model.safetensors";
/// The Hadamard manifest file name.
pub(crate) const PACK_HADAMARD_FILE_NAME: &str = "hadamard.json";
/// The pack configuration file name.
const PACK_CONFIG_FILE_NAME: &str = "config.json";
/// The tokenizer file required by the serving stack.
const PACK_TOKENIZER_FILE_NAME: &str = "tokenizer.json";
/// Byte bound applied while reading sidecar documents.
const MAXIMUM_SIDECAR_DOCUMENT_BYTES: u64 = 4 * 1024 * 1024;

/// The validated folded pack: everything the family's serving path needs to
/// load weights and apply transforms without further discovery.
#[derive(Debug)]
pub struct ValidatedFoldedPack {
    config: FoldedPackConfig,
    hadamard_contract: HadamardContract,
    // The retained weights descriptor only has an owner when the direct-MLX
    // model exists to consume it; the CPU-only build never loads weights.
    #[cfg(feature = "direct-mlx")]
    required_files: HashMap<String, ValidatedRequiredFile>,
    model_id: String,
    revision: String,
    total_payload_bytes: u64,
    max_output_tokens: u32,
}

impl ValidatedFoldedPack {
    /// The validated pack configuration.
    pub const fn config(&self) -> &FoldedPackConfig {
        &self.config
    }

    /// The validated Hadamard transform contract.
    pub const fn hadamard_contract(&self) -> &HadamardContract {
        &self.hadamard_contract
    }

    /// Transfers the pack's retained weights descriptor to its runtime owner.
    #[cfg(feature = "direct-mlx")]
    pub(crate) fn take_weights_file(
        &mut self,
    ) -> Result<ValidatedWeightsFile, FoldedPackValidationError> {
        let weights_required_file = self.required_files.remove(PACK_WEIGHTS_FILE_NAME).ok_or(
            FoldedPackValidationError::MissingPackedTensor {
                module_path: "pack weights".to_owned(),
                suffix: PACK_WEIGHTS_FILE_NAME.to_owned(),
            },
        )?;
        weights_required_file
            .into_validated_weights_file()
            .map_err(FoldedPackValidationError::Artifact)
    }

    /// The captured tokenizer document the text processor loads from.
    #[cfg(feature = "direct-mlx")]
    pub(crate) fn tokenizer_bytes(&self) -> Option<&[u8]> {
        let tokenizer_file = self.required_files.get(PACK_TOKENIZER_FILE_NAME)?;
        tokenizer_file.captured_bytes()
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    pub const fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }
}

/// Validates a folded ternary pack directory before any weight is read.
pub struct FoldedPackValidator;

impl FoldedPackValidator {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Discovers everything from the pack directory:
    /// - Required files (`config.json`, `hadamard.json`, `tokenizer.json`, `model.safetensors`)
    /// - The folded configuration and Hadamard transform contract
    /// - A per-tensor shape contract over the entire single-file inventory
    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
        max_output_tokens: u32,
    ) -> Result<ValidatedFoldedPack, FoldedPackValidationError> {
        let model_directory = model_directory.as_ref();
        if !model_directory.is_dir() {
            return Err(ArtifactValidationError::ModelDirectoryNotFound {
                model_directory: model_directory.to_path_buf(),
            }
            .into());
        }
        let required_file_profiles = [
            required_file_profile(PACK_CONFIG_FILE_NAME),
            required_file_profile(PACK_HADAMARD_FILE_NAME),
            required_file_profile(PACK_TOKENIZER_FILE_NAME),
            required_file_profile(PACK_WEIGHTS_FILE_NAME),
        ];
        let required_files = validate_required_files(model_directory, &required_file_profiles)?;

        let config_bytes = read_sidecar_bytes(&required_files[PACK_CONFIG_FILE_NAME])?;
        let config = FoldedPackConfig::from_json_bytes(&config_bytes)?;

        let hadamard_bytes = read_sidecar_bytes(&required_files[PACK_HADAMARD_FILE_NAME])?;
        let weights_inventory =
            read_raw_safetensors_inventory(&required_files[PACK_WEIGHTS_FILE_NAME])?;
        let tensor_names = weights_inventory
            .tensor_descriptors
            .iter()
            .map(|descriptor| descriptor.tensor_name.clone())
            .collect::<BTreeSet<String>>();
        let hadamard_contract = HadamardContract::from_json_bytes(&hadamard_bytes, &tensor_names)?;

        module_shape_contract::validate_pack_inventory(
            &config,
            &hadamard_contract,
            &weights_inventory,
        )?;

        let revision = derive_revision_from_config_bytes(&config_bytes);
        let model_id = hugging_face_snapshot_model_id(model_directory).unwrap_or_else(|| {
            model_directory
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_owned())
        });
        let total_payload_bytes = weights_inventory.shard_payload_bytes;

        Ok(ValidatedFoldedPack {
            config,
            hadamard_contract,
            #[cfg(feature = "direct-mlx")]
            required_files,
            model_id,
            revision,
            total_payload_bytes,
            max_output_tokens,
        })
    }
}

fn read_sidecar_bytes(
    required_file: &ValidatedRequiredFile,
) -> Result<Vec<u8>, FoldedPackValidationError> {
    Ok(read_bounded_required_file_bytes(
        required_file,
        MAXIMUM_SIDECAR_DOCUMENT_BYTES,
    )?)
}

fn required_file_profile(file_name: &str) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: file_name.to_owned(),
        size_bytes: 0,
    }
}

/// Derives a hex revision from the config bytes (zero-padded to at least 12
/// characters), matching the repository's convention so cache keys change
/// when packs change.
pub(crate) fn derive_revision_from_config_bytes(config_bytes: &[u8]) -> String {
    let mut sha256_hasher = Sha256::new();
    sha256_hasher.update(config_bytes);
    let config_hash = sha256_hasher.finalize();
    format!(
        "{:012x}",
        u64::from_be_bytes(config_hash[..8].try_into().unwrap_or([0u8; 8]))
    )
}

/// A cause-preserving failure while validating a folded ternary pack.
#[derive(Debug, Error)]
pub enum FoldedPackValidationError {
    #[error("folded pack file or safetensors validation failed: {0}")]
    Artifact(#[from] ArtifactValidationError),
    #[error("folded pack config validation failed")]
    Config(#[from] FoldedPackConfigError),
    #[error("folded pack hadamard transform contract refused")]
    Hadamard(#[from] HadamardContractError),
    #[error("folded pack sidecar document exceeds the readable bound")]
    SidecarDocumentTooLarge,
    #[error("folded pack module {module_path:?} is missing its {suffix} tensor")]
    MissingPackedTensor { module_path: String, suffix: String },
    #[error(
        "folded pack tensor {tensor_name:?} declares shape {actual_shape:?} but the family contract requires {expected_shape:?}"
    )]
    TensorShapeMismatch {
        tensor_name: String,
        expected_shape: String,
        actual_shape: String,
    },
    #[error(
        "folded pack tensor {tensor_name:?} declares dtype {actual_dtype:?} but the family contract requires {expected_dtype:?}"
    )]
    TensorDtypeMismatch {
        tensor_name: String,
        expected_dtype: &'static str,
        actual_dtype: String,
    },
    #[error("folded pack tensor {tensor_name:?} is not part of the family's recognized inventory")]
    UnrecognizedLanguageTensor { tensor_name: String },
    #[error("folded pack hadamard manifest and packed-module records disagree: {detail}")]
    ContractModuleDisagreement { detail: String },
    #[error(
        "folded pack module {module_path:?} declares hadamard block {module_block} but the manifest declares {manifest_block}"
    )]
    ModuleBlockMismatch {
        module_path: String,
        module_block: u32,
        manifest_block: u32,
    },
}

impl FoldedPackValidationError {
    /// Returns a bounded explanation suitable for a public model-load error.
    #[must_use]
    pub fn public_failure_reason(&self) -> String {
        match self {
            Self::Artifact(validation_error) => validation_error.public_failure_reason(),
            Self::Config(config_error) => {
                format!("folded pack config validation failed: {config_error}")
            }
            Self::Hadamard(hadamard_error) => {
                format!("folded pack hadamard transform contract refused: {hadamard_error}")
            }
            Self::SidecarDocumentTooLarge => {
                "folded pack sidecar document exceeds the readable bound".to_owned()
            }
            Self::MissingPackedTensor {
                module_path,
                suffix,
            } => format!("folded pack module {module_path} is missing its {suffix} tensor"),
            Self::TensorShapeMismatch {
                tensor_name,
                expected_shape,
                actual_shape,
            } => format!(
                "folded pack tensor {tensor_name} shape {actual_shape} does not match the family contract {expected_shape}"
            ),
            Self::TensorDtypeMismatch {
                tensor_name,
                expected_dtype,
                actual_dtype,
            } => format!(
                "folded pack tensor {tensor_name} dtype {actual_dtype} does not match the family contract {expected_dtype}"
            ),
            Self::UnrecognizedLanguageTensor { tensor_name } => {
                format!("folded pack tensor {tensor_name} is not part of the recognized inventory")
            }
            Self::ContractModuleDisagreement { detail } => format!(
                "folded pack hadamard manifest and packed-module records disagree: {detail}"
            ),
            Self::ModuleBlockMismatch {
                module_path,
                module_block,
                manifest_block,
            } => format!(
                "folded pack module {module_path} declares hadamard block {module_block} but the manifest declares {manifest_block}"
            ),
        }
    }
}
