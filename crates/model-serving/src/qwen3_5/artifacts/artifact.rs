use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorFeature, TensorSemanticRole,
    ValidatedSafetensorsSource, hugging_face_snapshot_model_id, validate_required_file,
    validate_required_files,
};

use super::artifact_helpers::{
    captured_required_file_bytes, read_required_file_bytes, required_file,
};
use super::artifact_inventory::{build_index_tensor_inventory, source_id_by_file_name};
use super::sidecar_declaration::{Qwen3_5MtpSidecarCandidate, Qwen3_5MtpSidecarDeclaration};
use super::tensor_spec::qwen3_5_language_tensor_profiles;
use super::validated_artifact::ValidatedQwen3_5Artifact;
use super::vision_tensor_spec::qwen3_5_vision_tensor_profiles;
use super::vision_validation::validate_vision_tower_inventory;
use super::{
    OptiQMetadata, OptiQMetadataError, Qwen3_5Config, Qwen3_5ConfigError, Qwen3_5VisionConfig,
};
use super::{Qwen3_5ArtifactError, Qwen3_5ShardIndex};
use super::{
    Qwen3_5MtpArtifactCapability, Qwen3_5MtpContract, Qwen3_5MtpContractError,
    Qwen3_5MtpTargetOnlyReason,
};
use crate::qwen3_5::multi_token_prediction::qwen3_5_mtp_tensor_profiles;

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
        let mtp_contract = parse_optional_mtp_contract(model_directory, config_bytes);

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
        let mut canonical_tensor_names =
            Qwen3_5ShardIndex::extract_language_tensor_names_from_json(&shard_index_bytes)?;
        let sidecar_declaration = config.sidecar_mtp_file().and_then(|relative_path| {
            tracing::debug!(
                declared_sidecar = relative_path,
                "attempting to parse MTP sidecar declaration"
            );
            Qwen3_5MtpSidecarDeclaration::parse(relative_path)
                .inspect_err(|_| {
                    tracing::debug!(
                        "optional MTP sidecar declaration is invalid; serving target-only"
                    )
                })
                .ok()
        });
        let sidecar_candidate = sidecar_declaration.as_ref().and_then(|declaration| {
            Qwen3_5MtpSidecarCandidate::open(model_directory, declaration)
                .inspect_err(|error| {
                    tracing::debug!(
                        sidecar_path = declaration.relative_path(),
                        error = %error,
                        "optional MTP sidecar source is unavailable; serving target-only"
                    )
                })
                .ok()
        });
        let has_unavailable_declared_sidecar =
            config.sidecar_mtp_file().is_some() && sidecar_candidate.is_none();
        if let Some(candidate) = sidecar_candidate.as_ref() {
            canonical_tensor_names.extend(candidate.canonical_names().cloned());
        }
        config.resolve_unquantized_modules_from_shard_index(&canonical_tensor_names);
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
        let validated_vision_tower_storage =
            validate_vision_tower_inventory(&shard_index, vision_config.as_ref())?;
        let has_separate_vision_sidecar = validated_vision_tower_storage.has_separate_sidecar();
        let vision_tensor_profiles = vision_config
            .as_ref()
            .map(qwen3_5_vision_tensor_profiles)
            .unwrap_or_default();
        let mut mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&config);
        // Packed switch_mlp profiles describe the resident MTP expert layout.
        // A sidecar that stores per-expert 2D tensors omits those names; drop the
        // packed profiles so validation does not require a layout the sidecar
        // does not use. Extra sidecar tensors remain accepted.
        if let Some(candidate) = sidecar_candidate.as_ref() {
            let sidecar_canonical_names = candidate
                .canonical_names()
                .cloned()
                .collect::<BTreeSet<_>>();
            mtp_tensor_profiles.retain(|profile| {
                !profile.name.contains(".mlp.switch_mlp.")
                    || sidecar_canonical_names.contains(&profile.name)
            });
        }
        let embedded_mtp_names = shard_index
            .mtp_tensor_name_to_shard_file_name()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let has_sidecar_collision = sidecar_candidate.as_ref().is_some_and(|candidate| {
            candidate
                .canonical_names()
                .any(|name| embedded_mtp_names.contains(name))
        });
        let had_sidecar_candidate = sidecar_candidate.is_some();
        let mut sidecar_validation_diagnostic: Option<String> = None;
        let validated_mtp_sidecar = if has_sidecar_collision {
            tracing::debug!(
                "optional MTP canonical tensor collision detected; serving target-only"
            );
            None
        } else {
            sidecar_candidate.and_then(|candidate| {
                match candidate.validate(&mtp_tensor_profiles, &embedded_mtp_names) {
                    Ok(validated) => Some(validated),
                    Err(error) => {
                        sidecar_validation_diagnostic = Some(error.to_string());
                        tracing::debug!(
                            error = %error,
                            "optional MTP sidecar inventory failed validation; serving target-only"
                        );
                        None
                    }
                }
            })
        };
        let has_invalid_declared_sidecar = config.sidecar_mtp_file().is_some()
            && had_sidecar_candidate
            && validated_mtp_sidecar.is_none();
        let mut tensor_inventory = build_index_tensor_inventory(&shard_index)?;
        if !has_sidecar_collision && let Some(sidecar) = validated_mtp_sidecar.as_ref() {
            for location in sidecar.inventory.locations().cloned() {
                tensor_inventory.insert(location).map_err(|_| {
                    ArtifactValidationError::UnexpectedTensor {
                        tensor_name: "optional MTP canonical collision".to_owned(),
                    }
                })?;
            }
        }
        let canonical_mtp_names = tensor_inventory
            .locations()
            .filter(|location| location.semantic_role() == TensorSemanticRole::MultiTokenPrediction)
            .map(|location| location.canonical_name().to_owned())
            .collect::<BTreeSet<_>>();
        let mut mtp_artifact_capability = if has_sidecar_collision {
            Qwen3_5MtpArtifactCapability::target_only(
                Qwen3_5MtpTargetOnlyReason::CanonicalTensorCollision,
            )
        } else if has_unavailable_declared_sidecar {
            Qwen3_5MtpArtifactCapability::target_only(
                Qwen3_5MtpTargetOnlyReason::SidecarUnavailable,
            )
        } else if has_invalid_declared_sidecar {
            Qwen3_5MtpArtifactCapability::target_only(
                Qwen3_5MtpTargetOnlyReason::SidecarValidationFailed(
                    sidecar_validation_diagnostic
                        .unwrap_or_else(|| "declared MTP sidecar failed validation".to_owned()),
                ),
            )
        } else if let Err(contract_error) = mtp_contract.as_ref() {
            Qwen3_5MtpArtifactCapability::target_only(contract_error.into())
        } else {
            Qwen3_5MtpArtifactCapability::from_canonical_tensor_names(
                &config,
                canonical_mtp_names,
                mtp_contract.as_ref().ok(),
            )
        };
        let mut recognized_tensor_profiles = language_tensor_profiles.clone();
        recognized_tensor_profiles.extend(mtp_tensor_profiles.clone());
        recognized_tensor_profiles.extend(vision_tensor_profiles.clone());
        let mut source_id_by_file_name = source_id_by_file_name(&shard_index)?;
        let mut safetensors_sources = HashMap::new();
        let mut total_payload_bytes = 0_u64;
        let mut embedded_mtp_profile_validation_failed = false;
        for (file_name, source_id) in &source_id_by_file_name {
            if shard_index.is_mtp_only_shard_file(file_name) {
                continue;
            }
            let required_file = validate_required_file(
                model_directory,
                &RequiredFileProfile {
                    file_name: file_name.clone(),
                    size_bytes: 0,
                },
            )?;
            let source = ValidatedSafetensorsSource::parse(*source_id, required_file)?;
            // Required target and vision tensors remain load-bearing even when the same physical
            // shard also contains an optional MTP head. Validate required profiles first, then
            // classify an MTP-only profile defect as target-only instead of rejecting the model.
            source.validate_required_inventory_profiles(
                &tensor_inventory,
                &recognized_tensor_profiles,
            )?;
            if source
                .validate_feature_inventory_profiles(
                    &tensor_inventory,
                    &recognized_tensor_profiles,
                    TensorFeature::MultiTokenPrediction,
                )
                .is_err()
            {
                embedded_mtp_profile_validation_failed = true;
            }
            total_payload_bytes = total_payload_bytes
                .checked_add(source.payload_bytes())
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
            safetensors_sources.insert(*source_id, source);
        }
        if embedded_mtp_profile_validation_failed {
            tracing::debug!(
                "optional embedded MTP tensor profile failed validation; serving target-only"
            );
            mtp_artifact_capability = Qwen3_5MtpArtifactCapability::target_only(
                Qwen3_5MtpTargetOnlyReason::TensorValidationFailed,
            );
        }
        if mtp_artifact_capability.is_mtp_capable() {
            let mut optional_mtp_sources = Vec::new();
            let optional_mtp_validation = shard_index
                .mtp_only_shard_file_names()
                .iter()
                .try_for_each(|file_name| -> Result<(), ArtifactValidationError> {
                    let source_id =
                        source_id_by_file_name
                            .get(file_name)
                            .copied()
                            .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                                file_name: file_name.clone(),
                            })?;
                    let required_file = validate_required_file(
                        model_directory,
                        &RequiredFileProfile {
                            file_name: file_name.clone(),
                            size_bytes: 0,
                        },
                    )?;
                    let source = ValidatedSafetensorsSource::parse(source_id, required_file)?;
                    source.validate_inventory_profiles(
                        &tensor_inventory,
                        &recognized_tensor_profiles,
                    )?;
                    optional_mtp_sources.push(source);
                    Ok(())
                });
            if optional_mtp_validation.is_err() {
                tracing::debug!(
                    "optional indexed MTP source failed validation; serving target-only"
                );
                mtp_artifact_capability = Qwen3_5MtpArtifactCapability::target_only(
                    Qwen3_5MtpTargetOnlyReason::TensorValidationFailed,
                );
            } else {
                for source in optional_mtp_sources {
                    total_payload_bytes =
                        total_payload_bytes
                            .checked_add(source.payload_bytes())
                            .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
                    safetensors_sources.insert(source.source_id(), source);
                }
            }
        }
        if !mtp_artifact_capability.is_mtp_capable() {
            tensor_inventory.remove_feature(TensorFeature::MultiTokenPrediction);
        }
        let should_retain_mtp_sidecar = mtp_artifact_capability.is_mtp_capable();
        let mtp_sidecar_file_name = validated_mtp_sidecar
            .as_ref()
            .filter(|_| should_retain_mtp_sidecar)
            .map(|sidecar| sidecar.source.file_name().to_owned());
        if should_retain_mtp_sidecar && let Some(sidecar) = validated_mtp_sidecar {
            total_payload_bytes = total_payload_bytes
                .checked_add(sidecar.source.payload_bytes())
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
            source_id_by_file_name.insert(
                sidecar.source.file_name().to_owned(),
                sidecar.source.source_id(),
            );
            safetensors_sources.insert(sidecar.source.source_id(), sidecar.source);
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
            tensor_inventory,
            safetensors_sources,
            source_id_by_file_name,
            mtp_sidecar_file_name,
            model_id,
            revision,
            max_output_tokens,
        })
    }
}

fn parse_optional_mtp_contract(
    model_directory: &Path,
    config_bytes: &[u8],
) -> Result<Qwen3_5MtpContract, Qwen3_5MtpContractError> {
    let runtime_path = model_directory.join("mtplx_runtime.json");
    let optional_runtime_bytes = if runtime_path.exists() {
        let runtime_file = validate_required_file(
            model_directory,
            &RequiredFileProfile {
                file_name: "mtplx_runtime.json".to_owned(),
                size_bytes: 0,
            },
        )
        .map_err(|_| Qwen3_5MtpContractError::Malformed)?;
        if runtime_file.size_bytes() > super::MAXIMUM_MTPLX_RUNTIME_BYTES as u64 {
            return Err(Qwen3_5MtpContractError::RuntimeDocumentTooLarge);
        }
        Some(
            read_required_file_bytes(&runtime_file)
                .map_err(|_| Qwen3_5MtpContractError::Malformed)?,
        )
    } else {
        None
    };
    Qwen3_5MtpContract::parse(config_bytes, optional_runtime_bytes.as_deref())
}

impl From<&Qwen3_5MtpContractError> for Qwen3_5MtpTargetOnlyReason {
    fn from(contract_error: &Qwen3_5MtpContractError) -> Self {
        match contract_error {
            Qwen3_5MtpContractError::Malformed => Self::ContractMalformed,
            Qwen3_5MtpContractError::RuntimeDocumentTooLarge => {
                Self::ContractRuntimeDocumentTooLarge
            }
            Qwen3_5MtpContractError::FieldDisagreement => Self::ContractFieldDisagreement,
            Qwen3_5MtpContractError::Incompatible => Self::ContractIncompatible,
        }
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
