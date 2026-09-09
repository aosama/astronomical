//! Validation for converted per-expert streaming revisions.
//!
//! The manifest replaces the shard index: it declares the resident weight
//! bundle, the expert packs, and every file's byte count. Dense tensors bind
//! to the standard `resident.safetensors`; sparse expert tensors live only in
//! the per-expert packs, so their inventory locations are stripped and the
//! expert pager derives their plans from pack headers. A synthesized index
//! document keeps the shard-index, inventory, and binding contracts identical
//! to the shard path.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorFeature, TensorSemanticRole,
    ValidatedSafetensorsSource, hugging_face_snapshot_model_id, validate_required_file,
    validate_required_files,
};
use crate::qwen3_5::artifacts::artifact::{
    derive_revision_from_config_bytes, parse_optional_mtp_contract,
};
use crate::qwen3_5::artifacts::artifact_helpers::{
    captured_required_file_bytes, read_required_file_bytes, required_file,
};
use crate::qwen3_5::artifacts::artifact_inventory::{
    build_index_tensor_inventory, source_id_by_file_name,
};
use crate::qwen3_5::artifacts::tensor_spec::qwen3_5_language_tensor_profiles;
use crate::qwen3_5::artifacts::validated_artifact::ValidatedQwen3_5Artifact;
use crate::qwen3_5::artifacts::vision_tensor_spec::qwen3_5_vision_tensor_profiles;
use crate::qwen3_5::artifacts::vision_validation::validate_vision_tower_inventory;
use crate::qwen3_5::artifacts::{
    OptiQMetadata, Qwen3_5Config, Qwen3_5ShardIndex, Qwen3_5VisionConfig,
};
use crate::qwen3_5::artifacts::{
    Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, Qwen3_5MtpArtifactCapability,
};
use crate::qwen3_5::multi_token_prediction::qwen3_5_mtp_tensor_profiles;

/// On-disk format version of a converted per-expert streaming revision.
const STREAMING_REVISION_FORMAT_VERSION: u32 = 3;

/// Minimal streaming revision manifest view used by artifact validation.
#[derive(serde::Deserialize)]
struct StreamingRevisionManifestProbe {
    format_version: u32,
    resident_files: Vec<StreamingRevisionResidentFileProbe>,
    expert_files: Vec<StreamingRevisionExpertFileProbe>,
}

#[derive(serde::Deserialize)]
struct StreamingRevisionResidentFileProbe {
    file_name: String,
    expected_file_byte_count: u64,
}

#[derive(serde::Deserialize)]
struct StreamingRevisionExpertFileProbe {
    file_name: String,
    expected_file_byte_count: u64,
}

impl Qwen3_5ArtifactValidator {
    /// Validates a converted per-expert streaming revision.
    ///
    /// The manifest replaces the shard index: it declares the resident weight
    /// bundle, the expert packs, and every file's byte count. Dense tensors
    /// bind to the standard `resident.safetensors`; sparse expert tensors live
    /// only in the per-expert packs, so their inventory locations are stripped
    /// and the expert pager derives their plans from pack headers. A
    /// synthesized index document keeps the shard-index, inventory, and
    /// binding contracts identical to the shard path.
    pub(super) fn validate_streaming_revision(
        model_directory: &Path,
        max_output_tokens: u32,
    ) -> Result<ValidatedQwen3_5Artifact, Qwen3_5ArtifactValidationError> {
        let mut required_file_profiles = vec![
            required_file("config.json"),
            required_file("manifest.json"),
            required_file("tokenizer.json"),
            required_file("resident.safetensors"),
        ];
        if model_directory.join("generation_config.json").is_file() {
            required_file_profiles.push(required_file("generation_config.json"));
        }
        if model_directory.join("optiq_metadata.json").is_file() {
            required_file_profiles.push(required_file("optiq_metadata.json"));
        }
        let required_files = validate_required_files(model_directory, &required_file_profiles)?;

        let config_bytes = captured_required_file_bytes(&required_files, "config.json")?;
        let revision = derive_revision_from_config_bytes(config_bytes);
        let mut config = Qwen3_5Config::from_json_bytes(config_bytes)?;
        let vision_config = Qwen3_5VisionConfig::from_optional_json_bytes(config_bytes)?;
        if let Some(optiq_metadata_required_file) = required_files.get("optiq_metadata.json") {
            let optiq_metadata_bytes = read_required_file_bytes(optiq_metadata_required_file)?;
            OptiQMetadata::from_json_bytes(&optiq_metadata_bytes)?
                .validate_against_config(&config)?;
        }

        // Manifest completeness: every declared file exists with its declared
        // byte count. Content hashes gate preparation and re-conversion; serving
        // integrity is bound by declared sizes plus per-pack header identity.
        // The manifest is too large for the small-file byte capture contract
        // (it declares one entry per expert); read it directly instead.
        let manifest_bytes = std::fs::read(model_directory.join("manifest.json"))
            .map_err(
                |source| ArtifactValidationError::StreamingDeclaredFileMissing {
                    file_name: "manifest.json".to_owned(),
                    source,
                },
            )
            .map_err(Qwen3_5ArtifactValidationError::from)?;
        let manifest: StreamingRevisionManifestProbe =
            serde_json::from_slice(manifest_bytes.as_slice())
                .map_err(|source| ArtifactValidationError::InvalidStreamingManifest { source })
                .map_err(Qwen3_5ArtifactValidationError::from)?;
        if manifest.format_version != STREAMING_REVISION_FORMAT_VERSION {
            return Err(
                ArtifactValidationError::UnsupportedStreamingManifestVersion {
                    actual_format_version: manifest.format_version,
                }
                .into(),
            );
        }
        let declared_file_entries = manifest
            .expert_files
            .iter()
            .map(|expert_file| (&expert_file.file_name, expert_file.expected_file_byte_count))
            .chain(manifest.resident_files.iter().map(|resident_file| {
                (
                    &resident_file.file_name,
                    resident_file.expected_file_byte_count,
                )
            }));
        for (file_name, expected_file_byte_count) in declared_file_entries {
            let declared_path = model_directory.join(file_name);
            let actual_file_byte_count = std::fs::metadata(&declared_path)
                .map_err(
                    |source| ArtifactValidationError::StreamingDeclaredFileMissing {
                        file_name: file_name.clone(),
                        source,
                    },
                )
                .map_err(Qwen3_5ArtifactValidationError::from)?
                .len();
            if actual_file_byte_count != expected_file_byte_count {
                return Err(ArtifactValidationError::StreamingDeclaredFileSize {
                    file_name: file_name.clone(),
                    expected_bytes: expected_file_byte_count,
                    actual_bytes: actual_file_byte_count,
                }
                .into());
            }
        }

        // The resident bundle carries every non-expert tensor; the vision
        // bundle travels beside it unchanged from the source artifact.
        let resident_header = crate::expert_paging::parse_safetensors_header(
            &model_directory.join("resident.safetensors"),
        )
        .map_err(|source| ArtifactValidationError::InvalidResidentBundle { source })
        .map_err(Qwen3_5ArtifactValidationError::from)?;
        const VISION_BUNDLE_FILE_NAME: &str = "optiq/optiq_vision.safetensors";
        if vision_config.is_some() && !model_directory.join(VISION_BUNDLE_FILE_NAME).is_file() {
            return Err(ArtifactValidationError::StreamingDeclaredFileMissing {
                file_name: VISION_BUNDLE_FILE_NAME.to_owned(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "the model declares a vision tower but the revision carries no vision bundle",
                ),
            }
            .into());
        }
        let vision_header = if model_directory.join(VISION_BUNDLE_FILE_NAME).is_file() {
            Some(
                crate::expert_paging::parse_safetensors_header(
                    &model_directory.join(VISION_BUNDLE_FILE_NAME),
                )
                .map_err(|source| ArtifactValidationError::InvalidResidentBundle { source })
                .map_err(Qwen3_5ArtifactValidationError::from)?,
            )
        } else {
            None
        };

        // Synthesize the standard index document. Sparse expert names map to
        // the resident bundle as a placeholder that is never opened: MoE weight
        // binding excludes sparse tensors and the pager reads pack headers.
        let mut weight_map = serde_json::Map::new();
        for tensor_entry in &resident_header.tensor_entries {
            weight_map.insert(
                tensor_entry.tensor_name.clone(),
                serde_json::Value::String("resident.safetensors".to_owned()),
            );
        }
        for tensor_profile in qwen3_5_language_tensor_profiles(&config) {
            if tensor_profile.name.contains(".mlp.switch_mlp.") {
                weight_map
                    .entry(tensor_profile.name.clone())
                    .or_insert_with(|| {
                        serde_json::Value::String("resident.safetensors".to_owned())
                    });
            }
        }
        if let Some(vision_header) = &vision_header {
            for tensor_entry in &vision_header.tensor_entries {
                weight_map.insert(
                    tensor_entry.tensor_name.clone(),
                    serde_json::Value::String(VISION_BUNDLE_FILE_NAME.to_owned()),
                );
            }
        }
        let mut total_index_payload_bytes = 0_u64;
        for bundle_file_name in ["resident.safetensors", VISION_BUNDLE_FILE_NAME] {
            if model_directory.join(bundle_file_name).is_file() {
                let bundle_byte_count = std::fs::metadata(model_directory.join(bundle_file_name))
                    .map_err(
                        |source| ArtifactValidationError::StreamingDeclaredFileMissing {
                            file_name: bundle_file_name.to_owned(),
                            source,
                        },
                    )
                    .map_err(Qwen3_5ArtifactValidationError::from)?
                    .len();
                total_index_payload_bytes = total_index_payload_bytes
                    .checked_add(bundle_byte_count)
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
            }
        }
        let index_document = serde_json::json!({
            "metadata": { "total_size": total_index_payload_bytes },
            "weight_map": serde_json::Value::Object(weight_map),
        });
        let index_bytes = serde_json::to_vec(&index_document)
            .map_err(|source| ArtifactValidationError::InvalidStreamingManifest { source })
            .map_err(Qwen3_5ArtifactValidationError::from)?;
        let canonical_tensor_names =
            Qwen3_5ShardIndex::extract_language_tensor_names_from_json(&index_bytes)?;
        config.resolve_unquantized_modules_from_shard_index(&canonical_tensor_names);
        let language_tensor_profiles = qwen3_5_language_tensor_profiles(&config);
        let shard_index =
            Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)?;
        let validated_vision_tower_storage =
            validate_vision_tower_inventory(&shard_index, vision_config.as_ref())?;
        let has_separate_vision_sidecar = validated_vision_tower_storage.has_separate_sidecar();
        let vision_tensor_profiles = vision_config
            .as_ref()
            .map(qwen3_5_vision_tensor_profiles)
            .unwrap_or_default();
        let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&config);
        let mut tensor_inventory = build_index_tensor_inventory(&shard_index)?;
        let sparse_expert_names: BTreeSet<String> = language_tensor_profiles
            .iter()
            .filter(|tensor_profile| tensor_profile.name.contains(".mlp.switch_mlp."))
            .map(|tensor_profile| tensor_profile.name.clone())
            .collect();
        tensor_inventory.remove_canonical_names(&sparse_expert_names);
        let canonical_mtp_names: BTreeSet<String> = tensor_inventory
            .locations()
            .filter(|location| location.semantic_role() == TensorSemanticRole::MultiTokenPrediction)
            .map(|location| location.canonical_name().to_owned())
            .collect();
        let mtp_contract = parse_optional_mtp_contract(model_directory, config_bytes);
        let mtp_artifact_capability = Qwen3_5MtpArtifactCapability::from_canonical_tensor_names(
            &config,
            canonical_mtp_names,
            mtp_contract.as_ref().ok(),
        );
        if !mtp_artifact_capability.is_mtp_capable() {
            tensor_inventory.remove_feature(TensorFeature::MultiTokenPrediction);
        }
        let mut recognized_tensor_profiles = language_tensor_profiles.clone();
        recognized_tensor_profiles.extend(mtp_tensor_profiles.clone());
        recognized_tensor_profiles.extend(vision_tensor_profiles.clone());
        let source_id_by_file_name_map = source_id_by_file_name(&shard_index)?;
        let mut safetensors_sources = HashMap::new();
        let mut validated_total_payload_bytes = 0_u64;
        for (file_name, source_id) in &source_id_by_file_name_map {
            let required_file_entry = validate_required_file(
                model_directory,
                &RequiredFileProfile {
                    file_name: file_name.clone(),
                    size_bytes: 0,
                },
            )?;
            let source = ValidatedSafetensorsSource::parse(*source_id, required_file_entry)?;
            source.validate_required_inventory_profiles(
                &tensor_inventory,
                &recognized_tensor_profiles,
            )?;
            validated_total_payload_bytes = validated_total_payload_bytes
                .checked_add(source.payload_bytes())
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
            safetensors_sources.insert(*source_id, source);
        }
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
            total_payload_bytes: validated_total_payload_bytes,
            has_separate_vision_sidecar,
            has_validated_vision_tower: validated_vision_tower_storage.has_validated_vision_tower(),
            mtp_artifact_capability,
            tensor_inventory,
            safetensors_sources,
            source_id_by_file_name: source_id_by_file_name_map,
            mtp_sidecar_file_name: None,
            model_id,
            revision,
            max_output_tokens,
        })
    }
}
