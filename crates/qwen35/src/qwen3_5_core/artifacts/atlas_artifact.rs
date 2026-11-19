//! Validation for Expert Atlas models.
//!
//! The Atlas manifest replaces the shard index: it declares the resident
//! weight bundle, one record file per expert layer, every file's byte count,
//! and where each expert tensor sits inside a record. Dense tensors bind to
//! the standard `resident.safetensors`; routed expert tensors live only in
//! the layer files, so their inventory locations are stripped here and the
//! expert plans are built from the manifest when the model loads. A
//! synthesized index document keeps the shard-index, inventory, and binding
//! contracts identical to the shard path.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use super::artifact::derive_revision_from_config_bytes;
use super::artifact_helpers::{captured_required_file_bytes, required_file};
use super::artifact_inventory::{build_index_tensor_inventory, source_id_by_file_name};
use super::optiq_metadata_validation::{
    register_optional_optiq_metadata, validate_optional_optiq_metadata,
};
use super::tensor_spec;
use super::validated_artifact::ValidatedQwen3_5Artifact;
use super::vision_validation::{self};
use super::{
    Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5ShardIndex,
    Qwen3_5VisionConfig,
};
use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, TensorInventory, TensorSourceId,
    ValidatedRequiredFile, ValidatedSafetensorsSource, hugging_face_snapshot_model_id,
    validate_required_file, validate_required_files,
};
use crate::expert_paging::SafetensorsHeader;
use crate::qwen3_5_core::vision::vision_tensor_spec;
use astronomical_expert_atlas::{RESIDENT_BUNDLE_FILE_NAME, VISION_BUNDLE_FILE_NAME};

impl Qwen3_5ArtifactValidator {
    /// Validates an Expert Atlas model (see the module documentation).
    pub(super) fn validate_atlas_artifact(
        model_directory: &Path,
        max_output_tokens: u32,
    ) -> Result<ValidatedQwen3_5Artifact, Qwen3_5ArtifactValidationError> {
        let required_files = collect_atlas_required_files(model_directory)?;
        let config_bytes = captured_required_file_bytes(&required_files, "config.json")?;
        let revision = derive_revision_from_config_bytes(config_bytes);
        let mut config = Qwen3_5Config::from_json_bytes(config_bytes)?;
        let vision_config = Qwen3_5VisionConfig::from_optional_json_bytes(config_bytes)?;
        validate_optional_optiq_metadata(&required_files, &config)?;
        let expert_atlas = review_atlas_manifest(model_directory)?;
        let (resident_header, vision_header) =
            parse_atlas_bundle_headers(model_directory, &vision_config)?;
        let shard_index = synthesize_atlas_shard_index(
            model_directory,
            &resident_header,
            vision_header.as_ref(),
            &mut config,
        )?;
        let source_validation = validate_atlas_sources(
            model_directory,
            &shard_index,
            &config,
            vision_config.as_ref(),
        )?;
        Ok(ValidatedQwen3_5Artifact {
            config,
            vision_config,
            required_files,
            shard_index,
            // Every weight byte counts, not just the resident bundle: memory
            // budgeting compares this with the ceiling, and leaving the
            // experts out would make a 35B model look small enough to hold.
            total_payload_bytes: source_validation
                .validated_total_payload_bytes
                .checked_add(atlas_expert_payload_bytes(&expert_atlas)?)
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?,
            has_separate_vision_sidecar: source_validation.has_separate_vision_sidecar,
            has_validated_vision_tower: source_validation.has_validated_vision_tower,
            tensor_inventory: source_validation.tensor_inventory,
            safetensors_sources: source_validation.safetensors_sources,
            source_id_by_file_name: source_validation.source_id_by_file_name_map,
            model_id: source_validation.model_id,
            revision,
            max_output_tokens,
            expert_atlas: Some(expert_atlas),
        })
    }
}

/// Collects the always-required files plus the two optional documents when
/// they are present on disk, validating each against its profile.
fn collect_atlas_required_files(
    model_directory: &Path,
) -> Result<HashMap<String, ValidatedRequiredFile>, Qwen3_5ArtifactValidationError> {
    let mut required_file_profiles = vec![
        required_file("config.json"),
        required_file("manifest.json"),
        required_file("tokenizer.json"),
        required_file(RESIDENT_BUNDLE_FILE_NAME),
    ];
    if model_directory.join("generation_config.json").is_file() {
        required_file_profiles.push(required_file("generation_config.json"));
    }
    register_optional_optiq_metadata(&mut required_file_profiles, model_directory);
    Ok(validate_required_files(
        model_directory,
        &required_file_profiles,
    )?)
}

/// The routed-expert payload the layer files hold, excluding record padding.
fn atlas_expert_payload_bytes(
    expert_atlas: &astronomical_expert_atlas::AtlasManifest,
) -> Result<u64, Qwen3_5ArtifactValidationError> {
    expert_atlas
        .expert_layers
        .iter()
        .try_fold(0_u64, |payload_bytes, expert_layer| {
            expert_layer
                .record_byte_count
                .checked_mul(expert_layer.expert_count as u64)
                .and_then(|layer_payload_bytes| payload_bytes.checked_add(layer_payload_bytes))
        })
        .ok_or_else(|| ArtifactValidationError::TensorPayloadSizeOverflow.into())
}

/// Reads the Atlas manifest (format, version and per-layer geometry are
/// checked by the reader) and requires every declared file to exist with its
/// declared byte count. Content digests are checked by the conversion that
/// wrote them; serving binds to the declared sizes and record layout.
fn review_atlas_manifest(
    model_directory: &Path,
) -> Result<astronomical_expert_atlas::AtlasManifest, Qwen3_5ArtifactValidationError> {
    let expert_atlas =
        astronomical_expert_atlas::AtlasManifest::read_from_directory(model_directory).map_err(
            |manifest_error| ArtifactValidationError::InvalidAtlasManifest {
                description: manifest_error.to_string(),
            },
        )?;
    for declared_file in expert_atlas
        .resident_files
        .iter()
        .chain(&expert_atlas.expert_files)
    {
        let declared_path = model_directory.join(&declared_file.file_name);
        let actual_file_byte_count = std::fs::metadata(&declared_path)
            .map_err(|source| ArtifactValidationError::AtlasDeclaredFileMissing {
                file_name: declared_file.file_name.clone(),
                source,
            })?
            .len();
        if actual_file_byte_count != declared_file.expected_file_byte_count {
            return Err(ArtifactValidationError::AtlasDeclaredFileSize {
                file_name: declared_file.file_name.clone(),
                expected_bytes: declared_file.expected_file_byte_count,
                actual_bytes: actual_file_byte_count,
            }
            .into());
        }
    }
    Ok(expert_atlas)
}

/// Parses the resident bundle header, requiring the vision bundle beside it
/// whenever the artifact declares a vision tower.
fn parse_atlas_bundle_headers(
    model_directory: &Path,
    vision_config: &Option<Qwen3_5VisionConfig>,
) -> Result<(SafetensorsHeader, Option<SafetensorsHeader>), Qwen3_5ArtifactValidationError> {
    // The resident bundle carries every non-expert tensor; the vision
    // bundle travels beside it unchanged from the source artifact.
    let resident_header = crate::expert_paging::parse_safetensors_header(
        &model_directory.join(RESIDENT_BUNDLE_FILE_NAME),
    )
    .map_err(|source| ArtifactValidationError::InvalidResidentBundle { source })
    .map_err(Qwen3_5ArtifactValidationError::from)?;
    if vision_config.is_some() && !model_directory.join(VISION_BUNDLE_FILE_NAME).is_file() {
        return Err(ArtifactValidationError::AtlasDeclaredFileMissing {
            file_name: VISION_BUNDLE_FILE_NAME.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "the model declares a vision tower but the Atlas carries no vision bundle",
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
    Ok((resident_header, vision_header))
}

/// Synthesizes the standard shard-index document from the bundle headers and
/// the reviewed tensor profiles: sparse expert names map to the resident
/// bundle as a placeholder that is never opened (MoE weight binding excludes
/// sparse tensors and the expert plans come from the Atlas manifest), the vision bundle maps
/// its own tensors, and the resolved canonical names feed back into the
/// config before the index is parsed.
fn synthesize_atlas_shard_index(
    model_directory: &Path,
    resident_header: &SafetensorsHeader,
    vision_header: Option<&SafetensorsHeader>,
    config: &mut Qwen3_5Config,
) -> Result<Qwen3_5ShardIndex, Qwen3_5ArtifactValidationError> {
    let mut weight_map = serde_json::Map::new();
    for tensor_entry in &resident_header.tensor_entries {
        weight_map.insert(
            tensor_entry.tensor_name.clone(),
            serde_json::Value::String(RESIDENT_BUNDLE_FILE_NAME.to_owned()),
        );
    }
    for tensor_profile in tensor_spec::qwen3_5_language_tensor_profiles(config)? {
        if tensor_profile.name.contains(".mlp.switch_mlp.") {
            weight_map
                .entry(tensor_profile.name.clone())
                .or_insert_with(|| serde_json::Value::String(RESIDENT_BUNDLE_FILE_NAME.to_owned()));
        }
    }
    if let Some(vision_header) = vision_header {
        for tensor_entry in &vision_header.tensor_entries {
            weight_map.insert(
                tensor_entry.tensor_name.clone(),
                serde_json::Value::String(VISION_BUNDLE_FILE_NAME.to_owned()),
            );
        }
    }
    let mut total_index_payload_bytes = 0_u64;
    for bundle_file_name in [RESIDENT_BUNDLE_FILE_NAME, VISION_BUNDLE_FILE_NAME] {
        if model_directory.join(bundle_file_name).is_file() {
            let bundle_byte_count = std::fs::metadata(model_directory.join(bundle_file_name))
                .map_err(|source| ArtifactValidationError::AtlasDeclaredFileMissing {
                    file_name: bundle_file_name.to_owned(),
                    source,
                })
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
        .map_err(|source| ArtifactValidationError::InvalidAtlasManifest {
            description: source.to_string(),
        })
        .map_err(Qwen3_5ArtifactValidationError::from)?;
    let canonical_model_tensor_names =
        Qwen3_5ShardIndex::extract_language_and_vision_tensor_names_from_json(&index_bytes)?;
    config.resolve_unquantized_modules_from_shard_index(&canonical_model_tensor_names);
    let language_tensor_profiles = tensor_spec::qwen3_5_language_tensor_profiles(config)?;
    Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .map_err(Qwen3_5ArtifactValidationError::from)
}

/// The inventory-and-sources outcome one Atlas model carries into its
/// validated artifact.
struct AtlasSourceValidation {
    tensor_inventory: TensorInventory,
    safetensors_sources: HashMap<TensorSourceId, ValidatedSafetensorsSource>,
    source_id_by_file_name_map: BTreeMap<String, TensorSourceId>,
    validated_total_payload_bytes: u64,
    has_separate_vision_sidecar: bool,
    has_validated_vision_tower: bool,
    model_id: String,
}

/// Validates the vision tower's inventory, strips the sparse expert names
/// from the index inventory (the expert plans come from the Atlas manifest), and validates
/// every declared safetensors source against the recognized profiles.
fn validate_atlas_sources(
    model_directory: &Path,
    shard_index: &Qwen3_5ShardIndex,
    config: &Qwen3_5Config,
    vision_config: Option<&Qwen3_5VisionConfig>,
) -> Result<AtlasSourceValidation, Qwen3_5ArtifactValidationError> {
    let validated_vision_tower_storage =
        vision_validation::validate_vision_tower_inventory(shard_index, vision_config, config)?;
    let has_separate_vision_sidecar = validated_vision_tower_storage.has_separate_sidecar();
    let vision_tensor_profiles = vision_config
        .as_ref()
        .map(|vision_config| {
            vision_tensor_spec::qwen3_5_vision_tensor_profiles_for_model(vision_config, config)
        })
        .transpose()?
        .unwrap_or_default();
    let mut tensor_inventory = build_index_tensor_inventory(shard_index)?;
    let sparse_expert_names: BTreeSet<String> =
        tensor_spec::qwen3_5_language_tensor_profiles(config)?
            .iter()
            .filter(|tensor_profile| tensor_profile.name.contains(".mlp.switch_mlp."))
            .map(|tensor_profile| tensor_profile.name.clone())
            .collect();
    tensor_inventory.remove_canonical_names(&sparse_expert_names);
    let mut recognized_tensor_profiles = tensor_spec::qwen3_5_language_tensor_profiles(config)?;
    recognized_tensor_profiles.extend(vision_tensor_profiles.clone());
    let source_id_by_file_name_map = source_id_by_file_name(shard_index)?;
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
        source
            .validate_required_inventory_profiles(&tensor_inventory, &recognized_tensor_profiles)?;
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
    Ok(AtlasSourceValidation {
        tensor_inventory,
        safetensors_sources,
        source_id_by_file_name_map,
        validated_total_payload_bytes,
        has_separate_vision_sidecar,
        has_validated_vision_tower: validated_vision_tower_storage.has_validated_vision_tower(),
        model_id,
    })
}
