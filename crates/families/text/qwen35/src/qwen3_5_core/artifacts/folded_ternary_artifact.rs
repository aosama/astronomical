//! Validation for folded ternary packs (`prism_hadamard_qwen35`, such as
//! Ternary Bonsai): a Qwen 3.5 text model whose projections are stored as
//! 2-bit affine weights in a Hadamard-rotated basis, in one
//! `model.safetensors` with a `hadamard.json` sign table beside it.
//!
//! The pack first passes its own strict validation (every tensor against the
//! family's shape contract, every rotated module against the sign table).
//! Its configuration is then read as the Qwen 3.5 configuration it extends,
//! and a shard index is synthesized from the single file's header, as the
//! Expert Atlas branch does, so the shared core's inventory and binding
//! contracts run unchanged. The pack's per-module `.signs` tensors repeat the
//! sign table and are left out of that index; the rotations come from the
//! contract the validated artifact carries. The pack's float16 vision tower
//! is indexed and checked like any Qwen 3.5 tower, so the core reads images
//! with it.

use std::collections::HashMap;
use std::path::Path;

use super::artifact::derive_revision_from_config_bytes;
use super::artifact_helpers::{
    artifact_model_id, captured_required_file_bytes, parsed_index_source,
    push_optional_generation_config, required_file, shard_index_from_synthesized_document,
};
use super::artifact_inventory::{build_index_tensor_inventory, source_id_by_file_name};
use super::tensor_spec;
use super::validated_artifact::ValidatedQwen3_5Artifact;
use super::vision_validation;
use super::{
    Qwen3_5ArtifactValidationError, Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5ShardIndex,
    Qwen3_5VisionConfig,
};
use crate::qwen3_5_core::vision::vision_tensor_spec;
use std::collections::BTreeMap;

use crate::artifact_validation::{
    ArtifactValidationError, TensorInventory, TensorSourceId, ValidatedRequiredFile,
    ValidatedSafetensorsSource, validate_required_files,
};
use crate::prism_hadamard_qwen35::artifacts::{
    FoldedPackValidator, PACK_HADAMARD_FILE_NAME, PACK_WEIGHTS_FILE_NAME,
};
use engine_core::expert_paging::SafetensorsHeader;

/// The `model_type` a folded ternary pack declares.
const FOLDED_TERNARY_MODEL_TYPE: &str = "prism_hadamard_qwen35";
/// The Qwen 3.5 model type and architecture the pack's text model is.
const SHARED_CORE_MODEL_TYPE: &str = "qwen3_5";
const SHARED_CORE_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
/// A pack's `config.json` lists every module; a few hundred kilobytes at most.
const MAXIMUM_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
/// The suffix of the per-module sign tensors the sign table already holds.
const MODULE_SIGNS_SUFFIX: &str = ".signs";
/// The configuration keys only the folded pack uses.
const FOLDED_PACK_ONLY_KEYS: [&str; 6] = [
    "modules",
    "hadamard_config",
    "requires_runtime",
    "components",
    "tensor_namespace",
    "tied_output",
];

/// Whether `model_directory` holds a folded ternary pack: its `config.json`
/// declares the family's model type.
pub(super) fn declares_folded_ternary_pack(model_directory: &Path) -> bool {
    let config_path = model_directory.join("config.json");
    let Ok(config_metadata) = std::fs::metadata(&config_path) else {
        return false;
    };
    if config_metadata.len() > MAXIMUM_CONFIG_BYTES {
        return false;
    }
    let Ok(config_bytes) = std::fs::read(&config_path) else {
        return false;
    };
    declared_model_type(&config_bytes).as_deref() == Some(FOLDED_TERNARY_MODEL_TYPE)
}

/// Whether `model_directory` holds an artifact with no shard index: an
/// Expert Atlas model (its manifest's format is checked, not just its file
/// name, since a stock checkpoint may ship an unrelated `manifest.json`) or a
/// folded ternary pack.
pub(super) fn declares_layout_without_shard_index(model_directory: &Path) -> bool {
    expert_atlas::declares_atlas_manifest(model_directory)
        || declares_folded_ternary_pack(model_directory)
}

fn declared_model_type(config_bytes: &[u8]) -> Option<String> {
    let config_document: serde_json::Value = serde_json::from_slice(config_bytes).ok()?;
    Some(config_document.get("model_type")?.as_str()?.to_owned())
}

impl Qwen3_5ArtifactValidator {
    /// Validates an artifact that carries no shard index: an Expert Atlas
    /// model or a folded ternary pack.
    pub(super) fn validate_artifact_without_shard_index(
        model_directory: &Path,
        max_output_tokens: u32,
    ) -> Result<ValidatedQwen3_5Artifact, Qwen3_5ArtifactValidationError> {
        if declares_folded_ternary_pack(model_directory) {
            return Self::validate_folded_ternary_artifact(model_directory, max_output_tokens);
        }
        Self::validate_atlas_artifact(model_directory, max_output_tokens)
    }

    /// Validates a folded ternary pack (see the module documentation).
    pub(super) fn validate_folded_ternary_artifact(
        model_directory: &Path,
        max_output_tokens: u32,
    ) -> Result<ValidatedQwen3_5Artifact, Qwen3_5ArtifactValidationError> {
        let folded_pack = FoldedPackValidator::new()
            .validate(model_directory, max_output_tokens)
            .map_err(folded_pack_refusal)?;
        let required_files = collect_folded_pack_required_files(model_directory)?;
        let config_bytes = captured_required_file_bytes(&required_files, "config.json")?;
        let revision = derive_revision_from_config_bytes(config_bytes);
        let shared_config_bytes = shared_core_config_bytes(config_bytes)?;
        let mut config = Qwen3_5Config::from_json_bytes(&shared_config_bytes)?;
        let vision_config = Qwen3_5VisionConfig::from_optional_json_bytes(&shared_config_bytes)?;
        let (weights_header, weights_file_bytes) = folded_pack_weights_header(model_directory)?;
        let shard_index =
            synthesize_folded_pack_shard_index(&weights_header, weights_file_bytes, &mut config)?;
        let source_validation = validate_folded_pack_source(
            model_directory,
            &shard_index,
            (&config, vision_config.as_ref()),
        )?;
        Ok(ValidatedQwen3_5Artifact {
            hadamard_contract: Some(folded_pack.into_hadamard_contract()),
            expert_atlas: None,
            vision_config,
            has_separate_vision_sidecar: false,
            has_validated_vision_tower: source_validation.has_validated_vision_tower,
            model_id: artifact_model_id(model_directory),
            revision,
            max_output_tokens,
            total_payload_bytes: indexed_payload_bytes(&weights_header),
            config,
            required_files,
            shard_index,
            tensor_inventory: source_validation.tensor_inventory,
            safetensors_sources: source_validation.safetensors_sources,
            source_id_by_file_name: source_validation.source_id_by_file_name,
        })
    }
}

fn folded_pack_refusal(
    validation_error: crate::prism_hadamard_qwen35::artifacts::FoldedPackValidationError,
) -> Qwen3_5ArtifactValidationError {
    folded_pack_error(format!(
        "the folded ternary pack is invalid: {validation_error}"
    ))
}

fn collect_folded_pack_required_files(
    model_directory: &Path,
) -> Result<HashMap<String, ValidatedRequiredFile>, Qwen3_5ArtifactValidationError> {
    let mut required_file_profiles = vec![
        required_file("config.json"),
        required_file(PACK_HADAMARD_FILE_NAME),
        required_file("tokenizer.json"),
        required_file(PACK_WEIGHTS_FILE_NAME),
    ];
    push_optional_generation_config(&mut required_file_profiles, model_directory);
    Ok(validate_required_files(
        model_directory,
        &required_file_profiles,
    )?)
}

/// The pack's configuration as the Qwen 3.5 text configuration it extends:
/// the model type and architecture of the base model, without the keys only
/// the folded pack reads.
pub(super) fn shared_core_config_bytes(
    config_bytes: &[u8],
) -> Result<Vec<u8>, Qwen3_5ArtifactValidationError> {
    let mut config_document: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(config_bytes).map_err(|parse_error| {
            folded_pack_error(format!(
                "the folded pack's config.json is not an object: {parse_error}"
            ))
        })?;
    let base_model_type = config_document
        .get("base_model_type")
        .and_then(serde_json::Value::as_str);
    if base_model_type != Some(SHARED_CORE_MODEL_TYPE) {
        return Err(folded_pack_error(format!(
            "a folded ternary pack must extend `{SHARED_CORE_MODEL_TYPE}`, not {base_model_type:?}"
        )));
    }
    for folded_pack_only_key in FOLDED_PACK_ONLY_KEYS {
        config_document.remove(folded_pack_only_key);
    }
    config_document.insert("model_type".to_owned(), SHARED_CORE_MODEL_TYPE.into());
    config_document.insert(
        "architectures".to_owned(),
        serde_json::json!([SHARED_CORE_ARCHITECTURE]),
    );
    serde_json::to_vec(&config_document)
        .map_err(|serialize_error| folded_pack_error(serialize_error.to_string()))
}

/// Whether a pack tensor belongs in the shared core's index: every tensor
/// except the per-module sign copies, which the sign table already holds.
pub(super) fn is_indexed_tensor(tensor_name: &str) -> bool {
    !tensor_name.ends_with(MODULE_SIGNS_SUFFIX)
}

/// The header of the pack's one weights file, and the file's size.
fn folded_pack_weights_header(
    model_directory: &Path,
) -> Result<(SafetensorsHeader, u64), Qwen3_5ArtifactValidationError> {
    let weights_path = model_directory.join(PACK_WEIGHTS_FILE_NAME);
    let weights_header = engine_core::expert_paging::parse_safetensors_header(&weights_path)
        .map_err(|header_error| {
            folded_pack_error(format!(
                "its weights file header is invalid: {header_error}"
            ))
        })?;
    let weights_file_bytes = std::fs::metadata(&weights_path)
        .map_err(|source| folded_pack_error(format!("its weights file cannot be read: {source}")))?
        .len();
    Ok((weights_header, weights_file_bytes))
}

/// The shard index of the pack's single file, over its indexed tensors.
fn synthesize_folded_pack_shard_index(
    weights_header: &SafetensorsHeader,
    weights_file_bytes: u64,
    config: &mut Qwen3_5Config,
) -> Result<Qwen3_5ShardIndex, Qwen3_5ArtifactValidationError> {
    let mut weight_map = serde_json::Map::new();
    for tensor_entry in &weights_header.tensor_entries {
        if is_indexed_tensor(&tensor_entry.tensor_name) {
            weight_map.insert(
                tensor_entry.tensor_name.clone(),
                PACK_WEIGHTS_FILE_NAME.into(),
            );
        }
    }
    let index_bytes = serde_json::to_vec(&serde_json::json!({
        "metadata": { "total_size": weights_file_bytes },
        "weight_map": weight_map,
    }))
    .map_err(|serialize_error| folded_pack_error(serialize_error.to_string()))?;
    shard_index_from_synthesized_document(&index_bytes, config)
}

/// The payload the shared core binds: the indexed tensors' bytes, without
/// the sign copies it never loads.
fn indexed_payload_bytes(weights_header: &SafetensorsHeader) -> u64 {
    weights_header
        .tensor_entries
        .iter()
        .filter(|tensor_entry| is_indexed_tensor(&tensor_entry.tensor_name))
        .map(|tensor_entry| tensor_entry.data_end_offset - tensor_entry.data_start_offset)
        .sum()
}

/// The inventory and source of a folded pack's one file.
struct FoldedPackSourceValidation {
    tensor_inventory: TensorInventory,
    safetensors_sources: HashMap<TensorSourceId, ValidatedSafetensorsSource>,
    source_id_by_file_name: BTreeMap<String, TensorSourceId>,
    has_validated_vision_tower: bool,
}

/// Checks every indexed tensor against the shared core's language and
/// vision profiles, passing over the sign copies the index leaves out,
/// which the pack's own contract has already checked.
fn validate_folded_pack_source(
    model_directory: &Path,
    shard_index: &Qwen3_5ShardIndex,
    (config, vision_config): (&Qwen3_5Config, Option<&Qwen3_5VisionConfig>),
) -> Result<FoldedPackSourceValidation, Qwen3_5ArtifactValidationError> {
    let vision_tower_storage =
        vision_validation::validate_vision_tower_inventory(shard_index, vision_config, config)?;
    let tensor_inventory = build_index_tensor_inventory(shard_index)?;
    let mut recognized_tensor_profiles = tensor_spec::qwen3_5_language_tensor_profiles(config)?;
    if let Some(vision_config) = vision_config {
        recognized_tensor_profiles.extend(
            vision_tensor_spec::qwen3_5_vision_tensor_profiles_for_model(vision_config, config)?,
        );
    }
    let source_id_by_file_name = source_id_by_file_name(shard_index)?;
    let mut safetensors_sources = HashMap::new();
    for (file_name, source_id) in &source_id_by_file_name {
        let source = parsed_index_source(model_directory, file_name, *source_id)?;
        source.validate_inventory_profiles_leaving_out(
            &tensor_inventory,
            &recognized_tensor_profiles,
            is_left_out_of_the_index,
        )?;
        safetensors_sources.insert(*source_id, source);
    }
    Ok(FoldedPackSourceValidation {
        tensor_inventory,
        safetensors_sources,
        source_id_by_file_name,
        has_validated_vision_tower: vision_tower_storage.has_validated_vision_tower(),
    })
}

fn is_left_out_of_the_index(tensor_name: &str) -> bool {
    !is_indexed_tensor(tensor_name)
}

fn folded_pack_error(description: String) -> Qwen3_5ArtifactValidationError {
    ArtifactValidationError::InvalidFoldedTernaryPack { description }.into()
}
