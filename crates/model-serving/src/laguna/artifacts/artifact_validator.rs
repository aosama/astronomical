use std::collections::BTreeMap;
use std::path::Path;

use crate::artifact_validation::{
    ArtifactValidationError, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    read_bounded_required_file_bytes, validate_required_file, validate_required_files,
};
use crate::laguna::{
    LagunaFeedForwardDescriptor, LagunaNormalizationError, LagunaTargetContract,
    LagunaTargetNormalizer, LagunaTextArtifactDescriptor, LagunaTextArtifactNormalizer,
    LagunaTextArtifactSources,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::artifact_error::LagunaArtifactValidationError;
use super::canonical_tensor_contract::{
    LocatedRawTensorDescriptor, build_canonical_tensor_contract,
};
use super::retained_artifact::{LagunaIndexTotalSizeSemantics, ValidatedLagunaArtifact};
use super::shard_index::LagunaShardIndex;
use super::storage_fingerprint::storage_fingerprint;
use super::template_source_validator::LagunaTemplateSourceValidator;
use super::tensor_assembly::LagunaRawTensorNameRecord;
use super::tensor_name_normalizer::LagunaTensorNameNormalizer;

const CONFIG_FILE_NAME: &str = "config.json";
const INDEX_FILE_NAME: &str = "model.safetensors.index.json";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
const GENERATION_CONFIG_FILE_NAME: &str = "generation_config.json";
const MAXIMUM_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_INDEX_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_TEXT_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;

/// Validates the complete Laguna artifact before any model object is constructed.
#[derive(Debug, Default)]
pub struct LagunaArtifactValidator;

impl LagunaArtifactValidator {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Produces one canonical target, tensor, storage, and retained-file contract.
    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
    ) -> Result<ValidatedLagunaArtifact, LagunaArtifactValidationError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        self.validate_with_performance_attribution(model_directory, &mut performance_attribution)
    }

    /// Validates while attributing the outer operation, retained mapping, and tensor binding.
    pub fn validate_with_performance_attribution(
        self,
        model_directory: impl AsRef<Path>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<ValidatedLagunaArtifact, LagunaArtifactValidationError> {
        let model_directory = model_directory.as_ref();
        performance_attribution.measure_operation(
            PerformanceOperation::ArtifactValidation,
            |performance_attribution| {
                self.validate_attributed(model_directory, performance_attribution)
            },
        )
    }

    fn validate_attributed(
        self,
        model_directory: &Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<ValidatedLagunaArtifact, LagunaArtifactValidationError> {
        if !model_directory.is_dir() {
            return Err(LagunaArtifactValidationError::ModelDirectoryUnavailable);
        }
        let mut required_files = validate_required_files(
            model_directory,
            &[
                required_file_profile(CONFIG_FILE_NAME),
                required_file_profile(INDEX_FILE_NAME),
                required_file_profile(TOKENIZER_FILE_NAME),
                required_file_profile(TOKENIZER_CONFIG_FILE_NAME),
                required_file_profile(GENERATION_CONFIG_FILE_NAME),
            ],
        )?;
        let config_file = remove_required_file(&mut required_files, CONFIG_FILE_NAME)?;
        let index_file = remove_required_file(&mut required_files, INDEX_FILE_NAME)?;
        let tokenizer_file = remove_required_file(&mut required_files, TOKENIZER_FILE_NAME)?;
        let tokenizer_config_file =
            remove_required_file(&mut required_files, TOKENIZER_CONFIG_FILE_NAME)?;
        let generation_config_file =
            remove_required_file(&mut required_files, GENERATION_CONFIG_FILE_NAME)?;

        // Configuration bytes come from the descriptor validated above; no mutable
        // configuration path participates after this point.
        let config_bytes = read_bounded_required_file_bytes(&config_file, MAXIMUM_CONFIG_BYTES)?;
        let target_contract = match LagunaTargetNormalizer::normalize(&config_bytes) {
            Ok(target_contract) => target_contract,
            Err(LagunaNormalizationError::AmbiguousGatingBoolean) => {
                // The published legacy boolean means per-head gating. The later
                // canonical tensor build proves that every gate projection has that shape.
                LagunaTargetNormalizer::normalize_with_per_head_boolean_gating(&config_bytes)?
            }
            Err(normalization_error) => return Err(normalization_error.into()),
        };
        let validated_text_sidecars = performance_attribution.measure_operation(
            PerformanceOperation::TokenizerInitialization,
            |_performance_attribution| {
                validate_text_sidecars(
                    model_directory,
                    &target_contract,
                    &config_bytes,
                    tokenizer_file,
                    tokenizer_config_file,
                    generation_config_file,
                )
            },
        )?;
        let index_bytes = read_bounded_required_file_bytes(&index_file, MAXIMUM_INDEX_BYTES)?;
        let shard_index = LagunaShardIndex::from_json_bytes(&index_bytes)?;
        let (shard_files, located_tensors, shard_inventory_byte_totals) = performance_attribution
            .measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |_performance_attribution| {
                let shard_files = validate_shard_descriptors(model_directory, &shard_index)?;
                let (located_tensors, shard_inventory_byte_totals) =
                    inventory_and_validate_ownership(&shard_index, &shard_files)?;
                Ok::<_, LagunaArtifactValidationError>((
                    shard_files,
                    located_tensors,
                    shard_inventory_byte_totals,
                ))
            },
        )?;
        let index_total_size_semantics = if shard_index.declared_total_size_bytes()
            == shard_inventory_byte_totals.shard_file_bytes
        {
            LagunaIndexTotalSizeSemantics::SerializedShardFiles
        } else if shard_index.declared_total_size_bytes()
            == shard_inventory_byte_totals.tensor_payload_bytes
        {
            LagunaIndexTotalSizeSemantics::TensorPayload
        } else {
            return Err(LagunaArtifactValidationError::IndexTotalSizeMismatch {
                declared_total_size_bytes: shard_index.declared_total_size_bytes(),
                actual_shard_file_bytes: shard_inventory_byte_totals.shard_file_bytes,
                actual_tensor_payload_bytes: shard_inventory_byte_totals.tensor_payload_bytes,
            });
        };

        let (tensor_contract, storage_fingerprint) = performance_attribution.measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| {
                let raw_tensor_records = located_tensors
                    .keys()
                    .cloned()
                    .map(LagunaRawTensorNameRecord::new)
                    .collect::<Vec<_>>();
                let tensor_name_contract = LagunaTensorNameNormalizer::new(
                    target_contract.model().layer_count(),
                    target_expert_count(&target_contract)?,
                )
                .normalize(&raw_tensor_records)?;
                let tensor_contract = build_canonical_tensor_contract(
                    &target_contract,
                    &tensor_name_contract,
                    &located_tensors,
                )?;
                let storage_fingerprint = storage_fingerprint(
                    &target_contract,
                    &tensor_contract,
                    shard_inventory_byte_totals.tensor_payload_bytes,
                )?;
                Ok::<_, LagunaArtifactValidationError>((tensor_contract, storage_fingerprint))
            },
        )?;
        Ok(ValidatedLagunaArtifact {
            target_contract,
            shard_index,
            tensor_contract,
            text_artifact: validated_text_sidecars.text_artifact,
            config_file,
            index_file,
            tokenizer_file: validated_text_sidecars.tokenizer_file,
            tokenizer_config_file: validated_text_sidecars.tokenizer_config_file,
            generation_config_file: validated_text_sidecars.generation_config_file,
            included_template_files: validated_text_sidecars.included_template_files,
            shard_files,
            total_shard_file_bytes: shard_inventory_byte_totals.shard_file_bytes,
            total_tensor_payload_bytes: shard_inventory_byte_totals.tensor_payload_bytes,
            index_total_size_semantics,
            storage_fingerprint,
        })
    }
}

/// Owns normalized text semantics together with every descriptor that supplied them.
struct ValidatedLagunaTextSidecars {
    text_artifact: LagunaTextArtifactDescriptor,
    tokenizer_file: ValidatedRequiredFile,
    tokenizer_config_file: ValidatedRequiredFile,
    generation_config_file: ValidatedRequiredFile,
    included_template_files: BTreeMap<String, ValidatedRequiredFile>,
}

fn validate_text_sidecars(
    model_directory: &Path,
    target_contract: &LagunaTargetContract,
    config_bytes: &[u8],
    tokenizer_file: ValidatedRequiredFile,
    tokenizer_config_file: ValidatedRequiredFile,
    generation_config_file: ValidatedRequiredFile,
) -> Result<ValidatedLagunaTextSidecars, LagunaArtifactValidationError> {
    // tokenizer.json is captured by the neutral descriptor validator. Borrow that
    // one allocation directly so Laguna does not retain or parse a second byte copy.
    let tokenizer_bytes_fallback;
    let tokenizer_bytes = if let Some(captured_tokenizer_bytes) = tokenizer_file.captured_bytes() {
        captured_tokenizer_bytes
    } else {
        tokenizer_bytes_fallback =
            read_bounded_required_file_bytes(&tokenizer_file, MAXIMUM_TEXT_DOCUMENT_BYTES)?;
        &tokenizer_bytes_fallback
    };
    let tokenizer_config_bytes =
        read_bounded_required_file_bytes(&tokenizer_config_file, MAXIMUM_TEXT_DOCUMENT_BYTES)?;
    let generation_config_bytes =
        read_bounded_required_file_bytes(&generation_config_file, MAXIMUM_TEXT_DOCUMENT_BYTES)?;

    // The graph owner recursively selects and bounded-reads every descriptor once.
    let included_templates =
        LagunaTemplateSourceValidator::new(model_directory).validate(&tokenizer_config_bytes)?;

    let text_artifact = LagunaTextArtifactNormalizer::normalize(
        target_contract,
        LagunaTextArtifactSources {
            model_config_bytes: config_bytes,
            tokenizer_bytes,
            tokenizer_config_bytes: &tokenizer_config_bytes,
            generation_config_bytes: Some(&generation_config_bytes),
            included_template_bytes_by_name: &included_templates.bytes_by_name,
        },
    )?;
    Ok(ValidatedLagunaTextSidecars {
        text_artifact,
        tokenizer_file,
        tokenizer_config_file,
        generation_config_file,
        included_template_files: included_templates.files_by_name,
    })
}

fn validate_shard_descriptors(
    model_directory: &Path,
    shard_index: &LagunaShardIndex,
) -> Result<BTreeMap<String, ValidatedWeightsFile>, LagunaArtifactValidationError> {
    let mut shard_files = BTreeMap::new();
    for shard_file_name in shard_index.shard_file_names() {
        let validated_required_file = match validate_required_file(
            model_directory,
            &required_file_profile(shard_file_name),
        ) {
            Ok(validated_required_file) => validated_required_file,
            Err(ArtifactValidationError::InspectRequiredFile { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                return Err(LagunaArtifactValidationError::MissingShard {
                    shard_file_name: shard_file_name.to_owned(),
                });
            }
            Err(validation_error) => return Err(validation_error.into()),
        };
        shard_files.insert(
            shard_file_name.to_owned(),
            validated_required_file.into_validated_weights_file()?,
        );
    }
    Ok(shard_files)
}

fn inventory_and_validate_ownership(
    shard_index: &LagunaShardIndex,
    shard_files: &BTreeMap<String, ValidatedWeightsFile>,
) -> Result<
    (
        BTreeMap<String, LocatedRawTensorDescriptor>,
        LagunaShardInventoryByteTotals,
    ),
    LagunaArtifactValidationError,
> {
    let mut located_tensors = BTreeMap::new();
    let mut shard_inventory_byte_totals = LagunaShardInventoryByteTotals::default();
    for (shard_file_name, shard_file) in shard_files {
        shard_inventory_byte_totals.shard_file_bytes = shard_inventory_byte_totals
            .shard_file_bytes
            .checked_add(shard_file.size_bytes())
            .ok_or(LagunaArtifactValidationError::ShardFileSizeAccountingOverflow)?;
        let raw_inventory = shard_file.read_raw_safetensors_inventory()?;
        shard_inventory_byte_totals.tensor_payload_bytes = shard_inventory_byte_totals
            .tensor_payload_bytes
            .checked_add(raw_inventory.shard_payload_bytes)
            .ok_or(LagunaArtifactValidationError::TensorPayloadAccountingOverflow)?;
        for raw_tensor in raw_inventory.tensor_descriptors {
            let tensor_name = raw_tensor.tensor_name.clone();
            let located_tensor = LocatedRawTensorDescriptor {
                shard_file_name: shard_file_name.clone(),
                raw_tensor_name: raw_tensor.tensor_name,
                dtype: raw_tensor.dtype,
                shape: raw_tensor.shape,
                data_start_offset_bytes: raw_tensor.data_start_offset_bytes,
                data_end_offset_bytes: raw_tensor.data_end_offset_bytes,
                payload_bytes: raw_tensor.tensor_payload_bytes,
            };
            if let Some(first_location) =
                located_tensors.insert(tensor_name.clone(), located_tensor)
            {
                return Err(LagunaArtifactValidationError::DuplicatePhysicalTensor {
                    tensor_name,
                    first_shard_file_name: first_location.shard_file_name,
                    second_shard_file_name: shard_file_name.clone(),
                });
            }
        }
    }

    for (tensor_name, located_tensor) in &located_tensors {
        let expected_shard_file_name = shard_index
            .shard_file_name_for_tensor(tensor_name)
            .ok_or_else(|| LagunaArtifactValidationError::PhysicalTensorNotIndexed {
                tensor_name: tensor_name.clone(),
            })?;
        if expected_shard_file_name != located_tensor.shard_file_name {
            return Err(LagunaArtifactValidationError::PhysicalTensorInWrongShard {
                tensor_name: tensor_name.clone(),
                expected_shard_file_name: expected_shard_file_name.to_owned(),
                actual_shard_file_name: located_tensor.shard_file_name.clone(),
            });
        }
    }
    for (tensor_name, shard_file_name) in shard_index.tensor_name_to_shard_file_name() {
        if !located_tensors.contains_key(tensor_name) {
            return Err(LagunaArtifactValidationError::IndexedTensorMissing {
                tensor_name: tensor_name.clone(),
                shard_file_name: shard_file_name.clone(),
            });
        }
    }
    Ok((located_tensors, shard_inventory_byte_totals))
}

/// Separate checked totals prevent serialized framing bytes from being mislabeled as tensors.
#[derive(Default)]
struct LagunaShardInventoryByteTotals {
    shard_file_bytes: u64,
    tensor_payload_bytes: u64,
}

fn target_expert_count(
    target_contract: &crate::laguna::LagunaTargetContract,
) -> Result<usize, LagunaArtifactValidationError> {
    let expert_count = target_contract
        .layers()
        .iter()
        .filter_map(|layer| match layer.feed_forward() {
            LagunaFeedForwardDescriptor::Dense(_) => None,
            LagunaFeedForwardDescriptor::Moe(moe) => Some(moe.expert_count()),
        })
        .max()
        .unwrap_or(0);
    usize::try_from(expert_count).map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)
}

fn required_file_profile(file_name: &str) -> RequiredFileProfile {
    RequiredFileProfile {
        file_name: file_name.to_owned(),
        size_bytes: 0,
    }
}

fn remove_required_file(
    required_files: &mut std::collections::HashMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<ValidatedRequiredFile, ArtifactValidationError> {
    required_files.remove(file_name).ok_or_else(|| {
        ArtifactValidationError::ProfileMissingRequiredFile {
            file_name: file_name.to_owned(),
        }
    })
}
