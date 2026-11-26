//! The validator itself: provenance, file opening, config parsing and the per-component weight
//! checks, run in a fixed order so the first rejected stage decides the error.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use artifact_validation::{
    RawSafetensorsInventory, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    read_bounded_required_file_bytes, validate_required_file,
};
use model_primitives::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::component_validation::{
    open_document_files, open_processor_sidecars, parse_component_configs,
    validate_text_encoder_component, validate_transformer_component, validate_vae_component,
};
use super::{
    QWEN_IMAGE_21_LICENSE_IDENTIFIER, QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
    QWEN_IMAGE_21_PROVIDER_MODEL_ID, QwenImage21ArtifactError, QwenImage21ArtifactProvenance,
    QwenImage21License, ValidatedQwenImage21Artifact,
};
use crate::inventory::QwenImage21TensorInventory;

const MAXIMUM_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct QwenImage21ArtifactValidator;

impl QwenImage21ArtifactValidator {
    pub const fn new() -> Self {
        Self
    }

    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
        provenance: QwenImage21ArtifactProvenance,
    ) -> Result<ValidatedQwenImage21Artifact, QwenImage21ArtifactError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        self.validate_with_performance_attribution(
            model_directory,
            provenance,
            &mut performance_attribution,
        )
    }

    pub fn validate_with_performance_attribution(
        self,
        model_directory: impl AsRef<Path>,
        provenance: QwenImage21ArtifactProvenance,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<ValidatedQwenImage21Artifact, QwenImage21ArtifactError> {
        let model_directory = model_directory.as_ref();
        performance_attribution.measure_operation(PerformanceOperation::ArtifactValidation, |_| {
            self.validate_inner(model_directory, provenance)
        })
    }

    fn validate_inner(
        self,
        model_directory: &Path,
        provenance: QwenImage21ArtifactProvenance,
    ) -> Result<ValidatedQwenImage21Artifact, QwenImage21ArtifactError> {
        if !model_directory.is_dir() {
            return Err(QwenImage21ArtifactError::ModelDirectoryUnavailable);
        }
        validate_provenance(&provenance)?;

        let document_files = open_document_files(model_directory)?;
        let configs = parse_component_configs(&document_files)?;
        let processor_sidecars = open_processor_sidecars(model_directory)?;

        let (text_encoder, text_encoder_inventory) = validate_text_encoder_component(
            model_directory,
            &document_files,
            &configs.text_encoder,
        )?;
        let (transformer, transformer_inventory) =
            validate_transformer_component(model_directory, &document_files, &configs.transformer)?;
        let (vae, vae_inventory) =
            validate_vae_component(model_directory, &document_files, &configs.vae)?;

        Ok(ValidatedQwenImage21Artifact {
            revision: provenance.revision().to_owned(),
            license: QwenImage21License,
            pipeline_config: configs.pipeline,
            scheduler_config: configs.scheduler,
            text_encoder_config: configs.text_encoder,
            text_encoder_inventory,
            transformer_config: configs.transformer,
            transformer_inventory,
            vae_config: configs.vae,
            vae_inventory,
            document_files,
            processor_sidecars,
            text_encoder,
            transformer,
            vae,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ShardIndexMetadataDocument {
    total_size: u64,
}

#[derive(Debug, Deserialize)]
struct ShardIndexDocument {
    metadata: ShardIndexMetadataDocument,
    weight_map: BTreeMap<String, String>,
}

/// Index bookkeeping: single reviewed shard, and `total_size` must equal the physical payload
/// the raw inventory already measured against the retained file.
fn validate_shard_index(
    component: &'static str,
    index_bytes: &[u8],
    actual_payload_bytes: u64,
) -> Result<(), QwenImage21ArtifactError> {
    let document: ShardIndexDocument = serde_json::from_slice(index_bytes)
        .map_err(|source| QwenImage21ArtifactError::MalformedShardIndex { component, source })?;
    if let Some((tensor_name, shard_file_name)) = document
        .weight_map
        .iter()
        .find(|(_, shard_file_name)| shard_file_name.as_str() != "model.safetensors")
    {
        return Err(QwenImage21ArtifactError::UnexpectedShardName {
            component,
            tensor_name: tensor_name.clone(),
            shard_file_name: shard_file_name.clone(),
        });
    }
    if document.metadata.total_size != actual_payload_bytes {
        return Err(QwenImage21ArtifactError::ShardIndexTotalSizeMismatch {
            component,
            declared_bytes: document.metadata.total_size,
            actual_bytes: actual_payload_bytes,
        });
    }
    Ok(())
}

pub(super) fn validate_weight_component(
    component: &'static str,
    weights_file_name: &str,
    index_file_name: &str,
    model_directory: &Path,
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
    validate_inventory: impl FnOnce(
        RawSafetensorsInventory,
    ) -> Result<QwenImage21TensorInventory, QwenImage21ArtifactError>,
) -> Result<(ValidatedWeightsFile, QwenImage21TensorInventory), QwenImage21ArtifactError> {
    let index_bytes = read_document(document_files, index_file_name)?;
    let weights = open_required(model_directory, weights_file_name)?
        .into_validated_weights_file()
        .map_err(|_| QwenImage21ArtifactError::ArtifactFile {
            file_name: weights_file_name.to_owned(),
        })?;
    let raw_inventory = weights.read_raw_safetensors_inventory().map_err(|_| {
        QwenImage21ArtifactError::ArtifactFile {
            file_name: weights_file_name.to_owned(),
        }
    })?;
    let inventory = validate_inventory(raw_inventory)?;
    validate_shard_index(component, &index_bytes, inventory.payload_bytes())?;
    Ok((weights, inventory))
}

fn provenance_identifies_qwen_image_21(model_id: &str) -> bool {
    model_id == QWEN_IMAGE_21_OFFICIAL_MODEL_ID
        || model_id == QWEN_IMAGE_21_PROVIDER_MODEL_ID
        || model_id.rsplit('/').next() == Some(QWEN_IMAGE_21_OFFICIAL_MODEL_ID)
}

fn provenance_records_immutable_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_provenance(
    provenance: &QwenImage21ArtifactProvenance,
) -> Result<(), QwenImage21ArtifactError> {
    // Architecture files and exact tensor profiles prove the reviewed package. Provenance only
    // has to name that family, record an immutable revision, and keep the research license.
    if provenance.license_identifier() == QWEN_IMAGE_21_LICENSE_IDENTIFIER
        && provenance_identifies_qwen_image_21(provenance.model_id())
        && provenance_records_immutable_revision(provenance.revision())
    {
        return Ok(());
    }
    Err(QwenImage21ArtifactError::UnsupportedProvenance {
        model_id: provenance.model_id().to_owned(),
        revision: provenance.revision().to_owned(),
        license_identifier: provenance.license_identifier().to_owned(),
    })
}

pub(super) fn open_required(
    model_directory: &Path,
    file_name: &str,
) -> Result<ValidatedRequiredFile, QwenImage21ArtifactError> {
    validate_required_file(
        model_directory,
        &RequiredFileProfile {
            file_name: file_name.to_owned(),
            // Zero means "no expected size": every document here is bounded later by
            // `read_document`, whose bound is the security-relevant one.
            size_bytes: 0,
        },
    )
    .map_err(|_| QwenImage21ArtifactError::ArtifactFile {
        file_name: file_name.to_owned(),
    })
}

pub(super) fn read_document(
    files: &BTreeMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<Vec<u8>, QwenImage21ArtifactError> {
    let file = files
        .get(file_name)
        .ok_or_else(|| QwenImage21ArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        })?;
    read_bounded_required_file_bytes(file, MAXIMUM_DOCUMENT_BYTES).map_err(|_| {
        QwenImage21ArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        }
    })
}
