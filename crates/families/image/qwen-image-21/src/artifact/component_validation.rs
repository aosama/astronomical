//! The stages of artifact validation that open files, parse the component configs and check each
//! weight component. They sit beside `validator.rs` so the validator itself reads as the sequence
//! of stages, each one failing with the same path-free rejection the whole validator uses.

use std::collections::BTreeMap;
use std::path::Path;

use artifact_validation::{ValidatedRequiredFile, ValidatedWeightsFile};

use super::validator::{open_required, read_document, validate_weight_component};
use super::{
    CONFIG_FILE_NAMES, INDEX_FILE_NAMES, MODEL_INDEX_FILE_NAME, PROCESSOR_SIDECAR_FILE_NAMES,
    QwenImage21ArtifactError, SCHEDULER_CONFIG_FILE_NAME, TEXT_ENCODER_CONFIG_FILE_NAME,
    TEXT_ENCODER_INDEX_FILE_NAME, TEXT_ENCODER_WEIGHTS_FILE_NAME, TRANSFORMER_CONFIG_FILE_NAME,
    TRANSFORMER_INDEX_FILE_NAME, TRANSFORMER_WEIGHTS_FILE_NAME, VAE_CONFIG_FILE_NAME,
    VAE_INDEX_FILE_NAME, VAE_WEIGHTS_FILE_NAME,
};
use crate::configuration::{
    QwenImage21PipelineConfig, QwenImage21SchedulerConfig, QwenImage21TextEncoderConfig,
    QwenImage21TransformerConfig, QwenImage21VaeConfig,
};
use crate::inventory::{
    QwenImage21TensorInventory, validate_text_encoder_inventory, validate_transformer_inventory,
    validate_vae_inventory,
};

/// The retained weight file of one component together with its validated tensor inventory.
pub(super) type ValidatedWeightComponent = (ValidatedWeightsFile, QwenImage21TensorInventory);

/// The five parsed config documents of the artifact.
pub(super) struct ComponentConfigs {
    pub(super) pipeline: QwenImage21PipelineConfig,
    pub(super) scheduler: QwenImage21SchedulerConfig,
    pub(super) text_encoder: QwenImage21TextEncoderConfig,
    pub(super) transformer: QwenImage21TransformerConfig,
    pub(super) vae: QwenImage21VaeConfig,
}

/// Opens each named file exactly once and keys the handles by their artifact-relative name.
/// The first file that cannot be opened fails the whole call, so later names are never touched.
fn open_required_files<'a>(
    model_directory: &Path,
    file_names: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeMap<String, ValidatedRequiredFile>, QwenImage21ArtifactError> {
    let mut opened_files = BTreeMap::new();
    for file_name in file_names {
        opened_files.insert(
            file_name.to_owned(),
            open_required(model_directory, file_name)?,
        );
    }
    Ok(opened_files)
}

/// Opens every document the validator reads later: the model index, the component configs and
/// the shard indexes.
pub(super) fn open_document_files(
    model_directory: &Path,
) -> Result<BTreeMap<String, ValidatedRequiredFile>, QwenImage21ArtifactError> {
    open_required_files(
        model_directory,
        [MODEL_INDEX_FILE_NAME]
            .into_iter()
            .chain(CONFIG_FILE_NAMES)
            .chain(INDEX_FILE_NAMES),
    )
}

pub(super) fn open_processor_sidecars(
    model_directory: &Path,
) -> Result<BTreeMap<String, ValidatedRequiredFile>, QwenImage21ArtifactError> {
    open_required_files(model_directory, PROCESSOR_SIDECAR_FILE_NAMES)
}

/// Parses the model index, then the scheduler, text encoder, transformer and VAE configs, in
/// that order, so the first rejected document decides the error.
pub(super) fn parse_component_configs(
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
) -> Result<ComponentConfigs, QwenImage21ArtifactError> {
    Ok(ComponentConfigs {
        pipeline: QwenImage21PipelineConfig::parse(&read_document(
            document_files,
            MODEL_INDEX_FILE_NAME,
        )?)?,
        scheduler: QwenImage21SchedulerConfig::parse(&read_document(
            document_files,
            SCHEDULER_CONFIG_FILE_NAME,
        )?)?,
        text_encoder: QwenImage21TextEncoderConfig::parse(&read_document(
            document_files,
            TEXT_ENCODER_CONFIG_FILE_NAME,
        )?)?,
        transformer: QwenImage21TransformerConfig::parse(&read_document(
            document_files,
            TRANSFORMER_CONFIG_FILE_NAME,
        )?)?,
        vae: QwenImage21VaeConfig::parse(&read_document(document_files, VAE_CONFIG_FILE_NAME)?)?,
    })
}

pub(super) fn validate_text_encoder_component(
    model_directory: &Path,
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
    config: &QwenImage21TextEncoderConfig,
) -> Result<ValidatedWeightComponent, QwenImage21ArtifactError> {
    validate_weight_component(
        "text_encoder",
        TEXT_ENCODER_WEIGHTS_FILE_NAME,
        TEXT_ENCODER_INDEX_FILE_NAME,
        model_directory,
        document_files,
        |raw| validate_text_encoder_inventory(TEXT_ENCODER_WEIGHTS_FILE_NAME, raw, config),
    )
}

pub(super) fn validate_transformer_component(
    model_directory: &Path,
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
    config: &QwenImage21TransformerConfig,
) -> Result<ValidatedWeightComponent, QwenImage21ArtifactError> {
    validate_weight_component(
        "transformer",
        TRANSFORMER_WEIGHTS_FILE_NAME,
        TRANSFORMER_INDEX_FILE_NAME,
        model_directory,
        document_files,
        |raw| validate_transformer_inventory(TRANSFORMER_WEIGHTS_FILE_NAME, raw, config),
    )
}

pub(super) fn validate_vae_component(
    model_directory: &Path,
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
    config: &QwenImage21VaeConfig,
) -> Result<ValidatedWeightComponent, QwenImage21ArtifactError> {
    validate_weight_component(
        "vae",
        VAE_WEIGHTS_FILE_NAME,
        VAE_INDEX_FILE_NAME,
        model_directory,
        document_files,
        |raw| validate_vae_inventory(VAE_WEIGHTS_FILE_NAME, raw, config),
    )
}

#[cfg(test)]
#[path = "component_validation_tests.rs"]
mod component_validation_tests;
