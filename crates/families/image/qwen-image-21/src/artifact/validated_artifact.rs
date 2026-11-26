//! The validated artifact and the retained file handles it hands to the concrete MLX engine.
//! The validator builds the artifact once every stage has passed; nothing else constructs it.

use std::collections::BTreeMap;
use std::fs::File;

use artifact_validation::{ValidatedRequiredFile, ValidatedWeightsFile};

use super::{QwenImage21ArtifactError, QwenImage21License};
use crate::configuration::{
    QwenImage21PipelineConfig, QwenImage21SchedulerConfig, QwenImage21TextEncoderConfig,
    QwenImage21TransformerConfig, QwenImage21VaeConfig,
};
use crate::inventory::QwenImage21TensorInventory;

/// A validated artifact: the checked configs and the tensor inventories of all three components,
/// with one retained file handle per weight file for the component that will load it.
#[derive(Debug)]
pub struct ValidatedQwenImage21Artifact {
    pub(super) revision: String,
    pub(super) license: QwenImage21License,
    pub(super) pipeline_config: QwenImage21PipelineConfig,
    pub(super) scheduler_config: QwenImage21SchedulerConfig,
    pub(super) text_encoder_config: QwenImage21TextEncoderConfig,
    pub(super) text_encoder_inventory: QwenImage21TensorInventory,
    pub(super) transformer_config: QwenImage21TransformerConfig,
    pub(super) transformer_inventory: QwenImage21TensorInventory,
    pub(super) vae_config: QwenImage21VaeConfig,
    pub(super) vae_inventory: QwenImage21TensorInventory,
    pub(super) document_files: BTreeMap<String, ValidatedRequiredFile>,
    pub(super) processor_sidecars: BTreeMap<String, ValidatedRequiredFile>,
    pub(super) text_encoder: ValidatedWeightsFile,
    pub(super) transformer: ValidatedWeightsFile,
    pub(super) vae: ValidatedWeightsFile,
}

impl ValidatedQwenImage21Artifact {
    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub const fn license(&self) -> &QwenImage21License {
        &self.license
    }

    pub const fn pipeline_config(&self) -> &QwenImage21PipelineConfig {
        &self.pipeline_config
    }

    pub const fn scheduler_config(&self) -> &QwenImage21SchedulerConfig {
        &self.scheduler_config
    }

    pub const fn text_encoder_config(&self) -> &QwenImage21TextEncoderConfig {
        &self.text_encoder_config
    }

    pub const fn text_encoder_inventory(&self) -> &QwenImage21TensorInventory {
        &self.text_encoder_inventory
    }

    pub const fn transformer_config(&self) -> &QwenImage21TransformerConfig {
        &self.transformer_config
    }

    pub const fn transformer_inventory(&self) -> &QwenImage21TensorInventory {
        &self.transformer_inventory
    }

    pub const fn vae_config(&self) -> &QwenImage21VaeConfig {
        &self.vae_config
    }

    pub const fn vae_inventory(&self) -> &QwenImage21TensorInventory {
        &self.vae_inventory
    }

    pub fn into_retained_files(
        self,
    ) -> Result<QwenImage21RetainedArtifactFiles, QwenImage21ArtifactError> {
        Ok(QwenImage21RetainedArtifactFiles {
            document_files: transfer_documents(self.document_files)?,
            processor_sidecars: transfer_documents(self.processor_sidecars)?,
            text_encoder: self.text_encoder,
            transformer: self.transformer,
            vae: self.vae,
        })
    }
}

/// Exact file owners transferred once to the concrete MLX engine.
#[derive(Debug)]
pub struct QwenImage21RetainedArtifactFiles {
    document_files: BTreeMap<String, File>,
    processor_sidecars: BTreeMap<String, File>,
    text_encoder: ValidatedWeightsFile,
    transformer: ValidatedWeightsFile,
    vae: ValidatedWeightsFile,
}

impl QwenImage21RetainedArtifactFiles {
    pub const fn document_files(&self) -> &BTreeMap<String, File> {
        &self.document_files
    }

    pub const fn processor_sidecars(&self) -> &BTreeMap<String, File> {
        &self.processor_sidecars
    }

    pub const fn text_encoder(&self) -> &ValidatedWeightsFile {
        &self.text_encoder
    }

    pub const fn transformer(&self) -> &ValidatedWeightsFile {
        &self.transformer
    }

    pub const fn vae(&self) -> &ValidatedWeightsFile {
        &self.vae
    }
}

fn transfer_documents(
    files: BTreeMap<String, ValidatedRequiredFile>,
) -> Result<BTreeMap<String, File>, QwenImage21ArtifactError> {
    files
        .into_iter()
        .map(|(file_name, file)| {
            file.into_validated_weights_file()
                .map(ValidatedWeightsFile::into_file)
                .map(|retained| (file_name.clone(), retained))
                .map_err(|_| QwenImage21ArtifactError::ArtifactFile { file_name })
        })
        .collect()
}
