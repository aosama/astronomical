//! Descriptor-retaining validation of a FLUX.2 Klein 4B artifact in either
//! reviewed build: the BF16 Diffusers tree, or the 4-bit MLX build (issue
//! #1904). The validator detects the layout once and dispatches; both paths
//! hand back the same validated artifact, with inventories in Diffusers names,
//! so nothing downstream branches on the layout except the tensor store.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Component, Path};

use crate::artifact_validation::{
    RawSafetensorsInventory, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    read_bounded_required_file_bytes, validate_required_file,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::mlx_4bit_build::validate_mlx_4bit_artifact;
use super::weight_layout::{Flux2KleinComponentWeightFiles, Flux2KleinWeightLayout};
use super::{
    Flux2KleinPipelineConfig, Flux2KleinSchedulerConfig, Flux2KleinTensorInventory,
    Flux2KleinTextEncoderConfig, Flux2KleinTransformerConfig, Flux2KleinVaeConfig,
};

mod diffusers_bf16;
mod error;
mod provenance;

pub use error::Flux2KleinArtifactError;
pub use provenance::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, FLUX2_KLEIN_PROVIDER_MODEL_ID,
    Flux2KleinArtifactProvenance, Flux2KleinLicense,
};

const MAXIMUM_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;

/// The parsed configuration of every pipeline component.
pub(super) struct Flux2KleinComponentConfigs {
    pub(super) pipeline: Flux2KleinPipelineConfig,
    pub(super) scheduler: Flux2KleinSchedulerConfig,
    pub(super) text_encoder: Flux2KleinTextEncoderConfig,
    pub(super) transformer: Flux2KleinTransformerConfig,
    pub(super) vae: Flux2KleinVaeConfig,
}

/// One component's weight files and the inventory proven from them.
pub(super) struct Flux2KleinValidatedComponent {
    pub(super) weight_files: Flux2KleinComponentWeightFiles,
    pub(super) inventory: Flux2KleinTensorInventory,
}

/// Everything a layout validator proves, before it becomes the artifact.
pub(super) struct Flux2KleinValidatedParts {
    pub(super) configs: Flux2KleinComponentConfigs,
    pub(super) document_files: BTreeMap<String, ValidatedRequiredFile>,
    pub(super) tokenizer_sidecars: BTreeMap<String, ValidatedRequiredFile>,
    pub(super) text_encoder: Flux2KleinValidatedComponent,
    pub(super) transformer: Flux2KleinValidatedComponent,
    pub(super) vae: Flux2KleinValidatedComponent,
}

/// Descriptor-backed result before the engine takes component ownership.
#[derive(Debug)]
pub struct ValidatedFlux2KleinArtifact {
    revision: String,
    weight_layout: Flux2KleinWeightLayout,
    license: Flux2KleinLicense,
    pipeline_config: Flux2KleinPipelineConfig,
    text_encoder_config: Flux2KleinTextEncoderConfig,
    text_encoder_inventory: Flux2KleinTensorInventory,
    transformer_config: Flux2KleinTransformerConfig,
    vae_config: Flux2KleinVaeConfig,
    scheduler_config: Flux2KleinSchedulerConfig,
    transformer_inventory: Flux2KleinTensorInventory,
    vae_inventory: Flux2KleinTensorInventory,
    document_files: BTreeMap<String, ValidatedRequiredFile>,
    tokenizer_sidecars: BTreeMap<String, ValidatedRequiredFile>,
    text_encoder_weights: Flux2KleinComponentWeightFiles,
    transformer_weights: Flux2KleinComponentWeightFiles,
    vae_weights: Flux2KleinComponentWeightFiles,
}

impl ValidatedFlux2KleinArtifact {
    pub(super) fn assemble(
        provenance: Flux2KleinArtifactProvenance,
        weight_layout: Flux2KleinWeightLayout,
        parts: Flux2KleinValidatedParts,
    ) -> Self {
        Self {
            revision: provenance.into_revision(),
            weight_layout,
            license: Flux2KleinLicense,
            pipeline_config: parts.configs.pipeline,
            text_encoder_config: parts.configs.text_encoder,
            text_encoder_inventory: parts.text_encoder.inventory,
            transformer_config: parts.configs.transformer,
            vae_config: parts.configs.vae,
            scheduler_config: parts.configs.scheduler,
            transformer_inventory: parts.transformer.inventory,
            vae_inventory: parts.vae.inventory,
            document_files: parts.document_files,
            tokenizer_sidecars: parts.tokenizer_sidecars,
            text_encoder_weights: parts.text_encoder.weight_files,
            transformer_weights: parts.transformer.weight_files,
            vae_weights: parts.vae.weight_files,
        }
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub const fn weight_layout(&self) -> Flux2KleinWeightLayout {
        self.weight_layout
    }
    pub const fn license(&self) -> &Flux2KleinLicense {
        &self.license
    }
    pub const fn pipeline_config(&self) -> &Flux2KleinPipelineConfig {
        &self.pipeline_config
    }
    pub const fn text_encoder_config(&self) -> &Flux2KleinTextEncoderConfig {
        &self.text_encoder_config
    }
    pub const fn text_encoder_inventory(&self) -> &Flux2KleinTensorInventory {
        &self.text_encoder_inventory
    }
    pub const fn transformer_config(&self) -> &Flux2KleinTransformerConfig {
        &self.transformer_config
    }
    pub const fn vae_config(&self) -> &Flux2KleinVaeConfig {
        &self.vae_config
    }
    pub const fn scheduler_config(&self) -> &Flux2KleinSchedulerConfig {
        &self.scheduler_config
    }
    pub const fn transformer_inventory(&self) -> &Flux2KleinTensorInventory {
        &self.transformer_inventory
    }
    pub const fn vae_inventory(&self) -> &Flux2KleinTensorInventory {
        &self.vae_inventory
    }
    pub fn text_shard_count(&self) -> usize {
        self.text_encoder_weights.shard_count()
    }

    pub fn into_retained_files(
        self,
    ) -> Result<Flux2KleinRetainedArtifactFiles, Flux2KleinArtifactError> {
        Ok(Flux2KleinRetainedArtifactFiles {
            document_files: transfer_documents(self.document_files)?,
            tokenizer_sidecars: transfer_documents(self.tokenizer_sidecars)?,
            text_encoder_weights: self.text_encoder_weights,
            transformer_weights: self.transformer_weights,
            vae_weights: self.vae_weights,
        })
    }
}

/// Exact file owners transferred once to the concrete MLX engine.
#[derive(Debug)]
pub struct Flux2KleinRetainedArtifactFiles {
    document_files: BTreeMap<String, File>,
    tokenizer_sidecars: BTreeMap<String, File>,
    text_encoder_weights: Flux2KleinComponentWeightFiles,
    transformer_weights: Flux2KleinComponentWeightFiles,
    vae_weights: Flux2KleinComponentWeightFiles,
}

impl Flux2KleinRetainedArtifactFiles {
    pub const fn document_files(&self) -> &BTreeMap<String, File> {
        &self.document_files
    }
    pub const fn tokenizer_sidecars(&self) -> &BTreeMap<String, File> {
        &self.tokenizer_sidecars
    }
    pub const fn text_encoder_weights(&self) -> &Flux2KleinComponentWeightFiles {
        &self.text_encoder_weights
    }
    pub const fn transformer_weights(&self) -> &Flux2KleinComponentWeightFiles {
        &self.transformer_weights
    }
    pub const fn vae_weights(&self) -> &Flux2KleinComponentWeightFiles {
        &self.vae_weights
    }
    /// The text encoder, transformer and VAE weight files, in that order.
    pub fn into_weight_files(
        self,
    ) -> (
        Flux2KleinComponentWeightFiles,
        Flux2KleinComponentWeightFiles,
        Flux2KleinComponentWeightFiles,
    ) {
        (
            self.text_encoder_weights,
            self.transformer_weights,
            self.vae_weights,
        )
    }
}

#[derive(Debug, Default)]
pub struct Flux2KleinArtifactValidator;

impl Flux2KleinArtifactValidator {
    pub const fn new() -> Self {
        Self
    }

    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
        provenance: Flux2KleinArtifactProvenance,
    ) -> Result<ValidatedFlux2KleinArtifact, Flux2KleinArtifactError> {
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
        provenance: Flux2KleinArtifactProvenance,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<ValidatedFlux2KleinArtifact, Flux2KleinArtifactError> {
        let model_directory = model_directory.as_ref();
        performance_attribution.measure_operation(PerformanceOperation::ArtifactValidation, |_| {
            validate_artifact(model_directory, provenance)
        })
    }
}

fn validate_artifact(
    model_directory: &Path,
    provenance: Flux2KleinArtifactProvenance,
) -> Result<ValidatedFlux2KleinArtifact, Flux2KleinArtifactError> {
    if !model_directory.is_dir() {
        return Err(Flux2KleinArtifactError::ModelDirectoryUnavailable);
    }
    let weight_layout = Flux2KleinWeightLayout::detect(model_directory);
    provenance::validate_provenance(&provenance, weight_layout)?;
    let parts = match weight_layout {
        Flux2KleinWeightLayout::DiffusersBf16 => {
            diffusers_bf16::validate_diffusers_bf16_artifact(model_directory)?
        }
        Flux2KleinWeightLayout::MlxAffine4Bit => validate_mlx_4bit_artifact(model_directory)?,
    };
    Ok(ValidatedFlux2KleinArtifact::assemble(
        provenance,
        weight_layout,
        parts,
    ))
}

pub(super) fn open_required(
    model_directory: &Path,
    file_name: &str,
) -> Result<ValidatedRequiredFile, Flux2KleinArtifactError> {
    validate_required_file(
        model_directory,
        &RequiredFileProfile {
            file_name: file_name.to_owned(),
            size_bytes: 0,
        },
    )
    .map_err(|_| Flux2KleinArtifactError::ArtifactFile {
        file_name: file_name.to_owned(),
    })
}

pub(super) fn open_weights(
    model_directory: &Path,
    file_name: &str,
) -> Result<(ValidatedWeightsFile, RawSafetensorsInventory), Flux2KleinArtifactError> {
    let weights = open_required(model_directory, file_name)?
        .into_validated_weights_file()
        .map_err(|_| Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        })?;
    let inventory = weights.read_raw_safetensors_inventory().map_err(|_| {
        Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        }
    })?;
    Ok((weights, inventory))
}

/// The bytes of one opened document, refusing anything oversized.
pub(super) fn read_required_document(
    file: &ValidatedRequiredFile,
    file_name: &str,
) -> Result<Vec<u8>, Flux2KleinArtifactError> {
    read_bounded_required_file_bytes(file, MAXIMUM_DOCUMENT_BYTES).map_err(|_| {
        Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        }
    })
}

/// Opens each named file, keyed by its name.
pub(super) fn open_named_files(
    model_directory: &Path,
    file_names: &[&str],
) -> Result<BTreeMap<String, ValidatedRequiredFile>, Flux2KleinArtifactError> {
    let mut opened_files = BTreeMap::new();
    for file_name in file_names {
        opened_files.insert(
            (*file_name).to_owned(),
            open_required(model_directory, file_name)?,
        );
    }
    Ok(opened_files)
}

fn transfer_documents(
    files: BTreeMap<String, ValidatedRequiredFile>,
) -> Result<BTreeMap<String, File>, Flux2KleinArtifactError> {
    files
        .into_iter()
        .map(|(file_name, file)| {
            file.into_validated_weights_file()
                .map(ValidatedWeightsFile::into_file)
                .map(|retained| (file_name.clone(), retained))
                .map_err(|_| Flux2KleinArtifactError::ArtifactFile { file_name })
        })
        .collect()
}

/// Whether an index names a shard file safely: a relative `.safetensors` path
/// that stays inside the component folder.
pub(super) fn is_safe_weight_shard_path(shard_file_name: &str) -> bool {
    let shard_path = Path::new(shard_file_name);
    !shard_file_name.is_empty()
        && !shard_file_name.contains('\\')
        && !shard_path.is_absolute()
        && shard_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && shard_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
}
