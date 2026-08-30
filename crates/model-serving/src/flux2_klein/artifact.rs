//! Descriptor-retaining validation of the authoritative nested Diffusers tree.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use thiserror::Error;

use crate::artifact_validation::{
    RawSafetensorsInventory, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    read_bounded_required_file_bytes, validate_required_file,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::artifact_text_validation::{
    TEXT_GENERATION_CONFIG_FILE_NAME, TEXT_INDEX_FILE_NAME, validate_text_artifacts,
};
use super::inventory::{validate_transformer_inventory, validate_vae_inventory};
use super::{
    Flux2KleinConfigError, Flux2KleinPipelineConfig, Flux2KleinSchedulerConfig,
    Flux2KleinTensorInventory, Flux2KleinTextEncoderConfig, Flux2KleinTransformerConfig,
    Flux2KleinVaeConfig,
};

pub const FLUX2_KLEIN_OFFICIAL_MODEL_ID: &str = "FLUX.2-klein-4B";
pub const FLUX2_KLEIN_PROVIDER_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
pub const FLUX2_KLEIN_OFFICIAL_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";
const OFFICIAL_LICENSE_IDENTIFIER: &str = "Apache-2.0";
const MAXIMUM_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;
const MODEL_INDEX_FILE_NAME: &str = "model_index.json";
const LICENSE_FILE_NAME: &str = "LICENSE.md";
const TRANSFORMER_FILE_NAME: &str = "transformer/diffusion_pytorch_model.safetensors";
const VAE_FILE_NAME: &str = "vae/diffusion_pytorch_model.safetensors";
const CONFIG_FILE_NAMES: [&str; 4] = [
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
];
const TOKENIZER_SIDECAR_FILE_NAMES: [&str; 7] = [
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
];

/// Discovery provenance required before local bytes may claim the official profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinArtifactProvenance {
    model_id: String,
    revision: String,
    license_identifier: String,
}

impl Flux2KleinArtifactProvenance {
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        license_identifier: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            license_identifier: license_identifier.into(),
        }
    }

    pub fn official() -> Self {
        Self::new(
            FLUX2_KLEIN_PROVIDER_MODEL_ID,
            FLUX2_KLEIN_OFFICIAL_REVISION,
            OFFICIAL_LICENSE_IDENTIFIER,
        )
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }
}

/// License metadata exposed without treating a directory or model name as authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinLicense;

impl Flux2KleinLicense {
    pub const fn identifier(&self) -> &'static str {
        OFFICIAL_LICENSE_IDENTIFIER
    }
    pub const fn display_name(&self) -> &'static str {
        "Apache License 2.0"
    }
    pub const fn canonical_url(&self) -> &'static str {
        "https://www.apache.org/licenses/LICENSE-2.0"
    }
}

/// A bounded artifact rejection whose display never includes a local path.
#[derive(Debug, Error)]
pub enum Flux2KleinArtifactError {
    #[error("FLUX.2 Klein artifact directory is unavailable")]
    ModelDirectoryUnavailable,
    #[error("unsupported FLUX.2 Klein model, revision, or license provenance")]
    UnsupportedProvenance {
        model_id: String,
        revision: String,
        license_identifier: String,
    },
    #[error("required FLUX.2 Klein artifact file '{file_name}' is unavailable or invalid")]
    ArtifactFile { file_name: String },
    #[error("FLUX.2 Klein configuration is incompatible")]
    Configuration(#[from] Flux2KleinConfigError),
    #[error("FLUX.2 Klein LICENSE.md is not the Apache License 2.0 text")]
    InvalidLicense,
    #[error("malformed FLUX.2 Klein text shard index")]
    MalformedTextShardIndex(#[source] serde_json::Error),
    #[error("FLUX.2 Klein text shard index contains an unsafe or unsupported shard name")]
    UnsupportedTextShardName { shard_file_name: String },
    #[error("FLUX.2 Klein text shard index and physical inventory disagree for '{tensor_name}'")]
    TextShardIndexDisagreement { tensor_name: String },
    #[error("FLUX.2 Klein text shard index total_size disagrees with physical tensor payload")]
    TextShardIndexTotalSizeMismatch {
        declared_bytes: u64,
        actual_bytes: u64,
    },
    #[error(
        "FLUX.2 Klein text shard index total_parameters disagrees with physical tensor geometry"
    )]
    TextShardIndexTotalParameterMismatch {
        declared_parameters: u64,
        actual_parameters: u64,
    },
    #[error("FLUX.2 Klein {component} tensor '{tensor_name}' must use BF16 storage")]
    TensorDtype {
        component: &'static str,
        tensor_name: String,
    },
    #[error(
        "FLUX.2 Klein {component} tensor '{tensor_name}' shape does not match model configuration"
    )]
    TensorShape {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein {component} tensor inventory is missing '{tensor_name}'")]
    MissingTensor {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein {component} tensor inventory contains unsupported '{tensor_name}'")]
    UnsupportedTensor {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein aggregate tensor payload accounting overflowed")]
    PayloadAccountingOverflow,
}

/// Descriptor-backed result before the future engine takes component ownership.
#[derive(Debug)]
pub struct ValidatedFlux2KleinArtifact {
    revision: String,
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
    text_shards: BTreeMap<String, ValidatedWeightsFile>,
    transformer: ValidatedWeightsFile,
    vae: ValidatedWeightsFile,
}

impl ValidatedFlux2KleinArtifact {
    pub fn revision(&self) -> &str {
        &self.revision
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
        self.text_shards.len()
    }

    pub fn into_retained_files(
        self,
    ) -> Result<Flux2KleinRetainedArtifactFiles, Flux2KleinArtifactError> {
        Ok(Flux2KleinRetainedArtifactFiles {
            document_files: transfer_documents(self.document_files)?,
            tokenizer_sidecars: transfer_documents(self.tokenizer_sidecars)?,
            text_shards: self.text_shards,
            transformer: self.transformer,
            vae: self.vae,
        })
    }
}

/// Exact file owners transferred once to the concrete MLX engine.
#[derive(Debug)]
pub struct Flux2KleinRetainedArtifactFiles {
    document_files: BTreeMap<String, File>,
    tokenizer_sidecars: BTreeMap<String, File>,
    text_shards: BTreeMap<String, ValidatedWeightsFile>,
    transformer: ValidatedWeightsFile,
    vae: ValidatedWeightsFile,
}

impl Flux2KleinRetainedArtifactFiles {
    pub const fn document_files(&self) -> &BTreeMap<String, File> {
        &self.document_files
    }
    pub const fn tokenizer_sidecars(&self) -> &BTreeMap<String, File> {
        &self.tokenizer_sidecars
    }
    pub const fn text_shards(&self) -> &BTreeMap<String, ValidatedWeightsFile> {
        &self.text_shards
    }
    pub const fn transformer(&self) -> &ValidatedWeightsFile {
        &self.transformer
    }
    pub const fn vae(&self) -> &ValidatedWeightsFile {
        &self.vae
    }
    pub fn into_weight_files(
        self,
    ) -> (
        BTreeMap<String, ValidatedWeightsFile>,
        ValidatedWeightsFile,
        ValidatedWeightsFile,
    ) {
        (self.text_shards, self.transformer, self.vae)
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
            self.validate_inner(model_directory, provenance)
        })
    }

    fn validate_inner(
        self,
        model_directory: &Path,
        provenance: Flux2KleinArtifactProvenance,
    ) -> Result<ValidatedFlux2KleinArtifact, Flux2KleinArtifactError> {
        if !model_directory.is_dir() {
            return Err(Flux2KleinArtifactError::ModelDirectoryUnavailable);
        }
        validate_provenance(&provenance)?;

        let mut document_files = BTreeMap::new();
        for file_name in [
            MODEL_INDEX_FILE_NAME,
            LICENSE_FILE_NAME,
            TEXT_INDEX_FILE_NAME,
            TEXT_GENERATION_CONFIG_FILE_NAME,
        ]
        .into_iter()
        .chain(CONFIG_FILE_NAMES)
        {
            document_files.insert(
                file_name.to_owned(),
                open_required(model_directory, file_name)?,
            );
        }
        let model_index_bytes = read_document(&document_files, MODEL_INDEX_FILE_NAME)?;
        let pipeline_config = Flux2KleinPipelineConfig::parse(&model_index_bytes)?;
        let scheduler_config = Flux2KleinSchedulerConfig::parse(&read_document(
            &document_files,
            CONFIG_FILE_NAMES[0],
        )?)?;
        let text_encoder_config = Flux2KleinTextEncoderConfig::parse(&read_document(
            &document_files,
            CONFIG_FILE_NAMES[1],
        )?)?;
        let transformer_config = Flux2KleinTransformerConfig::parse(&read_document(
            &document_files,
            CONFIG_FILE_NAMES[2],
        )?)?;
        let vae_config =
            Flux2KleinVaeConfig::parse(&read_document(&document_files, CONFIG_FILE_NAMES[3])?)?;
        validate_license(&read_document(&document_files, LICENSE_FILE_NAME)?)?;

        let tokenizer_sidecars = TOKENIZER_SIDECAR_FILE_NAMES
            .into_iter()
            .map(|file_name| {
                open_required(model_directory, file_name).map(|file| (file_name.to_owned(), file))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let (text_shards, text_encoder_inventory) =
            validate_text_artifacts(model_directory, &document_files)?;
        let (transformer, transformer_raw_inventory) =
            open_weights(model_directory, TRANSFORMER_FILE_NAME)?;
        let transformer_inventory = validate_transformer_inventory(
            TRANSFORMER_FILE_NAME,
            transformer_raw_inventory,
            &transformer_config,
        )?;
        let (vae, vae_raw_inventory) = open_weights(model_directory, VAE_FILE_NAME)?;
        let vae_inventory = validate_vae_inventory(VAE_FILE_NAME, vae_raw_inventory)?;
        // `flux-2-klein-4b.safetensors` is an alternate single-file packaging,
        // not a second owner in the authoritative Diffusers component graph.
        Ok(ValidatedFlux2KleinArtifact {
            revision: provenance.revision,
            license: Flux2KleinLicense,
            pipeline_config,
            text_encoder_config,
            text_encoder_inventory,
            transformer_config,
            vae_config,
            scheduler_config,
            transformer_inventory,
            vae_inventory,
            document_files,
            tokenizer_sidecars,
            text_shards,
            transformer,
            vae,
        })
    }
}

fn provenance_identifies_flux2_klein(model_id: &str) -> bool {
    model_id == FLUX2_KLEIN_OFFICIAL_MODEL_ID
        || model_id == FLUX2_KLEIN_PROVIDER_MODEL_ID
        || model_id.rsplit('/').next() == Some(FLUX2_KLEIN_OFFICIAL_MODEL_ID)
}

fn provenance_records_immutable_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_provenance(
    provenance: &Flux2KleinArtifactProvenance,
) -> Result<(), Flux2KleinArtifactError> {
    // Architecture files prove Klein 4B. Provenance only has to name that family,
    // record an immutable revision, and keep Apache-2.0. One Hub SHA is not a gate.
    if provenance.license_identifier == OFFICIAL_LICENSE_IDENTIFIER
        && provenance_identifies_flux2_klein(&provenance.model_id)
        && provenance_records_immutable_revision(&provenance.revision)
    {
        return Ok(());
    }
    Err(Flux2KleinArtifactError::UnsupportedProvenance {
        model_id: provenance.model_id.clone(),
        revision: provenance.revision.clone(),
        license_identifier: provenance.license_identifier.clone(),
    })
}

fn validate_license(bytes: &[u8]) -> Result<(), Flux2KleinArtifactError> {
    let text = std::str::from_utf8(bytes).map_err(|_| Flux2KleinArtifactError::InvalidLicense)?;
    if text.contains("Apache License")
        && text.contains("Version 2.0, January 2004")
        && text.contains("END OF TERMS AND CONDITIONS")
    {
        Ok(())
    } else {
        Err(Flux2KleinArtifactError::InvalidLicense)
    }
}

fn open_required(
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

fn open_weights(
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

fn read_document(
    files: &BTreeMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<Vec<u8>, Flux2KleinArtifactError> {
    let file = files
        .get(file_name)
        .ok_or_else(|| Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        })?;
    read_bounded_required_file_bytes(file, MAXIMUM_DOCUMENT_BYTES).map_err(|_| {
        Flux2KleinArtifactError::ArtifactFile {
            file_name: file_name.to_owned(),
        }
    })
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
