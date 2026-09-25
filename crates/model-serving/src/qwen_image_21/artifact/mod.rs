//! Descriptor-retaining validation of the Qwen-Image-2.1 MLX nested Diffusers tree.
//!
//! The validator opens every required file exactly once, proves the package is the reviewed
//! family (strict configs, exact physical tensor profiles, index/physical size agreement), and
//! hands the engine one owned file handle per component. Rejection messages are path-free so an
//! artifact error never leaks a local directory into a log or a REST response.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use serde::Deserialize;

use crate::artifact_validation::{
    RawSafetensorsInventory, RequiredFileProfile, ValidatedRequiredFile, ValidatedWeightsFile,
    read_bounded_required_file_bytes, validate_required_file,
};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::configuration::{
    QwenImage21PipelineConfig, QwenImage21SchedulerConfig, QwenImage21TextEncoderConfig,
    QwenImage21TransformerConfig, QwenImage21VaeConfig,
};
use super::inventory::{
    QwenImage21TensorInventory, validate_text_encoder_inventory, validate_transformer_inventory,
    validate_vae_inventory,
};

mod error;
mod provenance;

pub use error::QwenImage21ArtifactError;
pub use provenance::{
    QWEN_IMAGE_21_LICENSE_IDENTIFIER, QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
    QWEN_IMAGE_21_PROVIDER_MODEL_ID, QwenImage21ArtifactProvenance, QwenImage21License,
};

const MAXIMUM_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;
const MODEL_INDEX_FILE_NAME: &str = "model_index.json";
const SCHEDULER_CONFIG_FILE_NAME: &str = "scheduler/scheduler_config.json";
const TEXT_ENCODER_CONFIG_FILE_NAME: &str = "text_encoder/config.json";
const TRANSFORMER_CONFIG_FILE_NAME: &str = "transformer/config.json";
const VAE_CONFIG_FILE_NAME: &str = "vae/config.json";
const TEXT_ENCODER_WEIGHTS_FILE_NAME: &str = "text_encoder/model.safetensors";
const TRANSFORMER_WEIGHTS_FILE_NAME: &str = "transformer/model.safetensors";
const VAE_WEIGHTS_FILE_NAME: &str = "vae/model.safetensors";
const TEXT_ENCODER_INDEX_FILE_NAME: &str = "text_encoder/model.safetensors.index.json";
const TRANSFORMER_INDEX_FILE_NAME: &str = "transformer/model.safetensors.index.json";
const VAE_INDEX_FILE_NAME: &str = "vae/model.safetensors.index.json";
/// Every config document the validator opens, assembled from the named paths so a reader never
/// has to cross-reference an index against the call order.
const CONFIG_FILE_NAMES: [&str; 4] = [
    SCHEDULER_CONFIG_FILE_NAME,
    TEXT_ENCODER_CONFIG_FILE_NAME,
    TRANSFORMER_CONFIG_FILE_NAME,
    VAE_CONFIG_FILE_NAME,
];
const INDEX_FILE_NAMES: [&str; 3] = [
    TEXT_ENCODER_INDEX_FILE_NAME,
    TRANSFORMER_INDEX_FILE_NAME,
    VAE_INDEX_FILE_NAME,
];
const PROCESSOR_SIDECAR_FILE_NAMES: [&str; 9] = [
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer_config.json",
    "processor/tokenizer.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
];

/// A validated artifact: the checked configs and the tensor inventories of all three components,
/// with one retained file handle per weight file for the component that will load it.
#[derive(Debug)]
pub struct ValidatedQwenImage21Artifact {
    revision: String,
    license: QwenImage21License,
    pipeline_config: QwenImage21PipelineConfig,
    scheduler_config: QwenImage21SchedulerConfig,
    text_encoder_config: QwenImage21TextEncoderConfig,
    text_encoder_inventory: QwenImage21TensorInventory,
    transformer_config: QwenImage21TransformerConfig,
    transformer_inventory: QwenImage21TensorInventory,
    vae_config: QwenImage21VaeConfig,
    vae_inventory: QwenImage21TensorInventory,
    document_files: BTreeMap<String, ValidatedRequiredFile>,
    processor_sidecars: BTreeMap<String, ValidatedRequiredFile>,
    text_encoder: ValidatedWeightsFile,
    transformer: ValidatedWeightsFile,
    vae: ValidatedWeightsFile,
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

        let mut document_files = BTreeMap::new();
        for file_name in [MODEL_INDEX_FILE_NAME]
            .into_iter()
            .chain(CONFIG_FILE_NAMES)
            .chain(INDEX_FILE_NAMES)
        {
            document_files.insert(
                file_name.to_owned(),
                open_required(model_directory, file_name)?,
            );
        }
        let pipeline_config = QwenImage21PipelineConfig::parse(&read_document(
            &document_files,
            MODEL_INDEX_FILE_NAME,
        )?)?;
        let scheduler_config = QwenImage21SchedulerConfig::parse(&read_document(
            &document_files,
            SCHEDULER_CONFIG_FILE_NAME,
        )?)?;
        let text_encoder_config = QwenImage21TextEncoderConfig::parse(&read_document(
            &document_files,
            TEXT_ENCODER_CONFIG_FILE_NAME,
        )?)?;
        let transformer_config = QwenImage21TransformerConfig::parse(&read_document(
            &document_files,
            TRANSFORMER_CONFIG_FILE_NAME,
        )?)?;
        let vae_config =
            QwenImage21VaeConfig::parse(&read_document(&document_files, VAE_CONFIG_FILE_NAME)?)?;

        let processor_sidecars = PROCESSOR_SIDECAR_FILE_NAMES
            .into_iter()
            .map(|file_name| {
                open_required(model_directory, file_name).map(|file| (file_name.to_owned(), file))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        let (text_encoder, text_encoder_inventory) = validate_weight_component(
            "text_encoder",
            TEXT_ENCODER_WEIGHTS_FILE_NAME,
            TEXT_ENCODER_INDEX_FILE_NAME,
            model_directory,
            &document_files,
            |raw| {
                validate_text_encoder_inventory(
                    TEXT_ENCODER_WEIGHTS_FILE_NAME,
                    raw,
                    &text_encoder_config,
                )
            },
        )?;
        let (transformer, transformer_inventory) = validate_weight_component(
            "transformer",
            TRANSFORMER_WEIGHTS_FILE_NAME,
            TRANSFORMER_INDEX_FILE_NAME,
            model_directory,
            &document_files,
            |raw| {
                validate_transformer_inventory(
                    TRANSFORMER_WEIGHTS_FILE_NAME,
                    raw,
                    &transformer_config,
                )
            },
        )?;
        let (vae, vae_inventory) = validate_weight_component(
            "vae",
            VAE_WEIGHTS_FILE_NAME,
            VAE_INDEX_FILE_NAME,
            model_directory,
            &document_files,
            |raw| validate_vae_inventory(VAE_WEIGHTS_FILE_NAME, raw, &vae_config),
        )?;

        Ok(ValidatedQwenImage21Artifact {
            revision: provenance.revision().to_owned(),
            license: QwenImage21License,
            pipeline_config,
            scheduler_config,
            text_encoder_config,
            text_encoder_inventory,
            transformer_config,
            transformer_inventory,
            vae_config,
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

fn validate_weight_component(
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

fn open_required(
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

fn read_document(
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
