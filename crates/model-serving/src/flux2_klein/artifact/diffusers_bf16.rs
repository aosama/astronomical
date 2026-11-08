//! Validation of the BF16 Diffusers build: the authoritative nested tree with
//! its configuration documents, license text, sharded text encoder and
//! single-file transformer and VAE.

use std::collections::BTreeMap;
use std::path::Path;

use crate::artifact_validation::ValidatedRequiredFile;

use super::super::artifact_text_validation::{
    TEXT_GENERATION_CONFIG_FILE_NAME, TEXT_INDEX_FILE_NAME, validate_text_artifacts,
};
use super::super::inventory;
use super::super::weight_layout::{
    Flux2KleinComponent, Flux2KleinComponentWeightFiles, Flux2KleinWeightLayout,
};
use super::super::{
    Flux2KleinPipelineConfig, Flux2KleinSchedulerConfig, Flux2KleinTextEncoderConfig,
    Flux2KleinTransformerConfig, Flux2KleinVaeConfig,
};
use super::{
    Flux2KleinArtifactError, Flux2KleinComponentConfigs, Flux2KleinValidatedComponent,
    Flux2KleinValidatedParts, open_named_files, open_weights, read_required_document,
};

const MODEL_INDEX_FILE_NAME: &str = "model_index.json";
const LICENSE_FILE_NAME: &str = "LICENSE.md";
const TRANSFORMER_FILE_NAME: &str = "transformer/diffusion_pytorch_model.safetensors";
const VAE_FILE_NAME: &str = "vae/diffusion_pytorch_model.safetensors";
const SCHEDULER_CONFIG_FILE_NAME: &str = "scheduler/scheduler_config.json";
const TEXT_ENCODER_CONFIG_FILE_NAME: &str = "text_encoder/config.json";
const TRANSFORMER_CONFIG_FILE_NAME: &str = "transformer/config.json";
const VAE_CONFIG_FILE_NAME: &str = "vae/config.json";
const DOCUMENT_FILE_NAMES: [&str; 8] = [
    MODEL_INDEX_FILE_NAME,
    LICENSE_FILE_NAME,
    TEXT_INDEX_FILE_NAME,
    TEXT_GENERATION_CONFIG_FILE_NAME,
    SCHEDULER_CONFIG_FILE_NAME,
    TEXT_ENCODER_CONFIG_FILE_NAME,
    TRANSFORMER_CONFIG_FILE_NAME,
    VAE_CONFIG_FILE_NAME,
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

pub(super) fn validate_diffusers_bf16_artifact(
    model_directory: &Path,
) -> Result<Flux2KleinValidatedParts, Flux2KleinArtifactError> {
    let document_files = open_named_files(model_directory, &DOCUMENT_FILE_NAMES)?;
    let configs = parse_configs(&document_files)?;
    validate_license(&read_document(&document_files, LICENSE_FILE_NAME)?)?;
    let tokenizer_sidecars = open_named_files(model_directory, &TOKENIZER_SIDECAR_FILE_NAMES)?;
    let (text_shards, text_encoder_inventory) =
        validate_text_artifacts(model_directory, &document_files)?;
    let text_encoder = Flux2KleinValidatedComponent {
        weight_files: bf16_weight_files(
            Flux2KleinComponent::TextEncoder,
            text_shards.into_values().collect(),
        ),
        inventory: text_encoder_inventory,
    };
    let (transformer_file, transformer_raw_inventory) =
        open_weights(model_directory, TRANSFORMER_FILE_NAME)?;
    let transformer = Flux2KleinValidatedComponent {
        inventory: inventory::validate_transformer_inventory(
            TRANSFORMER_FILE_NAME,
            transformer_raw_inventory,
            &configs.transformer,
        )?,
        weight_files: bf16_weight_files(Flux2KleinComponent::Transformer, vec![transformer_file]),
    };
    // `flux-2-klein-4b.safetensors` is an alternate single-file packaging,
    // not a second owner in the authoritative Diffusers component graph.
    let (vae_file, vae_raw_inventory) = open_weights(model_directory, VAE_FILE_NAME)?;
    let vae = Flux2KleinValidatedComponent {
        inventory: inventory::validate_vae_inventory(VAE_FILE_NAME, vae_raw_inventory)?,
        weight_files: bf16_weight_files(Flux2KleinComponent::Vae, vec![vae_file]),
    };
    Ok(Flux2KleinValidatedParts {
        configs,
        document_files,
        tokenizer_sidecars,
        text_encoder,
        transformer,
        vae,
    })
}

fn parse_configs(
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
) -> Result<Flux2KleinComponentConfigs, Flux2KleinArtifactError> {
    Ok(Flux2KleinComponentConfigs {
        pipeline: Flux2KleinPipelineConfig::parse(&read_document(
            document_files,
            MODEL_INDEX_FILE_NAME,
        )?)?,
        scheduler: Flux2KleinSchedulerConfig::parse(&read_document(
            document_files,
            SCHEDULER_CONFIG_FILE_NAME,
        )?)?,
        text_encoder: Flux2KleinTextEncoderConfig::parse(&read_document(
            document_files,
            TEXT_ENCODER_CONFIG_FILE_NAME,
        )?)?,
        transformer: Flux2KleinTransformerConfig::parse(&read_document(
            document_files,
            TRANSFORMER_CONFIG_FILE_NAME,
        )?)?,
        vae: Flux2KleinVaeConfig::parse(&read_document(document_files, VAE_CONFIG_FILE_NAME)?)?,
    })
}

fn bf16_weight_files(
    component: Flux2KleinComponent,
    shard_files: Vec<crate::artifact_validation::ValidatedWeightsFile>,
) -> Flux2KleinComponentWeightFiles {
    Flux2KleinComponentWeightFiles::new(
        Flux2KleinWeightLayout::DiffusersBf16,
        component,
        shard_files,
    )
}

fn read_document(
    document_files: &BTreeMap<String, ValidatedRequiredFile>,
    file_name: &str,
) -> Result<Vec<u8>, Flux2KleinArtifactError> {
    let document_file =
        document_files
            .get(file_name)
            .ok_or_else(|| Flux2KleinArtifactError::ArtifactFile {
                file_name: file_name.to_owned(),
            })?;
    read_required_document(document_file, file_name)
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
