//! Exact shallow discovery contract for the reviewed distilled FLUX.2 Klein 4B pipeline.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path};

use thiserror::Error;

use super::bounded_artifact_file::{read_bounded_nonempty_file, read_json};
use super::classified_artifacts::immutable_file_revision;
use super::flux2_klein_documents::{
    PipelineClass, PipelineIndex, SchedulerGeometry, TextEncoderGeometry, TextEncoderIndex,
    TransformerGeometry, VaeGeometry,
};
use super::{ImageGenerationCapabilities, ModelLicense};

pub(super) const CANONICAL_MODEL_ID: &str = "FLUX.2-klein-4B";
pub(super) const PROVIDER_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
const REVIEWED_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";
const MAXIMUM_JSON_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_TEXT_ENCODER_INDEX_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_SIDECAR_BYTES: u64 = 64 * 1024 * 1024;
const MAXIMUM_LICENSE_BYTES: u64 = 64 * 1024;
const PIPELINE_CLASS_NAME: &str = "Flux2KleinPipeline";
const REQUIRED_SIDECARS: [&str; 8] = [
    "text_encoder/generation_config.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
];

/// Trusted shallow evidence reread from one selected FLUX artifact directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinDirectoryEvidence {
    pub canonical_model_id: String,
    pub provider_model_id: String,
    pub revision: String,
    pub license: ModelLicense,
    pub capabilities: ImageGenerationCapabilities,
    pub model_size_bytes: u64,
}

/// Bounded path-free rejection for a directory that does not prove the reviewed profile.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum Flux2KleinDirectoryVerificationError {
    #[error("FLUX.2 Klein pipeline index is missing, malformed, oversized, or unsupported")]
    InvalidPipelineIndex,
    #[error("FLUX.2 Klein transformer configuration does not match the reviewed profile")]
    InvalidTransformerConfiguration,
    #[error("FLUX.2 Klein text encoder configuration does not match the reviewed profile")]
    InvalidTextEncoderConfiguration,
    #[error("FLUX.2 Klein VAE configuration does not match the reviewed profile")]
    InvalidVaeConfiguration,
    #[error("FLUX.2 Klein scheduler configuration does not match the reviewed profile")]
    InvalidSchedulerConfiguration,
    #[error("FLUX.2 Klein license evidence is missing or invalid")]
    InvalidLicense,
    #[error("FLUX.2 Klein required sidecar {sidecar} is missing, empty, or oversized")]
    MissingOrInvalidSidecar { sidecar: &'static str },
    #[error("FLUX.2 Klein text encoder weight index is invalid")]
    InvalidTextEncoderWeightIndex,
    #[error("FLUX.2 Klein {component} weight file is missing, empty, or invalid")]
    MissingOrInvalidWeightFile { component: &'static str },
    #[error("FLUX.2 Klein weight size exceeds the supported integer range")]
    ModelSizeOverflow,
    #[error("FLUX.2 Klein immutable revision evidence is missing")]
    MissingRevision,
    #[error("FLUX.2 Klein artifact revision does not match the reviewed profile")]
    UnexpectedRevision,
}

pub(super) fn classifies_pipeline_index(
    pipeline_index_bytes: &[u8],
) -> Result<bool, serde_json::Error> {
    let pipeline_class: PipelineClass = serde_json::from_slice(pipeline_index_bytes)?;
    if pipeline_class.class_name.as_deref() != Some(PIPELINE_CLASS_NAME) {
        return Ok(false);
    }
    let pipeline_index: PipelineIndex = serde_json::from_slice(pipeline_index_bytes)?;
    Ok(is_exact_distilled_pipeline(&pipeline_index))
}

/// Rereads one exact directory without walking its parents, children, or configured scan roots.
pub fn verify_model_directory(
    model_directory: &Path,
) -> Result<Flux2KleinDirectoryEvidence, Flux2KleinDirectoryVerificationError> {
    verify_model_directory_evidence(model_directory)
}

/// Requires the authoritative modular package rather than its duplicate single-file export.
fn verify_model_directory_evidence(
    model_directory: &Path,
) -> Result<Flux2KleinDirectoryEvidence, Flux2KleinDirectoryVerificationError> {
    validate_pipeline_index(model_directory)?;
    validate_transformer_geometry(model_directory)?;
    validate_text_encoder_geometry(model_directory)?;
    validate_vae_geometry(model_directory)?;
    validate_scheduler_geometry(model_directory)?;
    validate_apache_2_license(model_directory)?;
    for required_sidecar in REQUIRED_SIDECARS {
        read_bounded_nonempty_file(
            &model_directory.join(required_sidecar),
            MAXIMUM_SIDECAR_BYTES,
        )
        .map_err(
            |_| Flux2KleinDirectoryVerificationError::MissingOrInvalidSidecar {
                sidecar: required_sidecar,
            },
        )?;
    }
    let model_size_bytes = measure_modular_weight_bytes(model_directory)?;
    let revision = immutable_file_revision(model_directory, "model_index.json")
        .ok_or(Flux2KleinDirectoryVerificationError::MissingRevision)?;
    if revision != REVIEWED_REVISION {
        return Err(Flux2KleinDirectoryVerificationError::UnexpectedRevision);
    }
    Ok(Flux2KleinDirectoryEvidence {
        canonical_model_id: CANONICAL_MODEL_ID.to_owned(),
        provider_model_id: PROVIDER_MODEL_ID.to_owned(),
        revision,
        license: ModelLicense::Apache20,
        capabilities: ImageGenerationCapabilities {
            supports_text_to_image: true,
            supports_image_editing: false,
            supports_multiple_reference_images: false,
        },
        model_size_bytes,
    })
}

fn validate_apache_2_license(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let license_bytes =
        read_bounded_nonempty_file(&model_directory.join("LICENSE.md"), MAXIMUM_LICENSE_BYTES)
            .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidLicense)?;
    let license_text = std::str::from_utf8(&license_bytes)
        .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidLicense)?;
    (license_text.contains("Apache License")
        && license_text.contains("Version 2.0, January 2004")
        && license_text.contains("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION")
        && license_text.contains("END OF TERMS AND CONDITIONS"))
    .then_some(())
    .ok_or(Flux2KleinDirectoryVerificationError::InvalidLicense)
}

fn validate_pipeline_index(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let pipeline_index: PipelineIndex = read_json(
        &model_directory.join("model_index.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidPipelineIndex)?;
    is_exact_distilled_pipeline(&pipeline_index)
        .then_some(())
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidPipelineIndex)
}

fn is_exact_distilled_pipeline(pipeline_index: &PipelineIndex) -> bool {
    pipeline_index.class_name == PIPELINE_CLASS_NAME
        && pipeline_index.is_distilled
        && pipeline_index.scheduler == ["diffusers", "FlowMatchEulerDiscreteScheduler"]
        && pipeline_index.text_encoder == ["transformers", "Qwen3ForCausalLM"]
        && pipeline_index.tokenizer == ["transformers", "Qwen2TokenizerFast"]
        && pipeline_index.transformer == ["diffusers", "Flux2Transformer2DModel"]
        && pipeline_index.vae == ["diffusers", "AutoencoderKLFlux2"]
}

fn validate_transformer_geometry(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let geometry: TransformerGeometry = read_json(
        &model_directory.join("transformer/config.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidTransformerConfiguration)?;
    (geometry.class_name == "Flux2Transformer2DModel"
        && geometry.attention_head_dim == 128
        && geometry.axes_dims_rope == [32, 32, 32, 32]
        && geometry.eps == 0.000001
        && !geometry.guidance_embeds
        && geometry.in_channels == 128
        && geometry.joint_attention_dim == 7_680
        && geometry.mlp_ratio == 3.0
        && geometry.num_attention_heads == 24
        && geometry.num_layers == 5
        && geometry.num_single_layers == 20
        && geometry.out_channels.is_none()
        && geometry.patch_size == 1
        && geometry.rope_theta == 2_000.0
        && geometry.timestep_guidance_channels == 256)
        .then_some(())
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidTransformerConfiguration)
}

fn validate_text_encoder_geometry(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let geometry: TextEncoderGeometry = read_json(
        &model_directory.join("text_encoder/config.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidTextEncoderConfiguration)?;
    (geometry.architectures == ["Qwen3ForCausalLM"]
        && !geometry.attention_bias
        && geometry.attention_dropout == 0.0
        && geometry.dtype == "bfloat16"
        && geometry.head_dim == 128
        && geometry.hidden_act == "silu"
        && geometry.hidden_size == 2_560
        && geometry.intermediate_size == 9_728
        && geometry.layer_types.len() == 36
        && geometry
            .layer_types
            .iter()
            .all(|layer_type| layer_type == "full_attention")
        && geometry.max_position_embeddings == 40_960
        && geometry.max_window_layers == 36
        && geometry.model_type == "qwen3"
        && geometry.num_attention_heads == 32
        && geometry.num_hidden_layers == 36
        && geometry.num_key_value_heads == 8
        && geometry.rms_norm_eps == 0.000001
        && geometry.rope_scaling.is_none()
        && geometry.rope_theta == 1_000_000.0
        && geometry.sliding_window.is_none()
        && geometry.tie_word_embeddings
        && geometry.use_cache
        && !geometry.use_sliding_window
        && geometry.vocab_size == 151_936)
        .then_some(())
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidTextEncoderConfiguration)
}

fn validate_vae_geometry(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let geometry: VaeGeometry =
        read_json(&model_directory.join("vae/config.json"), MAXIMUM_JSON_BYTES)
            .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidVaeConfiguration)?;
    let four_down_blocks = ["DownEncoderBlock2D"; 4];
    let four_up_blocks = ["UpDecoderBlock2D"; 4];
    (geometry.class_name == "AutoencoderKLFlux2"
        && geometry.act_fn == "silu"
        && geometry.batch_norm_eps == 0.0001
        && geometry.batch_norm_momentum == 0.1
        && geometry.block_out_channels == [128, 256, 512, 512]
        && geometry.down_block_types == four_down_blocks
        && geometry.force_upcast
        && geometry.in_channels == 3
        && geometry.latent_channels == 32
        && geometry.layers_per_block == 2
        && geometry.mid_block_add_attention
        && geometry.norm_num_groups == 32
        && geometry.out_channels == 3
        && geometry.patch_size == [2, 2]
        && geometry.sample_size == 1_024
        && geometry.up_block_types == four_up_blocks
        && geometry.use_post_quant_conv
        && geometry.use_quant_conv)
        .then_some(())
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidVaeConfiguration)
}

fn validate_scheduler_geometry(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let geometry: SchedulerGeometry = read_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidSchedulerConfiguration)?;
    (geometry.class_name == "FlowMatchEulerDiscreteScheduler"
        && geometry.base_image_seq_len == 256
        && geometry.base_shift == 0.5
        && !geometry.invert_sigmas
        && geometry.max_image_seq_len == 4_096
        && geometry.max_shift == 1.15
        && geometry.num_train_timesteps == 1_000
        && geometry.shift == 3.0
        && geometry.shift_terminal.is_none()
        && !geometry.stochastic_sampling
        && geometry.time_shift_type == "exponential"
        && !geometry.use_beta_sigmas
        && geometry.use_dynamic_shifting
        && !geometry.use_exponential_sigmas
        && !geometry.use_karras_sigmas)
        .then_some(())
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidSchedulerConfiguration)
}

fn measure_modular_weight_bytes(
    model_directory: &Path,
) -> Result<u64, Flux2KleinDirectoryVerificationError> {
    let text_encoder_index: TextEncoderIndex = read_json(
        &model_directory.join("text_encoder/model.safetensors.index.json"),
        MAXIMUM_TEXT_ENCODER_INDEX_BYTES,
    )
    .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex)?;
    if text_encoder_index.metadata.total_size == 0 {
        return Err(Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex);
    }
    let mut indexed_shard_paths = BTreeSet::new();
    for shard_path in text_encoder_index.weight_map.values() {
        if !is_safe_safetensors_path(shard_path) {
            return Err(Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex);
        }
        indexed_shard_paths.insert(shard_path.as_str());
    }
    if indexed_shard_paths.is_empty() {
        return Err(Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex);
    }
    let mut text_encoder_weight_size_bytes = 0_u64;
    let mut text_encoder_payload_size_bytes = 0_u64;
    for shard_path in indexed_shard_paths {
        let indexed_weight_path = model_directory.join("text_encoder").join(shard_path);
        text_encoder_weight_size_bytes = text_encoder_weight_size_bytes
            .checked_add(required_weight_size(&indexed_weight_path, "text encoder")?)
            .ok_or(Flux2KleinDirectoryVerificationError::ModelSizeOverflow)?;
        text_encoder_payload_size_bytes = text_encoder_payload_size_bytes
            .checked_add(required_safetensors_payload_size(&indexed_weight_path)?)
            .ok_or(Flux2KleinDirectoryVerificationError::ModelSizeOverflow)?;
    }
    if text_encoder_index.metadata.total_size != text_encoder_payload_size_bytes {
        return Err(Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex);
    }
    let mut modular_weight_size_bytes = text_encoder_weight_size_bytes;
    for (component, modular_weight_path) in [
        (
            "transformer",
            model_directory.join("transformer/diffusion_pytorch_model.safetensors"),
        ),
        (
            "vae",
            model_directory.join("vae/diffusion_pytorch_model.safetensors"),
        ),
    ] {
        modular_weight_size_bytes = modular_weight_size_bytes
            .checked_add(required_weight_size(&modular_weight_path, component)?)
            .ok_or(Flux2KleinDirectoryVerificationError::ModelSizeOverflow)?;
    }
    Ok(modular_weight_size_bytes)
}

fn required_safetensors_payload_size(
    weight_path: &Path,
) -> Result<u64, Flux2KleinDirectoryVerificationError> {
    let weight_file_size_bytes = fs::metadata(weight_path)
        .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex)?
        .len();
    let mut weight_file = File::open(weight_path)
        .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex)?;
    let mut header_length_bytes = [0_u8; 8];
    weight_file
        .read_exact(&mut header_length_bytes)
        .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex)?;
    let header_size_bytes = u64::from_le_bytes(header_length_bytes);
    weight_file_size_bytes
        .checked_sub(8)
        .and_then(|remaining_bytes| remaining_bytes.checked_sub(header_size_bytes))
        .filter(|payload_size_bytes| *payload_size_bytes > 0)
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex)
}

fn required_weight_size(
    weight_path: &Path,
    component: &'static str,
) -> Result<u64, Flux2KleinDirectoryVerificationError> {
    let weight_metadata = fs::metadata(weight_path).map_err(|_| {
        Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile { component }
    })?;
    (weight_metadata.is_file() && weight_metadata.len() > 0)
        .then_some(weight_metadata.len())
        .ok_or(Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile { component })
}

fn is_safe_safetensors_path(shard_path: &str) -> bool {
    let shard_file_path = Path::new(shard_path);
    !shard_path.is_empty()
        && !shard_path.contains('\\')
        && !shard_file_path.is_absolute()
        && shard_file_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && shard_file_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
}
