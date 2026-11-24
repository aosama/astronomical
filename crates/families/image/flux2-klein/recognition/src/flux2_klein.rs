//! Shallow discovery for FLUX.2 Klein 4B, in either reviewed build: the BF16
//! Diffusers pipeline (here) or the 4-bit MLX build (`flux2_klein_mlx_4bit`).
//!
//! For the BF16 build, geometry, the Apache-2.0 license, and the required
//! modular files prove the family. An immutable revision is recorded when
//! present; serving does not pin one Hub SHA.

use std::collections::BTreeSet;
use std::path::Path;

use model_discovery::ModelFamily;

use thiserror::Error;

use crate::flux2_klein_documents::{
    PipelineClass, PipelineIndex, TextEncoderGeometry, TextEncoderIndex, TransformerGeometry,
    VaeGeometry,
};
use crate::flux2_klein_mlx_4bit;
use model_discovery::bounded_artifact_file;
use model_discovery::{
    FlowMatchSchedulerProfile, ImageDirectoryEvidence, ImageGenerationCapabilities, ModelLicense,
    is_reviewed_flow_match_scheduler, is_safe_safetensors_path, nonempty_file_size,
    pipeline_provenance, safetensors_payload_size,
};

pub(super) const CANONICAL_MODEL_ID: &str = "FLUX.2-klein-4B";
pub(super) const PROVIDER_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
const MAXIMUM_JSON_BYTES: u64 = 4 * 1024 * 1024;
/// The scheduler settings the reviewed build ships with.
const REVIEWED_SCHEDULER_PROFILE: FlowMatchSchedulerProfile = FlowMatchSchedulerProfile {
    max_image_seq_len: 4_096,
    max_shift: 1.15,
    shift: 3.0,
    shift_terminal: None,
};
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
    #[error("FLUX.2 Klein {component} 4-bit weight index is missing, malformed, or not Klein 4B")]
    InvalidQuantizedWeightIndex { component: &'static str },
    #[error("FLUX.2 Klein provenance names a repository other than the 4-bit build")]
    UnsupportedProviderModelId,
}

pub fn classifies_pipeline_index(pipeline_index_bytes: &[u8]) -> Result<bool, serde_json::Error> {
    let pipeline_class: PipelineClass = serde_json::from_slice(pipeline_index_bytes)?;
    if pipeline_class.class_name.as_deref() != Some(PIPELINE_CLASS_NAME) {
        return Ok(false);
    }
    let pipeline_index: PipelineIndex = serde_json::from_slice(pipeline_index_bytes)?;
    Ok(is_exact_distilled_pipeline(&pipeline_index))
}

/// Rereads one exact directory without walking its parents, children, or configured scan roots.
/// A directory with a pipeline index is the BF16 build; any other is checked as the 4-bit build.
pub fn verify_model_directory(
    model_directory: &Path,
) -> Result<ImageDirectoryEvidence, Flux2KleinDirectoryVerificationError> {
    build_verifier(model_directory)(model_directory)
}

type BuildVerifier =
    fn(&Path) -> Result<ImageDirectoryEvidence, Flux2KleinDirectoryVerificationError>;

fn build_verifier(model_directory: &Path) -> BuildVerifier {
    if model_directory.join("model_index.json").is_file() {
        verify_model_directory_evidence
    } else {
        flux2_klein_mlx_4bit::verify_model_directory_evidence
    }
}

/// What both builds can do: the same pipeline, so the same reviewed envelope.
pub(super) fn flux2_klein_capabilities() -> ImageGenerationCapabilities {
    ImageGenerationCapabilities {
        supports_text_to_image: true,
        supports_image_editing: false,
        supports_multiple_reference_images: false,
        // FLUX.2 Klein is a distilled turbo model whose reference schedule is four steps.
        default_steps: 4,
        // The serving profile's reviewed envelope: 64-pixel minimum side, 1024-pixel
        // maximum, and the 16-pixel latent alignment.
        minimum_dimension_pixels: 64,
        maximum_dimension_pixels: 1_024,
        dimension_multiple_pixels: 16,
    }
}

/// Requires the authoritative modular package rather than its duplicate single-file export.
fn verify_model_directory_evidence(
    model_directory: &Path,
) -> Result<ImageDirectoryEvidence, Flux2KleinDirectoryVerificationError> {
    validate_pipeline_index(model_directory)?;
    validate_transformer_geometry(model_directory)?;
    validate_text_encoder_geometry(model_directory)?;
    validate_vae_geometry(model_directory)?;
    validate_scheduler_geometry(model_directory)?;
    validate_apache_2_license(model_directory)?;
    for required_sidecar in REQUIRED_SIDECARS {
        bounded_artifact_file::read_bounded_nonempty_file(
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
    let (provider_model_id, revision) = pipeline_provenance(model_directory, PROVIDER_MODEL_ID)
        .ok_or(Flux2KleinDirectoryVerificationError::MissingRevision)?;
    Ok(ImageDirectoryEvidence {
        canonical_model_id: CANONICAL_MODEL_ID.to_owned(),
        provider_model_id,
        revision,
        license: ModelLicense::Apache20,
        capabilities: flux2_klein_capabilities(),
        model_size_bytes,
    })
}

fn validate_apache_2_license(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let license_bytes = bounded_artifact_file::read_bounded_nonempty_file(
        &model_directory.join("LICENSE.md"),
        MAXIMUM_LICENSE_BYTES,
    )
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
    let pipeline_index: PipelineIndex = bounded_artifact_file::read_json(
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
    let geometry: TransformerGeometry = bounded_artifact_file::read_json(
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
    let geometry: TextEncoderGeometry = bounded_artifact_file::read_json(
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
    let geometry: VaeGeometry = bounded_artifact_file::read_json(
        &model_directory.join("vae/config.json"),
        MAXIMUM_JSON_BYTES,
    )
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
    is_reviewed_flow_match_scheduler(
        model_directory,
        MAXIMUM_JSON_BYTES,
        &REVIEWED_SCHEDULER_PROFILE,
    )
    .then_some(())
    .ok_or(Flux2KleinDirectoryVerificationError::InvalidSchedulerConfiguration)
}

fn measure_modular_weight_bytes(
    model_directory: &Path,
) -> Result<u64, Flux2KleinDirectoryVerificationError> {
    let text_encoder_index: TextEncoderIndex = bounded_artifact_file::read_json(
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
            .checked_add(nonempty_file_size(&indexed_weight_path).ok_or(
                Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile {
                    component: "text encoder",
                },
            )?)
            .ok_or(Flux2KleinDirectoryVerificationError::ModelSizeOverflow)?;
        text_encoder_payload_size_bytes = text_encoder_payload_size_bytes
            .checked_add(
                safetensors_payload_size(&indexed_weight_path)
                    .ok_or(Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex)?,
            )
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
            .checked_add(nonempty_file_size(&modular_weight_path).ok_or(
                Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile { component },
            )?)
            .ok_or(Flux2KleinDirectoryVerificationError::ModelSizeOverflow)?;
    }
    Ok(modular_weight_size_bytes)
}

/// The FLUX.2 Klein image family, as a Diffusers pipeline or the 4-bit build.
pub const FLUX2_KLEIN_MODEL_FAMILY: ModelFamily = ModelFamily::named("flux2_klein");
