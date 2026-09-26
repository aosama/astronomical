//! Discovery for the reviewed Qwen-Image-2.1 MLX diffusion pipeline.
//!
//! The pipeline contains three independently owned components. Discovery proves their public
//! identities, geometry, safetensors inventory, and license provenance without claiming that an
//! engine can execute them.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path};

use thiserror::Error;

use super::bounded_artifact_file::{read_bounded_nonempty_file, read_json};
use super::classified_artifacts::{immutable_file_revision, immutable_model_provenance};
use super::qwen_image_21_documents::{
    ComponentSafetensorsIndex, PipelineClass, PipelineIndex, QwenImage21SchedulerGeometry,
    QwenImage21TextEncoderGeometry, QwenImage21TransformerGeometry, QwenImage21VaeGeometry,
};
use super::{ImageGenerationCapabilities, ModelLicense};

const CANONICAL_MODEL_ID: &str = "Qwen-Image-2.1-MLX-4bit";
const PROVIDER_MODEL_ID: &str = "mlx-community/Qwen-Image-2.1-MLX-4bit";
const PIPELINE_CLASS_NAME: &str = "QwenImage21Pipeline";
const MAXIMUM_JSON_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_COMPONENT_INDEX_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_SIDECAR_BYTES: u64 = 64 * 1024 * 1024;
const MAXIMUM_README_BYTES: u64 = 256 * 1024;
const REQUIRED_PROCESSOR_FILES: [&str; 9] = [
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
];
const REVIEWED_COMPONENTS: [(&str, &str, &str); 3] = [
    (
        "text_encoder",
        "Qwen3VLForConditionalGeneration",
        "text encoder",
    ),
    (
        "transformer",
        "QwenImage21Transformer2DModel",
        "transformer",
    ),
    ("vae", "AutoencoderKLQwenImage21", "VAE"),
];

/// Shallow proof for one complete Qwen-Image-2.1 MLX pipeline directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenImage21DirectoryEvidence {
    pub canonical_model_id: String,
    pub provider_model_id: String,
    pub revision: String,
    pub license: ModelLicense,
    pub capabilities: ImageGenerationCapabilities,
    pub model_size_bytes: u64,
}

/// Bounded, path-free rejection for a package that does not prove the reviewed T2I profile.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum QwenImage21DirectoryVerificationError {
    #[error("Qwen-Image-2.1 pipeline index is missing, malformed, oversized, or unsupported")]
    InvalidPipelineIndex,
    #[error("Qwen-Image-2.1 transformer configuration does not match the reviewed profile")]
    InvalidTransformerConfiguration,
    #[error("Qwen-Image-2.1 text encoder configuration does not match the reviewed profile")]
    InvalidTextEncoderConfiguration,
    #[error("Qwen-Image-2.1 VAE configuration does not match the reviewed profile")]
    InvalidVaeConfiguration,
    #[error("Qwen-Image-2.1 scheduler configuration does not match the reviewed profile")]
    InvalidSchedulerConfiguration,
    #[error("Qwen-Image-2.1 license provenance is missing or invalid")]
    InvalidLicenseProvenance,
    #[error("Qwen-Image-2.1 processor file {processor_file} is missing, empty, or oversized")]
    MissingOrInvalidProcessorFile { processor_file: &'static str },
    #[error("Qwen-Image-2.1 {component} safetensors index is invalid")]
    InvalidComponentWeightIndex { component: &'static str },
    #[error("Qwen-Image-2.1 {component} weight file is missing, empty, or invalid")]
    MissingOrInvalidWeightFile { component: &'static str },
    #[error("Qwen-Image-2.1 weight size exceeds the supported integer range")]
    ModelSizeOverflow,
    #[error("Qwen-Image-2.1 immutable revision evidence is missing")]
    MissingRevision,
}

/// Reports whether a Diffusers pipeline index names the reviewed Qwen-Image-2.1 package.
pub(super) fn classifies_pipeline_index(
    pipeline_index_bytes: &[u8],
) -> Result<bool, serde_json::Error> {
    let pipeline_class: PipelineClass = serde_json::from_slice(pipeline_index_bytes)?;
    if pipeline_class.class_name.as_deref() != Some(PIPELINE_CLASS_NAME) {
        return Ok(false);
    }
    let pipeline_index: PipelineIndex = serde_json::from_slice(pipeline_index_bytes)?;
    Ok(is_reviewed_pipeline(&pipeline_index))
}

/// Verifies one selected pipeline package without treating nested component configs as models.
pub fn verify_model_directory(
    model_directory: &Path,
) -> Result<QwenImage21DirectoryEvidence, QwenImage21DirectoryVerificationError> {
    validate_pipeline_index(model_directory)?;
    validate_transformer_geometry(model_directory)?;
    validate_text_encoder_geometry(model_directory)?;
    validate_vae_geometry(model_directory)?;
    validate_scheduler_geometry(model_directory)?;
    validate_processor_files(model_directory)?;
    validate_license_provenance(model_directory)?;
    let model_size_bytes = measure_reviewed_weight_bytes(model_directory)?;
    let library_provenance = immutable_model_provenance(model_directory);
    let revision = library_provenance
        .as_ref()
        .map(|(_, recorded_revision)| recorded_revision.clone())
        .or_else(|| immutable_file_revision(model_directory, "model_index.json"))
        .ok_or(QwenImage21DirectoryVerificationError::MissingRevision)?;

    Ok(QwenImage21DirectoryEvidence {
        canonical_model_id: CANONICAL_MODEL_ID.to_owned(),
        provider_model_id: library_provenance
            .map(|(recorded_provider_id, _)| recorded_provider_id)
            .unwrap_or_else(|| PROVIDER_MODEL_ID.to_owned()),
        revision,
        license: ModelLicense::QwenResearch,
        capabilities: ImageGenerationCapabilities {
            supports_text_to_image: true,
            supports_image_editing: false,
            supports_multiple_reference_images: false,
            // The reference pipeline's own default; lower counts quarter-denoise the render.
            default_steps: 40,
        },
        model_size_bytes,
    })
}

fn is_reviewed_pipeline(pipeline_index: &PipelineIndex) -> bool {
    pipeline_index.class_name == PIPELINE_CLASS_NAME
        && pipeline_index.processor == ["transformers", "Qwen3VLProcessor"]
        && pipeline_index.scheduler == ["diffusers", "FlowMatchEulerDiscreteScheduler"]
        && pipeline_index.text_encoder == ["transformers", "Qwen3VLForConditionalGeneration"]
        && pipeline_index.transformer == ["diffusers", "QwenImage21Transformer2DModel"]
        && pipeline_index.vae == ["diffusers", "AutoencoderKLQwenImage21"]
}

fn validate_pipeline_index(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    let pipeline_index: PipelineIndex = read_json(
        &model_directory.join("model_index.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| QwenImage21DirectoryVerificationError::InvalidPipelineIndex)?;
    is_reviewed_pipeline(&pipeline_index)
        .then_some(())
        .ok_or(QwenImage21DirectoryVerificationError::InvalidPipelineIndex)
}

fn validate_transformer_geometry(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    let geometry: QwenImage21TransformerGeometry = read_json(
        &model_directory.join("transformer/config.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| QwenImage21DirectoryVerificationError::InvalidTransformerConfiguration)?;
    (geometry.class_name == "QwenImage21Transformer2DModel"
        && geometry.attention_head_dim == 128
        && geometry.axes_dims_rope == [16, 56, 56]
        && geometry.context_in_dim == 4096
        && geometry.in_channels == 64
        && geometry.num_attention_heads == 32
        && geometry.num_layers == 32
        && geometry.out_channels == 64
        && geometry.patch_size == 1
        && geometry.mlp_ratio == 3
        && geometry.eps == 0.000001
        && geometry.causal_condition
        && geometry.quantization.bits == 4
        && geometry.quantization.group_size == 64
        && geometry.quantization.mode == "affine"
        && geometry.mlx_format)
        .then_some(())
        .ok_or(QwenImage21DirectoryVerificationError::InvalidTransformerConfiguration)
}

fn validate_text_encoder_geometry(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    let geometry: QwenImage21TextEncoderGeometry = read_json(
        &model_directory.join("text_encoder/config.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| QwenImage21DirectoryVerificationError::InvalidTextEncoderConfiguration)?;
    let text = &geometry.text_config;
    (geometry.architectures == ["Qwen3VLForConditionalGeneration"]
        && geometry.dtype == "bfloat16"
        && geometry.model_type == "qwen3_vl"
        && geometry.tie_word_embeddings == false
        && geometry.quantization.bits == 4
        && geometry.quantization.group_size == 64
        && geometry.quantization.mode == "affine"
        && geometry.mlx_format
        && text.dtype == "bfloat16"
        && text.attention_bias == false
        && text.attention_dropout == 0.0
        && text.head_dim == 128
        && text.hidden_act == "silu"
        && text.hidden_size == 4096
        && text.intermediate_size == 12288
        && text.max_position_embeddings == 262144
        && text.model_type == "qwen3_vl_text"
        && text.num_attention_heads == 32
        && text.num_hidden_layers == 36
        && text.num_key_value_heads == 8
        && text.rms_norm_eps == 0.000001
        && text.rope_scaling.mrope_interleaved
        && text.rope_scaling.mrope_section == [24, 20, 20]
        && text.rope_scaling.rope_type == "default"
        && text.rope_theta == 5_000_000
        && text.use_cache
        && text.vocab_size == 151_936)
        .then_some(())
        .ok_or(QwenImage21DirectoryVerificationError::InvalidTextEncoderConfiguration)
}

fn validate_vae_geometry(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    let geometry: QwenImage21VaeGeometry =
        read_json(&model_directory.join("vae/config.json"), MAXIMUM_JSON_BYTES)
            .map_err(|_| QwenImage21DirectoryVerificationError::InvalidVaeConfiguration)?;
    (geometry.class_name == "AutoencoderKLQwenImage21"
        && geometry.attn_scales.is_empty()
        && geometry.base_dim == 96
        && geometry.decoder_base_dim == 144
        && geometry.dim_mult == [1, 2, 4, 8, 8]
        && geometry.dropout == 0.0
        && geometry.in_channels == 4
        && geometry.is_residual
        && geometry.latents_mean.len() == geometry.z_dim as usize
        && geometry.latents_std.len() == geometry.z_dim as usize
        && geometry
            .latents_std
            .iter()
            .all(|std_value| *std_value > 0.0)
        && geometry.num_res_blocks == 2
        && geometry.out_channels == 4
        && geometry.patch_size.is_none()
        && geometry.scale_factor_spatial == 16
        && geometry.scale_factor_temporal == 8
        && geometry.temporal_downsample == [false, true, true, true]
        && geometry.z_dim == 64
        && geometry.mlx_format)
        .then_some(())
        .ok_or(QwenImage21DirectoryVerificationError::InvalidVaeConfiguration)
}

fn validate_scheduler_geometry(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    let geometry: QwenImage21SchedulerGeometry = read_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        MAXIMUM_JSON_BYTES,
    )
    .map_err(|_| QwenImage21DirectoryVerificationError::InvalidSchedulerConfiguration)?;
    (geometry.class_name == "FlowMatchEulerDiscreteScheduler"
        && geometry.base_image_seq_len == 256
        && geometry.base_shift == 0.5
        && !geometry.invert_sigmas
        && geometry.max_image_seq_len == 8192
        && geometry.max_shift == 0.9
        && geometry.num_train_timesteps == 1000
        && geometry.shift == 1.0
        && geometry.shift_terminal == Some(0.02)
        && !geometry.stochastic_sampling
        && geometry.time_shift_type == "exponential"
        && !geometry.use_beta_sigmas
        && geometry.use_dynamic_shifting
        && !geometry.use_exponential_sigmas
        && !geometry.use_karras_sigmas)
        .then_some(())
        .ok_or(QwenImage21DirectoryVerificationError::InvalidSchedulerConfiguration)
}

fn validate_processor_files(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    for processor_file in REQUIRED_PROCESSOR_FILES {
        read_bounded_nonempty_file(&model_directory.join(processor_file), MAXIMUM_SIDECAR_BYTES)
            .map_err(
                |_| QwenImage21DirectoryVerificationError::MissingOrInvalidProcessorFile {
                    processor_file,
                },
            )?;
    }
    Ok(())
}

fn validate_license_provenance(
    model_directory: &Path,
) -> Result<(), QwenImage21DirectoryVerificationError> {
    let readme_bytes =
        read_bounded_nonempty_file(&model_directory.join("README.md"), MAXIMUM_README_BYTES)
            .map_err(|_| QwenImage21DirectoryVerificationError::InvalidLicenseProvenance)?;
    let readme = std::str::from_utf8(&readme_bytes)
        .map_err(|_| QwenImage21DirectoryVerificationError::InvalidLicenseProvenance)?;
    (has_front_matter_value(readme, "license", "other")
        && has_front_matter_value(readme, "license_name", "qwen-research")
        && has_front_matter_value(readme, "base_model", "Qwen/Qwen-Image-2.1"))
    .then_some(())
    .ok_or(QwenImage21DirectoryVerificationError::InvalidLicenseProvenance)
}

fn has_front_matter_value(readme: &str, key: &str, expected_value: &str) -> bool {
    let expected_line = format!("{key}: {expected_value}");
    let mut lines = readme.lines().map(str::trim);
    if lines.next() != Some("---") {
        return false;
    }
    for line in lines {
        if line == "---" {
            break;
        }
        if line == expected_line {
            return true;
        }
    }
    false
}

fn measure_reviewed_weight_bytes(
    model_directory: &Path,
) -> Result<u64, QwenImage21DirectoryVerificationError> {
    let mut total_weight_size_bytes = 0_u64;
    for (component_directory, _, component_error_name) in REVIEWED_COMPONENTS {
        total_weight_size_bytes = total_weight_size_bytes
            .checked_add(measure_component_weight_bytes(
                model_directory,
                component_directory,
                component_error_name,
            )?)
            .ok_or(QwenImage21DirectoryVerificationError::ModelSizeOverflow)?;
    }
    Ok(total_weight_size_bytes)
}

fn measure_component_weight_bytes(
    model_directory: &Path,
    component_directory: &str,
    component_error_name: &'static str,
) -> Result<u64, QwenImage21DirectoryVerificationError> {
    let component_path = model_directory.join(component_directory);
    let index_path = component_path.join("model.safetensors.index.json");
    let index: ComponentSafetensorsIndex = read_json(&index_path, MAXIMUM_COMPONENT_INDEX_BYTES)
        .map_err(
            |_| QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                component: component_error_name,
            },
        )?;
    if index.metadata.total_size == 0 {
        return Err(
            QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                component: component_error_name,
            },
        );
    }
    let mut shard_paths = BTreeSet::new();
    for shard_path in index.weight_map.values() {
        if !is_safe_safetensors_path(shard_path) {
            return Err(
                QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                    component: component_error_name,
                },
            );
        }
        shard_paths.insert(shard_path.as_str());
    }
    if shard_paths.is_empty() {
        return Err(
            QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                component: component_error_name,
            },
        );
    }

    let mut component_file_size_bytes = 0_u64;
    let mut component_payload_size_bytes = 0_u64;
    for shard_path in shard_paths {
        let shard_file_path = component_path.join(shard_path);
        let shard_size_bytes = required_weight_size(&shard_file_path, component_error_name)?;
        let payload_size_bytes =
            required_safetensors_payload_size(&shard_file_path).map_err(|_| {
                QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                    component: component_error_name,
                }
            })?;
        component_file_size_bytes = component_file_size_bytes
            .checked_add(shard_size_bytes)
            .ok_or(QwenImage21DirectoryVerificationError::ModelSizeOverflow)?;
        component_payload_size_bytes = component_payload_size_bytes
            .checked_add(payload_size_bytes)
            .ok_or(QwenImage21DirectoryVerificationError::ModelSizeOverflow)?;
    }
    if component_payload_size_bytes != index.metadata.total_size {
        return Err(
            QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                component: component_error_name,
            },
        );
    }
    Ok(component_file_size_bytes)
}

fn required_weight_size(
    weight_path: &Path,
    component: &'static str,
) -> Result<u64, QwenImage21DirectoryVerificationError> {
    let weight_metadata = fs::metadata(weight_path).map_err(|_| {
        QwenImage21DirectoryVerificationError::MissingOrInvalidWeightFile { component }
    })?;
    (weight_metadata.is_file() && weight_metadata.len() > 0)
        .then_some(weight_metadata.len())
        .ok_or(QwenImage21DirectoryVerificationError::MissingOrInvalidWeightFile { component })
}

fn required_safetensors_payload_size(weight_path: &Path) -> Result<u64, ()> {
    let weight_file_size_bytes = fs::metadata(weight_path).map_err(|_| ())?.len();
    let mut weight_file = File::open(weight_path).map_err(|_| ())?;
    let mut header_size_bytes = [0_u8; 8];
    weight_file
        .read_exact(&mut header_size_bytes)
        .map_err(|_| ())?;
    let header_size_bytes = u64::from_le_bytes(header_size_bytes);
    weight_file_size_bytes
        .checked_sub(8)
        .and_then(|remaining| remaining.checked_sub(header_size_bytes))
        .filter(|payload_size_bytes| *payload_size_bytes > 0)
        .ok_or(())
}

fn is_safe_safetensors_path(shard_path: &str) -> bool {
    let shard_file_path = Path::new(shard_path);
    !shard_path.is_empty()
        && !shard_path.contains('\\')
        && !shard_file_path.is_absolute()
        && shard_file_path
            .components()
            .all(|path_component| matches!(path_component, Component::Normal(_)))
        && shard_file_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
}
