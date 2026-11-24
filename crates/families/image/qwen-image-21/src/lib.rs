//! The Qwen-Image-2.1 image family, a crate of its own beside the
//! `image-generation` shell and the `flux2-klein` family (issue #2473, Step 7),
//! for the Qwen-Image-2.1 unified text-to-image + image-editing model. It
//! covers the whole path from an installed artifact to PNG pixels.
//!
//! The crate splits into three layers, each with its own files:
//!
//! - **Pure contracts** (`rope`, `mask`, `modulation`, `time`, `norm`, `scheduler`, `astro_image_prefix_key_value_cache`,
//!   `latent_layout`, `text_conditioning`): the index and formula logic that is easy to get wrong
//!   and cheap to verify — 3-axis RoPE, the block-causal mask, `causal_condition` modulation-row
//!   selection, sinusoidal timesteps, the flow-matching schedule, latent packing, and the chat
//!   template. They take no runtime dependency, so unit tests check them against the diffusers
//!   reference on the CPU.
//! - **Artifact layer** (`artifact`, `configuration`, `inventory`, `tensor_profiles`): validates the
//!   reviewed MLX package — strict component configs, exact physical tensor profiles for all three
//!   weight components, and index/physical size agreement — and hands each MLX component one owned
//!   file handle.
//! - **MLX execution** (`text_encoder`, `transformer`, `vae`, `pipeline`, `mlx_math`): the Qwen3-VL
//!   text encoder, the block-causal denoising transformer, the VAE decoder, and the pipeline that
//!   stages them. Everything here is behind `direct-mlx`; the components own their weights and
//!   release them in the reference's offload order.

#[cfg(test)]
pub mod oracle_test_support;

// `artifact`, `configuration`, `engine`, `scheduler`, `text_conditioning` and
// `vae` stay private: FLUX.2 Klein has public modules of the same names, and
// the serving crate re-exports both roots with a glob, where a name both
// declare is ambiguous. Everything callers need is re-exported below.
mod artifact;
mod astro_image_prefix_key_value_cache;
mod configuration;
mod inventory;
pub mod latent_layout;
mod mask;
mod modulation;
mod norm;
pub mod rope;
mod scheduler;
pub mod tensor_profiles;
mod text_conditioning;
#[cfg(any(test, feature = "direct-mlx"))]
pub mod time;
mod vae;

mod engine;
mod engine_error;
#[cfg(feature = "direct-mlx")]
mod mlx_math;
mod official_profile;
#[cfg(feature = "direct-mlx")]
mod pipeline;
pub mod render_contract;
#[cfg(feature = "direct-mlx")]
mod render_session;
#[cfg(feature = "direct-mlx")]
mod text_encoder;
#[cfg(feature = "direct-mlx")]
mod transformer;

pub use artifact::{
    QWEN_IMAGE_21_LICENSE_IDENTIFIER, QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
    QWEN_IMAGE_21_PROVIDER_MODEL_ID, QwenImage21ArtifactProvenance, QwenImage21License,
    QwenImage21RetainedArtifactFiles, ValidatedQwenImage21Artifact,
};
pub use astro_image_prefix_key_value_cache::{
    AstroImagePrefixKeyValueCacheMode, astro_image_prefix_key_value_cache_is_valid,
    astro_image_prefix_key_value_cache_query_slice, astro_image_prefix_key_value_cache_write_slice,
    prefix_length,
};
pub use configuration::{
    QwenImage21ConfigError, QwenImage21PipelineConfig, QwenImage21SchedulerConfig,
    quantized_group_count, quantized_row_count,
};
pub use inventory::{QwenImage21TensorDescriptor, QwenImage21TensorInventory};
pub use latent_layout::{
    QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH, QWEN_IMAGE_21_VAE_SCALE_FACTOR, VAE_SPATIAL_MULTIPLE,
    calculate_dimensions, latent_spatial_dimensions, pack_latents_seq_len,
    resolve_generation_dimensions, round_half_to_even, round_to_nearest_multiple,
    unpack_spatial_dims,
};
pub use mask::{build_block_causal_mask, build_image_ids, prefix_segments};
pub use modulation::{build_target_token_mask, causal_modulation_row_map};
pub use norm::zero_center_rms_norm;
pub use rope::frequencies_for_tests;
pub use scheduler::{
    FlowMatchSchedule, FlowMatchSchedulerParams, build_schedule, calculate_shift, default_shift,
    euler_step,
};
pub use tensor_profiles::{
    text_encoder_tensor_profiles, transformer_tensor_profiles, vae_tensor_profiles,
};
pub use text_conditioning::{
    build_prompt_conditioning, normalize_empty_prompt, render_t2i_prompt_template,
    render_ti2i_prompt_template, render_ti2i_prompt_template_with_image_count, special_tokens,
};
#[cfg(feature = "direct-mlx")]
pub use vae::QwenImage21VaeDecoder;
pub use vae::{
    QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT, QwenImage21VaeError,
    decoded_pixel_dimensions,
};

#[cfg(feature = "direct-mlx")]
pub use engine::QwenImage21MlxComponents;
#[cfg(feature = "direct-mlx")]
pub use engine::build_qwen_image_21_image_engine;
pub use engine::{QwenImage21ImageEngine, QwenImage21ImageFamily, validate_official_request};
pub use engine_error::QwenImage21EngineError;
pub use official_profile::{
    QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES, QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
    qwen_image_21_image_generation_capabilities, qwen_image_21_official_model_id,
};
#[cfg(feature = "direct-mlx")]
pub use pipeline::QwenImage21Pipeline;
pub use render_contract::QwenImage21RenderRequest;
#[cfg(feature = "direct-mlx")]
pub use text_encoder::QwenImage21TextEncoder;
#[cfg(feature = "direct-mlx")]
pub use transformer::{QwenImage21Transformer, QwenImage21TransformerRequest};

#[cfg(feature = "direct-mlx")]
pub use text_conditioning::QWEN_IMAGE_21_SYS_PROMPT;

// What the image-generation crate re-exported for this family before it
// became a crate of its own (issue #2473, Step 7).
pub use artifact::QwenImage21ArtifactValidator;
pub use artifact::error::QwenImage21ArtifactError;
pub use configuration::text_encoder::QwenImage21TextEncoderConfig;
pub use configuration::transformer::QwenImage21TransformerConfig;
pub use configuration::vae::QwenImage21VaeConfig;
pub use engine::components::{
    QwenImage21ComponentLoad, QwenImage21EngineComponents, QwenImage21EngineRequest,
};
pub use render_contract::{QwenImage21RenderAdvance, QwenImage21Rendered};
pub use tensor_profiles::QwenImage21TensorProfile;
