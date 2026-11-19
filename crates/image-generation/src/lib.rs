//! Image generation for Astronomical (issue #2340): the worker-facing seam
//! every image family implements, and the FLUX.2 klein and Qwen-Image-2.1
//! families.
//!
//! The families sit beside the other model families rather than inside the
//! serving crate, so an image change rebuilds and retests only this crate.
//! The edge is one way: this crate never depends on
//! `model-serving`, which `cargo xtask check-architecture`
//! enforces; the worker there drives these engines through the seam.
//!
//! The serving crate re-exports this crate's root with a glob, and a name it
//! declares itself would silently win over one from here, so every name at
//! this root stays unique across the two crates.

pub mod flux2_klein;

pub mod image_generation_engine;

pub mod qwen_image_21;

pub use flux2_klein::{
    FLUX2_KLEIN_MLX_4BIT_MODEL_ID, FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID,
    FLUX2_KLEIN_MLX_4BIT_REVISION, FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION,
    FLUX2_KLEIN_PROVIDER_MODEL_ID, Flux2KleinArtifactProvenance, Flux2KleinComponent,
    Flux2KleinComponentWeightFiles, Flux2KleinFlowSchedulerError, Flux2KleinImageDimensions,
    Flux2KleinImageEngine, Flux2KleinLicense, Flux2KleinRetainedArtifactFiles,
    Flux2KleinTensorDescriptor, Flux2KleinTensorInventory, Flux2KleinWeightLayout,
    ValidatedFlux2KleinArtifact,
};

#[cfg(feature = "direct-mlx")]
pub use flux2_klein::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
    Flux2KleinBlockKind, Flux2KleinComponentOracle, Flux2KleinImageEncodingError,
    Flux2KleinResidencyPlan, Flux2KleinTransformerError, Flux2KleinTransformerOutput,
    Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder, Flux2KleinVaeError, Flux2KleinVaeTile,
    apply_rope_for_component_oracle, flux2_klein_euler_update_for_tests,
    flux2_klein_initial_latents_for_tests, flux2_klein_inverse_batch_norm_reference,
    flux2_klein_keyed_noise_and_euler_for_tests, flux2_klein_reference_rgb_u8,
};

pub use image_generation_engine::{
    ImageGenerationEngine, ImageGenerationEngineStep, ImageGenerationUnavailableEngine,
};

pub use flux2_klein::artifact::{Flux2KleinArtifactError, Flux2KleinArtifactValidator};

pub use flux2_klein::configuration::Flux2KleinTransformerConfig;

pub use flux2_klein::engine::components::{Flux2KleinComponentLoad, Flux2KleinEngineComponents};

pub use flux2_klein::scheduler::{
    Flux2KleinFlowSchedule, Flux2KleinFlowScheduler, Flux2KleinFlowStep,
};

#[cfg(feature = "direct-mlx")]
pub use flux2_klein::transformer::execution::Flux2KleinTransformer;

#[cfg(feature = "direct-mlx")]
pub use flux2_klein::transformer::{
    Flux2KleinBlockGroupEvent, Flux2KleinTransformerGeometry, Flux2KleinTransformerGeometryError,
    Flux2KleinTransformerInputs, Flux2KleinTransformerWeights,
};

pub use image_generation_engine::ImageGenerationEngineLoadResult;

pub use qwen_image_21::artifact::QwenImage21ArtifactValidator;

pub use qwen_image_21::artifact::error::QwenImage21ArtifactError;

pub use qwen_image_21::configuration::text_encoder::QwenImage21TextEncoderConfig;

pub use qwen_image_21::configuration::transformer::QwenImage21TransformerConfig;

pub use qwen_image_21::configuration::vae::QwenImage21VaeConfig;

pub use qwen_image_21::engine::components::{
    QwenImage21ComponentLoad, QwenImage21EngineComponents, QwenImage21EngineRequest,
};

pub use qwen_image_21::render_contract::{QwenImage21RenderAdvance, QwenImage21Rendered};

pub use qwen_image_21::tensor_profiles::QwenImage21TensorProfile;

pub use qwen_image_21::{
    AstroImagePrefixKeyValueCacheMode, FlowMatchSchedule, QWEN_IMAGE_21_LATENT_CHANNEL_COUNT,
    QWEN_IMAGE_21_LICENSE_IDENTIFIER, QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
    QWEN_IMAGE_21_OUTPUT_CHANNEL_COUNT, QWEN_IMAGE_21_PROVIDER_MODEL_ID,
    QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH, QWEN_IMAGE_21_VAE_SCALE_FACTOR,
    QwenImage21ArtifactProvenance, QwenImage21ConfigError, QwenImage21License,
    QwenImage21PipelineConfig, QwenImage21RetainedArtifactFiles, QwenImage21SchedulerConfig,
    QwenImage21TensorDescriptor, QwenImage21TensorInventory, QwenImage21VaeError,
    ValidatedQwenImage21Artifact, astro_image_prefix_key_value_cache_is_valid,
    astro_image_prefix_key_value_cache_query_slice, astro_image_prefix_key_value_cache_write_slice,
    build_block_causal_mask, build_image_ids, build_prompt_conditioning, build_schedule,
    build_target_token_mask, calculate_dimensions, calculate_shift, causal_modulation_row_map,
    decoded_pixel_dimensions, default_shift, euler_step, frequencies_for_tests,
    latent_spatial_dimensions, normalize_empty_prompt, pack_latents_seq_len, prefix_length,
    prefix_segments, quantized_group_count, quantized_row_count, render_t2i_prompt_template,
    render_ti2i_prompt_template, render_ti2i_prompt_template_with_image_count,
    resolve_generation_dimensions, round_half_to_even, round_to_nearest_multiple, special_tokens,
    text_encoder_tensor_profiles, transformer_tensor_profiles, unpack_spatial_dims,
    vae_tensor_profiles, zero_center_rms_norm,
};

pub use qwen_image_21::{
    QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES, QWEN_IMAGE_21_GUIDANCE_THOUSANDTHS,
    QwenImage21EngineError, QwenImage21ImageEngine, QwenImage21RenderRequest,
    qwen_image_21_image_generation_capabilities, qwen_image_21_official_model_id,
    validate_official_request,
};

#[cfg(feature = "direct-mlx")]
pub use qwen_image_21::{
    QwenImage21MlxComponents, QwenImage21Pipeline, QwenImage21TextEncoder, QwenImage21Transformer,
    QwenImage21TransformerRequest, QwenImage21VaeDecoder,
};
