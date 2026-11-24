//! The FLUX.2 Klein 4B image family, a crate of its own beside the
//! `image-generation` shell (issue #2473, Step 7), in its BF16 Diffusers build
//! and its 4-bit MLX build (`mlx_4bit_build`).
//!
//! Artifact validation, configuration, schedules and memory admission carry no
//! MLX dependency, so the engine cannot bypass their contracts; the MLX engine
//! (`engine`, `transformer`, `text_conditioning`, `vae`, `tensor_store`) is
//! built only with the `direct-mlx` feature.

pub mod artifact;
mod artifact_text_validation;
pub mod configuration;
pub mod dimensions;
pub mod engine;
pub mod image_encoding;
pub mod inventory;
pub mod memory_admission;
pub mod mlx_4bit_build;
#[cfg(test)]
mod performance_attribution_tests;
#[cfg(test)]
mod test_fixtures;
// Image-lane utilization compose is called from the MLX component owner.
// Without `direct-mlx` that owner is absent, so keep this module off the
// default lib graph instead of leaving three dead-code warnings (issue #576).
#[cfg(any(test, feature = "direct-mlx"))]
mod memory_utilization;
pub mod official_profile;
pub mod scheduler;
#[cfg(feature = "direct-mlx")]
pub mod tensor_store;
#[cfg(feature = "direct-mlx")]
pub mod text_conditioning;
#[cfg(feature = "direct-mlx")]
pub mod transformer;
pub mod transformer_shape_profile;
pub mod vae;
pub mod weight_layout;

#[cfg(feature = "direct-mlx")]
pub use artifact::Flux2KleinArtifactValidator;
pub use artifact::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, FLUX2_KLEIN_PROVIDER_MODEL_ID,
    Flux2KleinArtifactError, Flux2KleinArtifactProvenance, Flux2KleinLicense,
    Flux2KleinRetainedArtifactFiles, ValidatedFlux2KleinArtifact,
};
pub use configuration::{
    Flux2KleinConfigError, Flux2KleinPipelineConfig, Flux2KleinSchedulerConfig,
    Flux2KleinTextEncoderConfig, Flux2KleinTransformerConfig, Flux2KleinVaeConfig,
};
pub use dimensions::Flux2KleinImageDimensions;
pub use engine::Flux2KleinImageEngine;
#[cfg(feature = "direct-mlx")]
pub use engine::build_flux2_klein_image_engine;
pub use engine::components::{Flux2KleinComponentLoad, Flux2KleinEngineComponents};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use engine::{
    flux2_klein_euler_update_for_tests, flux2_klein_initial_latents_for_tests,
    flux2_klein_keyed_noise_and_euler_for_tests,
};
#[cfg(feature = "direct-mlx")]
pub use image_encoding::{
    Flux2KleinImageEncodingError, Flux2KleinPngEncoder, flux2_klein_reference_rgb_u8,
};
pub use inventory::{Flux2KleinTensorDescriptor, Flux2KleinTensorInventory};
#[cfg(any(test, feature = "direct-mlx"))]
pub use memory_admission::Flux2KleinMemoryGeometry;
#[cfg(feature = "direct-mlx")]
pub use memory_admission::{
    Flux2KleinMemoryAdmission, Flux2KleinResidencyMode, Flux2KleinResidencyPlan,
};
pub use mlx_4bit_build::{
    FLUX2_KLEIN_MLX_4BIT_MODEL_ID, FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID,
    FLUX2_KLEIN_MLX_4BIT_REVISION,
};
pub use official_profile::Flux2KleinOfficialProfile;
pub use scheduler::{
    Flux2KleinFlowSchedule, Flux2KleinFlowScheduler, Flux2KleinFlowSchedulerError,
    Flux2KleinFlowStep,
};
#[cfg(feature = "direct-mlx")]
pub use transformer::{
    Flux2KleinBlockGroupEvent, Flux2KleinBlockKind, Flux2KleinComponentOracle,
    Flux2KleinTransformer, Flux2KleinTransformerError, Flux2KleinTransformerGeometry,
    Flux2KleinTransformerInputs, Flux2KleinTransformerOutput, Flux2KleinTransformerWeights,
    apply_rope_for_component_oracle,
};
#[cfg(feature = "direct-mlx")]
pub use vae::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
    Flux2KleinPackedLatentLayout, Flux2KleinVaeError, Flux2KleinVaeTile,
    flux2_klein_inverse_batch_norm_reference,
};
#[cfg(feature = "direct-mlx")]
pub use vae::{Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder};
pub use weight_layout::{
    Flux2KleinComponent, Flux2KleinComponentWeightFiles, Flux2KleinWeightLayout,
};
