//! Strict contracts for the first Apache-2.0 FLUX.2 Klein image profile.
//!
//! This domain validates immutable artifact ownership and computes engine inputs.
//! It intentionally contains no MLX execution so the future engine cannot bypass
//! configuration, transport, schedule, or unified-memory admission contracts.

mod artifact;
mod artifact_text_validation;
mod configuration;
mod dimensions;
mod engine;
mod image_encoding;
mod inventory;
mod memory_admission;
mod official_profile;
mod scheduler;
#[cfg(feature = "direct-mlx")]
mod text_conditioning;
#[cfg(feature = "direct-mlx")]
mod transformer;
mod transformer_shape_profile;
mod vae;

pub use artifact::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, FLUX2_KLEIN_PROVIDER_MODEL_ID,
    Flux2KleinArtifactError, Flux2KleinArtifactProvenance, Flux2KleinArtifactValidator,
    Flux2KleinLicense, Flux2KleinRetainedArtifactFiles, ValidatedFlux2KleinArtifact,
};
pub use configuration::{
    Flux2KleinConfigError, Flux2KleinPipelineConfig, Flux2KleinSchedulerConfig,
    Flux2KleinTextEncoderConfig, Flux2KleinTransformerConfig, Flux2KleinVaeConfig,
};
pub use dimensions::{Flux2KleinDimensionError, Flux2KleinImageDimensions};
pub use engine::{Flux2KleinComponentLoad, Flux2KleinEngineComponents, Flux2KleinImageEngine};
#[cfg(feature = "direct-mlx")]
#[doc(hidden)]
pub use engine::{
    flux2_klein_allocator_cache_limit_for_tests, flux2_klein_euler_update_for_tests,
    flux2_klein_initial_latents_for_tests, flux2_klein_keyed_noise_and_euler_for_tests,
};
pub use image_encoding::{
    Flux2KleinImageEncodingError, Flux2KleinPngEncoder, flux2_klein_reference_rgb_u8,
};
pub use inventory::{Flux2KleinTensorDescriptor, Flux2KleinTensorInventory};
pub use memory_admission::{
    Flux2KleinMemoryAdmission, Flux2KleinMemoryAdmissionError, Flux2KleinMemoryGeometry,
    Flux2KleinResidencyMode, Flux2KleinResidencyPlan,
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
    Flux2KleinTransformerGeometryError, Flux2KleinTransformerInputs, Flux2KleinTransformerOutput,
    Flux2KleinTransformerWeights, apply_rope_for_component_oracle,
};
pub use vae::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
    Flux2KleinPackedLatentLayout, Flux2KleinVaeError, Flux2KleinVaeTile, Flux2KleinVaeTilePlan,
    Flux2KleinVaeTilingConfig, flux2_klein_inverse_batch_norm_reference,
};
#[cfg(feature = "direct-mlx")]
pub use vae::{Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder};
