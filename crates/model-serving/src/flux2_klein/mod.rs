//! Strict contracts for the first Apache-2.0 FLUX.2 Klein image profile.
//!
//! This domain validates immutable artifact ownership and computes engine inputs.
//! It intentionally contains no MLX execution so the future engine cannot bypass
//! configuration, transport, schedule, or unified-memory admission contracts.

pub(crate) mod artifact;
mod artifact_text_validation;
pub(crate) mod configuration;
pub(crate) mod dimensions;
pub(crate) mod engine;
pub(crate) mod image_encoding;
pub(crate) mod inventory;
pub(crate) mod memory_admission;
#[cfg(test)]
mod test_fixtures;
// Image-lane utilization compose is called from the MLX component owner.
// Without `direct-mlx` that owner is absent, so keep this module off the
// default lib graph instead of leaving three dead-code warnings (issue #576).
#[cfg(any(test, feature = "direct-mlx"))]
mod memory_utilization;
pub(crate) mod official_profile;
pub(crate) mod scheduler;
#[cfg(feature = "direct-mlx")]
pub(crate) mod text_conditioning;
#[cfg(feature = "direct-mlx")]
pub(crate) mod transformer;
pub(crate) mod transformer_shape_profile;
pub(crate) mod vae;

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
pub use official_profile::Flux2KleinOfficialProfile;
#[cfg(feature = "direct-mlx")]
pub use scheduler::Flux2KleinFlowScheduler;
pub use scheduler::{Flux2KleinFlowSchedule, Flux2KleinFlowSchedulerError, Flux2KleinFlowStep};
#[cfg(feature = "direct-mlx")]
pub use transformer::{
    Flux2KleinBlockKind, Flux2KleinComponentOracle, Flux2KleinTransformer,
    Flux2KleinTransformerError, Flux2KleinTransformerGeometry, Flux2KleinTransformerInputs,
    Flux2KleinTransformerOutput, apply_rope_for_component_oracle,
};
#[cfg(feature = "direct-mlx")]
pub use vae::{
    FLUX2_KLEIN_PACKED_LATENT_CHANNEL_COUNT, FLUX2_KLEIN_VAE_LATENT_CHANNEL_COUNT,
    Flux2KleinPackedLatentLayout, Flux2KleinVaeError, Flux2KleinVaeTile,
    flux2_klein_inverse_batch_norm_reference,
};
#[cfg(feature = "direct-mlx")]
pub use vae::{Flux2KleinVaeDecodeMode, Flux2KleinVaeDecoder};
