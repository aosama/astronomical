//! Concrete bounded image-engine lifecycle and its native MLX component owner.

mod components;
mod lifecycle;
mod request_validation;

#[cfg(feature = "direct-mlx")]
pub(super) use super::text_conditioning::{
    FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH, Flux2KleinTextConditioner,
    Flux2KleinTextConditioning, Flux2KleinTextConditioningAdvance, Flux2KleinTextConditioningState,
    Flux2KleinTokenizer,
};

#[cfg(feature = "direct-mlx")]
mod request_geometry;

#[cfg(feature = "direct-mlx")]
mod mlx_components;

pub use components::{Flux2KleinComponentLoad, Flux2KleinEngineComponents};
pub use lifecycle::Flux2KleinImageEngine;
#[cfg(feature = "direct-mlx")]
pub use mlx_components::{
    flux2_klein_allocator_cache_limit_for_tests, flux2_klein_euler_update_for_tests,
    flux2_klein_initial_latents_for_tests, flux2_klein_keyed_noise_and_euler_for_tests,
};
