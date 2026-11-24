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

pub mod image_engine_shell;

#[cfg(any(test, feature = "test-support"))]
pub mod image_engine_shell_test_support;

pub mod image_generation_engine;

#[cfg(feature = "direct-mlx")]
pub use image_generation_engine::{
    ImageGenerationEngine, ImageGenerationEngineStep, ImageGenerationUnavailableEngine,
};

pub use image_engine_shell::{ImageComponentLoad, ImageEngineComponents};

pub use image_generation_engine::ImageGenerationEngineLoadResult;
