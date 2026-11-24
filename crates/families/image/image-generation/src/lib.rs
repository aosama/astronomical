//! The image-generation shell (issues #2340, #2473): the worker-facing seam
//! every image family implements, and the engine shell that runs a family's
//! lifecycle (load once, one request at a time, publish only after cleanup).
//! Each family is a crate of its own beside this one, `flux2-klein` and
//! `qwen-image-21`, holding only what makes it different.
//!
//! The edge is one way: neither this crate nor a family crate depends on
//! `model-serving`, which `cargo xtask check-architecture` enforces; the
//! worker there drives the engines through the seam.
//!
//! The serving crate re-exports this crate's root, and each family crate's,
//! with a glob. A name two of these roots declare makes the globs ambiguous,
//! which the build rejects, so every name stays unique across the crates;
//! that is also why `qwen-image-21` keeps its modules private.

pub mod image_engine_shell;

pub mod image_family_construction;

#[cfg(any(test, feature = "test-support"))]
pub mod image_engine_shell_test_support;

pub mod image_generation_engine;

pub use image_generation_engine::{
    ImageGenerationEngine, ImageGenerationEngineStep, ImageGenerationUnavailableEngine,
};

pub use image_engine_shell::{ImageComponentLoad, ImageEngineComponents};

pub use image_family_construction::{
    BuildImageEngine, ImageArtifactRefusal, ImageFamilyConstruction, ImageProvenanceFromEvidence,
    VerifyImageDirectory, image_engine_from_evidence, verified_image_artifact,
};

pub use image_generation_engine::ImageGenerationEngineLoadResult;
