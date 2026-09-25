//! The Qwen-Image-2.1 serving engine: one `ImageGenerationEngine` over the render session.
//!
//! The engine is a thin phase machine: the MLX components own the runtime, construct the
//! pipeline per request (the render releases components in the reference's offload order, so
//! each image reloads weights), and drive the render session one boundary per
//! `advance_generation` call. The lifecycle maps those boundaries onto the IPC progress phases
//! so a worker stays responsive between denoising steps and can cancel between any two.
//!
//! `components.rs` holds the architecture-neutral seam (real MLX owner or a hermetic fake),
//! `request_validation.rs` pins the official request envelope, `lifecycle.rs` is the engine,
//! and `mlx_components.rs` is the production owner.

mod components;
mod lifecycle;
#[cfg(feature = "direct-mlx")]
mod mlx_components;
mod request_validation;

pub use components::{
    QwenImage21ComponentLoad, QwenImage21EngineComponents, QwenImage21EngineRequest,
};
pub use lifecycle::QwenImage21ImageEngine;
#[cfg(feature = "direct-mlx")]
pub use mlx_components::QwenImage21MlxComponents;
pub use request_validation::validate_official_request;
