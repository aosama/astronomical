//! Astronomical's implementation of the quantization crate's sensitivity
//! probe for the Qwen3.5 family, in its own crate (issue #2340) so the
//! family and the worker never compile the quantizer.
//!
//! The quantization crate (the OptiQ port) defines what a probe model must
//! do: list its quantizable modules, forward a calibration sample, swap one
//! module's weight for a simulate-quantized copy and restore it, and
//! compute KL divergence between two logits tensors. This crate does that
//! with the Qwen 3.5 crate's held model and the MLX bindings, so the sweep
//! measures the exact graph Astronomical serves rather than an mlx-lm
//! re-implementation of the architecture. That is the point of running the
//! sweep inside Astronomical: the allocation is calibrated on the kernels
//! the artifact will actually run through.

mod expert_swaps;
mod kl_evaluation_engine;
mod layer_input_reuse_plan;
mod module_locations;
pub mod probe_model;
mod qwen3_5_probe;
mod simulated_quantization;

// The crates below the probe model, imported at the root so its
// `crate::...` paths into them read as they did inside the Qwen 3.5 crate,
// where the model came from.
use engine_core::*;
use model_primitives::*;

pub use kl_evaluation_engine::Qwen3_5KlDivergenceEngine;
pub use module_locations::{
    AttentionProjection, FeedForwardProjection, ModuleLocation, module_location_for,
};
pub use qwen3_5_probe::{Qwen3_5SensitivityProbe, load_qwen3_5_probe_model};
