//! GPU building blocks shared by more than one model family (issue #2340).
//!
//! Beside the kernels and layers (row-batched matmul, sparse experts, token
//! sampling, attention math, quantized and add-on linear layers) it holds the
//! small vocabulary they and the engine record with: performance attribution,
//! the routed-expert values, the serving phase and the sampling seed. Those
//! are not GPU code; they live here because the blocks report through them,
//! and putting them higher up would make the blocks depend on the engine.
//!
//! Every model family in `astronomical-model-serving` builds on these pieces,
//! and none of them knows about a family, so the dependency runs one way: this
//! crate never depends on the serving crate, which `cargo xtask
//! check-architecture` enforces. A change here rebuilds and retests this crate
//! on its own; the families then pick it up.
//!
//! Unsafe code is denied workspace-wide. The one exception lives in
//! `performance_attribution::macos_process_io`, whose macOS accounting syscall
//! is inseparable from the evidence types it produces (rationale lives there).

#[cfg(feature = "direct-mlx")]
pub mod affine_quantized_linear;

pub mod affine_quantization_scheme;

pub mod attention;

#[cfg(feature = "direct-mlx")]
pub mod gpu_token_sampling;

#[cfg(feature = "direct-mlx")]
pub mod linear_adapters;

#[cfg(all(test, feature = "direct-mlx"))]
mod mlx_test_runtime;

pub mod performance_attribution;

pub mod route_observation;

pub mod row_batched_matmul;

pub mod sampling_seed;

pub mod serving_phase;

pub mod sparse_experts;
