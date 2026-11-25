//! A step of an MLX graph that runs on the host, between two GPU steps.
//!
//! A model streamed from the SSD has to look at what the GPU computed before
//! it can say what the GPU reads next: a layer's router picks experts, and
//! only then is it known which rows of the `astro_expert_cache` the layer
//! gathers and which experts must first be read from storage. Looking from
//! the caller ends the evaluation at every such point: the graph is cut,
//! submitted, waited for and read back, and the next piece is built from the
//! answer, once a layer and so about thirty-five times a reply token.
//!
//! A host step keeps the look inside the graph. [`append_host_step`] adds a
//! node that MLX's evaluator runs on its CPU stream after the GPU steps it
//! reads and before the GPU steps that read it, so the whole token is one
//! graph and one evaluation. `cargo measure-graph-host-step` prices both.
//!
//! The step itself is native code (`native/graph_host_step.cpp`), an MLX
//! primitive compiled against the patched MLX sources; `host_step` is its
//! safe owner. Everything here needs MLX, so the crate is empty without the
//! `direct-mlx` feature.

// The only module that needs unsafe code: it hands the native step a pointer
// to the caller's state and receives it back on MLX's CPU stream thread.
// Every block carries its SAFETY argument.
#[cfg(feature = "direct-mlx")]
#[allow(unsafe_code)]
mod host_step;

#[cfg(feature = "direct-mlx")]
pub use host_step::{HostStep, HostStepFailed, HostStepInputs, HostStepOutput, append_host_step};
