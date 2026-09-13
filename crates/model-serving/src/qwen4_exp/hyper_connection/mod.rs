//! Residual stream mixing for `qwen4_exp` hyper-connections.
//!
//! The algebra is the only owner here so far. Graph integration, compiled
//! routes, kernel probes, and state handoff arrive with the forward work and
//! must enact this algebra rather than re-derive it.

pub mod stream_algebra;

// The executor binds this algebra to MLX graph buffers, so it exists only when
// the crate is built against the runtime integration. The algebra above stays
// available for configuration-only builds.
#[cfg(feature = "direct-mlx")]
pub mod execution;

#[cfg(feature = "direct-mlx")]
pub use execution::HyperConnectionExecutor;
pub use stream_algebra::{
    GatedMixOutput, GatedResidualWeights, StreamAlgebraError, StreamMixingPlan, average_combine,
    average_mix, gated_combine, gated_mix, grouped_rms_norm,
};
