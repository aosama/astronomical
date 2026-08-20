//! Laguna-owned model construction and execution.
//!
//! Layers are built only from `LagunaLayerDescriptor`. Sparse Mixture-of-Experts
//! feed-forward is owned by `laguna/moe`.

mod cache_layout;
mod error;
#[cfg(feature = "direct-mlx")]
mod expert_coverage;
#[cfg(feature = "direct-mlx")]
mod expert_release;

#[cfg(feature = "direct-mlx")]
mod attention;
#[cfg(feature = "direct-mlx")]
mod bound_linear;
#[cfg(feature = "direct-mlx")]
pub(in crate::laguna) use bound_linear::LagunaBoundLinear;
#[cfg(feature = "direct-mlx")]
mod decoder_layer;
#[cfg(feature = "direct-mlx")]
mod decoder_state;
#[cfg(feature = "direct-mlx")]
mod expert_residency;
#[cfg(feature = "direct-mlx")]
pub(in crate::laguna) use expert_residency::LagunaLastExpertForward;
#[cfg(feature = "direct-mlx")]
mod dense_feed_forward;
#[cfg(feature = "direct-mlx")]
mod memory_policy;
#[cfg(feature = "direct-mlx")]
mod model;
#[cfg(feature = "direct-mlx")]
mod rope_application;
#[cfg(feature = "direct-mlx")]
mod router_correction_bias;
#[cfg(feature = "direct-mlx")]
mod weights;

pub use cache_layout::laguna_decoder_cache_layout;
pub use error::LagunaExecutionError;

#[cfg(feature = "direct-mlx")]
pub use decoder_state::LagunaDecoderState;
#[cfg(feature = "direct-mlx")]
pub(in crate::laguna) use decoder_state::LagunaDecoderStateAllocationCheckpoint;
#[cfg(feature = "direct-mlx")]
pub use model::LagunaModel;
#[cfg(feature = "direct-mlx")]
pub use weights::LagunaNativeWeights;
