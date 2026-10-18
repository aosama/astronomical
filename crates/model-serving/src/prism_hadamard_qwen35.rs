//! The PrismML folded ternary Qwen3.5 family.
//!
//! Discovery model type `prism_hadamard_qwen35`: dense Qwen3.5-derived hybrid
//! attention whose ternary weights are stored in a Hadamard-rotated basis.
//! This family owns the folded-pack contract end to end — the transform is
//! applied unconditionally by construction here, and the standard qwen3_5
//! family never branches on folded artifacts.
pub mod configuration;
pub mod hadamard;

pub mod artifacts;

#[cfg(feature = "direct-mlx")]
pub mod engine;

#[cfg(feature = "direct-mlx")]
pub mod model;
