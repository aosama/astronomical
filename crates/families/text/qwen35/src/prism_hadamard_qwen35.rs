//! The PrismML folded ternary pack format.
//!
//! Discovery model type `prism_hadamard_qwen35`: dense Qwen3.5-derived hybrid
//! attention whose ternary weights are stored in a Hadamard-rotated basis.
//! This module reads and checks the pack: its configuration, its Hadamard
//! sign table and its files. The shared Qwen 3.5 core serves the pack
//! (`qwen3_5_core::model_math::folded_ternary_weights`); the separate folded
//! engine that served it before #2595 is retired.
pub mod configuration;
pub mod hadamard;

pub mod artifacts;
