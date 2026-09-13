//! Decoder state for `qwen4_exp`.
//!
//! `cache_layout.rs` names and lays out the per-layer state streams;
//! `state_geometry.rs` measures them in bytes so admission projects real
//! totals. Live state allocation, checkpoints, and the attention math that
//! consumes these streams arrive with the forward work and must build on
//! these owners rather than re-deriving geometry.

pub mod cache_layout;
pub mod state_geometry;

pub use cache_layout::{Qwen4ExpDecoderLayerCacheDtypes, qwen4_exp_decoder_cache_layout};
pub use state_geometry::Qwen4ExpStateGeometry;
