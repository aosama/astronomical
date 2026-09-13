//! Sparse-attention concerns for `qwen4_exp`.
//!
//! `selection.rs` owns the causal top-k key selection under the configured
//! budget; `sparse_attention.rs` owns attention over the gathered selected
//! keys. The index-key cache lives in the decoder layouts from #556, so the
//! cache and the selection evolve independently.

// Both files bind MLX buffers, so the whole module compiles only against the
// runtime integration feature.
#[cfg(feature = "direct-mlx")]
pub mod selection;
#[cfg(feature = "direct-mlx")]
pub mod sparse_attention;

#[cfg(feature = "direct-mlx")]
pub use selection::{Qwen4ExpSelectionPlan, select_keys};
#[cfg(feature = "direct-mlx")]
pub use sparse_attention::sparse_attention;
