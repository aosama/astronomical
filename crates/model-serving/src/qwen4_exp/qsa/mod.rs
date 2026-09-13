//! Sparse-attention concerns for `qwen4_exp`.
//!
//! `selection.rs` owns the causal top-k key selection under the configured
//! budget; `sparse_attention.rs` owns attention over the gathered selected
//! keys. The index-key cache lives in the decoder layouts from #556, so the
//! cache and the selection evolve independently.

pub mod selection;
pub mod sparse_attention;

pub use selection::{Qwen4ExpSelectionPlan, select_keys};
pub use sparse_attention::sparse_attention;
