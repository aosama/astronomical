//! Laguna-owned expert page plans built from canonical tensor IDs.
//!
//! The pager reads retained source intervals and declared assembly recipes. It
//! does not parse raw tensor names, strip namespaces, or invent quantization.

mod error;
mod geometry;
mod layer_plan;
mod page_manifest;
mod source_slices;

#[cfg(feature = "direct-mlx")]
mod paged_execution;
#[cfg(feature = "direct-mlx")]
mod weight_page;

pub use error::LagunaPagingError;
pub use geometry::{LagunaRequestMemoryRequirements, laguna_sliding_prefill_transient_token_count};
pub use layer_plan::{LagunaExpertPagingPlan, LagunaSparseLayerPagingPlan};
#[cfg(feature = "direct-mlx")]
pub use paged_execution::forward_paged_routed_swiglu;
#[cfg(feature = "direct-mlx")]
pub use weight_page::{LagunaExpertWeightPage, load_laguna_expert_page};
