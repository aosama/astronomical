//! Qwen-specific expert plans, payload interpretation, and SSD streaming.
//!
//! Layer-plan construction reads SafeTensors headers only and is available
//! without Machine Learning framework for Apple silicon so mixed OptiQ storage
//! can be proven hermetically. Pagers, pages, and resident caches still need
//! the graphics-processor runtime.

#[cfg(feature = "direct-mlx")]
pub mod expert_pager;
#[cfg(feature = "direct-mlx")]
mod expert_pager_construction;
#[cfg(feature = "direct-mlx")]
mod paged_expert_weights;
pub mod quantized_expert_layer_plan;
#[cfg(feature = "direct-mlx")]
mod retained_expert_cache;

#[cfg(feature = "direct-mlx")]
pub use expert_pager::{ExpertPagingError, Qwen3_5ExpertPager};
#[cfg(feature = "direct-mlx")]
pub(crate) use retained_expert_cache::RetainedExpertCache;
