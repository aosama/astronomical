//! Qwen-specific expert plans, payload interpretation, and SSD streaming.
//!
//! Layer-plan construction reads SafeTensors headers only and is available
//! without Machine Learning framework for Apple silicon so mixed OptiQ storage
//! can be proven in unit tests. Pagers, pages, and resident astro expert caches still need
//! the graphics-processor runtime.

#[cfg(feature = "direct-mlx")]
mod astro_expert_cache;
#[cfg(feature = "direct-mlx")]
pub mod expert_pager;
#[cfg(feature = "direct-mlx")]
mod expert_pager_construction;
#[cfg(feature = "direct-mlx")]
mod expert_slot_storage_fill;
#[cfg(feature = "direct-mlx")]
mod expert_usage_history;
#[cfg(feature = "direct-mlx")]
mod paged_expert_weights;
// Pure planning, built and unit-tested without the graphics-processor
// runtime; only the direct-MLX slot filler calls it.
#[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
mod record_read_plan;
#[cfg(feature = "direct-mlx")]
mod retained_page_payload;
pub mod route_observation;

#[cfg(feature = "direct-mlx")]
pub(crate) use expert_usage_history::PersistedExpertUsage;
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_usage_history::{
    expert_usage_file_path, read_expert_usage, write_expert_usage,
};

#[cfg(feature = "direct-mlx")]
pub(crate) use astro_expert_cache::AstroExpertCache;
#[cfg(feature = "direct-mlx")]
pub(crate) use astro_expert_cache::RoutedExpertCoverage;
#[cfg(feature = "direct-mlx")]
pub use expert_pager::{ExpertPagingError, Qwen3_5ExpertPager};
#[cfg(feature = "direct-mlx")]
pub(crate) use expert_slot_storage_fill::ExpertSlotStorageFiller;
#[cfg(feature = "direct-mlx")]
pub(crate) use retained_page_payload::retained_expert_layer_page_payload_bytes;
