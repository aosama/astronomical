//! Architecture-neutral, SSD-backed model-state reuse.
//!
//! Model packages provide a validated AstroDecoderStateCacheLayout and own live-state
//! restoration. This package owns block identity, bounded safetensors validation,
//! prefix lookup, quota enforcement, and descriptor-backed tensor files for
//! decoder state and projected visual embeddings.

pub mod block_causal_input;
pub mod block_format;
pub mod block_format_error;
mod block_key;
#[cfg(feature = "direct-mlx")]
mod block_manifest;
// Only the direct-MLX engines capture astro SSD prompt cache blocks; the pure planning
// units still compile under test so their unit tests run in every build.
#[cfg(any(test, feature = "direct-mlx"))]
pub mod capture_planning;
#[cfg(test)]
mod capture_planning_tests;
#[cfg(any(test, feature = "direct-mlx"))]
pub mod counters;
#[cfg(feature = "direct-mlx")]
pub mod disk_store;
#[cfg(feature = "direct-mlx")]
mod disk_store_block_transaction;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_clear;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_error;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_file;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_global_quota;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_global_quota_candidate;
#[cfg(feature = "direct-mlx")]
mod disk_store_global_quota_demand;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_global_quota_scan;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_index;
#[cfg(feature = "direct-mlx")]
mod disk_store_missing_boundary_transaction;
#[cfg(feature = "direct-mlx")]
mod disk_store_retention_reconciliation;
#[cfg(feature = "direct-mlx")]
pub mod disk_store_scan;
#[cfg(feature = "direct-mlx")]
mod disk_store_visual_embeddings;
#[cfg(feature = "direct-mlx")]
mod disk_store_write;
#[cfg(test)]
mod frozen_ornith_test_support;
mod model_contract;
mod model_contract_error;
mod model_contract_storage_geometry;
mod partial_tail_probe;
pub mod persistent_safetensors_header;
mod prefill_boundary;
pub mod prefix_lookup;
pub mod prefix_lookup_steps;
#[cfg(feature = "direct-mlx")]
pub mod quota_bounded_contract;
#[cfg(feature = "direct-mlx")]
pub mod restore_support;
#[cfg(all(test, feature = "direct-mlx"))]
mod restore_support_tests;
#[cfg(all(test, feature = "direct-mlx"))]
mod restore_support_tests_support;
#[cfg(feature = "direct-mlx")]
mod retention_policy;
#[cfg(feature = "direct-mlx")]
mod startup_cleanup_evidence;
// Pure planning like `capture_planning`, so its tests run in every build.
#[cfg(any(test, feature = "direct-mlx"))]
pub mod tail_capture_planning;
pub mod visual_embedding_format;
pub mod visual_embedding_key;
mod visual_embedding_model_contract;

pub use block_key::{AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheBlockKeyError};
#[cfg(feature = "direct-mlx")]
pub use counters::build_astro_ssd_prompt_cache_stats_event;
#[cfg(feature = "direct-mlx")]
pub use disk_store::{AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheDiskStoreConfig};
#[cfg(feature = "direct-mlx")]
pub use disk_store_clear::{
    AstroSsdPromptCacheClearOutcome, clear_astro_ssd_prompt_cache_directory,
};
#[cfg(feature = "direct-mlx")]
pub use disk_store_error::AstroSsdPromptCacheDiskStoreError;
#[cfg(feature = "direct-mlx")]
pub use disk_store_write::AstroSsdPromptCachePublicationOutcome;
pub use model_contract::AstroSsdPromptCacheModelContract;
pub use model_contract_error::AstroSsdPromptCacheModelContractError;
// Only the direct-MLX engines consume the span type and resolver; a
// default-features build has no consumer, and an ungated re-export there
// is dead code the compiler must reject.
#[cfg(feature = "direct-mlx")]
pub use prefill_boundary::{ResolvedPrefillChunkSpan, resolve_prompt_prefill_chunk_span};
pub use prefill_boundary::{
    astro_ssd_prompt_cache_boundary_clamped_prefill_chunk_end,
    astro_ssd_prompt_cache_boundary_completed_prefill_chunk_tokens,
    astro_ssd_prompt_cache_clamped_prefill_chunk_end,
    astro_ssd_prompt_cache_partial_tail_capture_end,
};
pub use prefix_lookup::{
    AstroSsdPromptCacheLookupDiagnostics, AstroSsdPromptCacheMissReason,
    AstroSsdPromptCachePrefixLookupResult,
};
#[cfg(feature = "direct-mlx")]
pub use startup_cleanup_evidence::{
    AstroSsdPromptCacheStartupCleanupCategory, AstroSsdPromptCacheStartupCleanupEvidence,
};
pub use visual_embedding_format::PersistentVisualEmbeddingFileError;
pub use visual_embedding_key::PersistentVisualEmbeddingKey;
pub use visual_embedding_model_contract::PersistentVisualEmbeddingModelContract;

#[cfg(feature = "direct-mlx")]
pub use counters::AstroSsdPromptCacheCounters;

#[cfg(feature = "direct-mlx")]
pub use visual_embedding_format::PersistentVisualEmbeddingFileHeader;

#[cfg(feature = "direct-mlx")]
pub use visual_embedding_key::PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION;
