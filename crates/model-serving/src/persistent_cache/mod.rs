//! Architecture-neutral, SSD-backed model-state reuse.
//!
//! Model packages provide a validated DecoderCacheLayout and own live-state
//! restoration. This package owns block identity, bounded safetensors validation,
//! prefix lookup, quota enforcement, and descriptor-backed tensor files for
//! decoder state and projected visual embeddings.

pub(crate) mod block_causal_input;
pub(crate) mod block_format;
pub(crate) mod block_format_error;
mod block_key;
#[cfg(feature = "direct-mlx")]
mod block_manifest;
#[cfg(any(test, feature = "direct-mlx"))]
pub(crate) mod counters;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store;
#[cfg(feature = "direct-mlx")]
mod disk_store_block_transaction;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_clear;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_error;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_file;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_global_quota;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_global_quota_candidate;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_global_quota_scan;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_index;
#[cfg(feature = "direct-mlx")]
mod disk_store_retention_reconciliation;
#[cfg(feature = "direct-mlx")]
pub(crate) mod disk_store_scan;
#[cfg(feature = "direct-mlx")]
mod disk_store_visual_embeddings;
#[cfg(feature = "direct-mlx")]
mod disk_store_write;
mod model_contract;
mod model_contract_error;
mod model_contract_storage_geometry;
mod partial_tail_probe;
pub(crate) mod persistent_safetensors_header;
mod prefill_boundary;
pub(crate) mod prefix_lookup;
#[cfg(feature = "direct-mlx")]
mod retention_policy;
#[cfg(feature = "direct-mlx")]
mod startup_cleanup_evidence;
pub(crate) mod visual_embedding_format;
pub(crate) mod visual_embedding_key;
mod visual_embedding_model_contract;

pub use block_key::{PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError};
#[cfg(feature = "direct-mlx")]
pub use counters::build_persistent_prompt_cache_stats_event;
#[cfg(feature = "direct-mlx")]
pub use disk_store::{PersistentPromptCacheDiskStore, PersistentPromptCacheDiskStoreConfig};
#[cfg(feature = "direct-mlx")]
pub use disk_store_clear::{
    PersistentPromptCacheClearOutcome, clear_persistent_prompt_cache_directory,
};
#[cfg(feature = "direct-mlx")]
pub use disk_store_error::PersistentPromptCacheDiskStoreError;
#[cfg(feature = "direct-mlx")]
pub use disk_store_write::PersistentPromptCachePublicationOutcome;
pub use model_contract::PersistentPromptCacheModelContract;
pub use model_contract_error::PersistentPromptCacheModelContractError;
pub use prefill_boundary::{
    persistent_prompt_cache_boundary_clamped_prefill_chunk_end,
    persistent_prompt_cache_boundary_completed_prefill_chunk_tokens,
    persistent_prompt_cache_partial_tail_capture_end,
};
pub use prefix_lookup::{
    PersistentPromptCacheLookupDiagnostics, PersistentPromptCacheMissReason,
    PersistentPromptCachePrefixLookupResult,
};
#[cfg(feature = "direct-mlx")]
pub use startup_cleanup_evidence::{
    PersistentPromptCacheStartupCleanupCategory, PersistentPromptCacheStartupCleanupEvidence,
};
pub use visual_embedding_format::PersistentVisualEmbeddingFileError;
pub use visual_embedding_key::PersistentVisualEmbeddingKey;
pub use visual_embedding_model_contract::PersistentVisualEmbeddingModelContract;

#[cfg(feature = "direct-mlx")]
pub use counters::PersistentPromptCacheCounters;

#[cfg(feature = "direct-mlx")]
pub use prefix_lookup::PersistentPromptCachePrefixLookup;

#[cfg(feature = "direct-mlx")]
pub use visual_embedding_format::PersistentVisualEmbeddingFileHeader;

#[cfg(feature = "direct-mlx")]
pub use visual_embedding_key::PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION;
