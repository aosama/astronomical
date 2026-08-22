//! Architecture-neutral, SSD-backed model-state reuse.
//!
//! Model packages provide a validated DecoderCacheLayout and own live-state
//! restoration. This package owns block identity, bounded safetensors validation,
//! prefix lookup, quota enforcement, and descriptor-backed tensor files for
//! decoder state and projected visual embeddings.

mod block_causal_input;
mod block_format;
mod block_format_error;
mod block_key;
#[cfg(feature = "direct-mlx")]
mod block_manifest;
mod counters;
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
mod disk_store_speculative_prefill_policy_purge;
#[cfg(feature = "direct-mlx")]
mod disk_store_speculative_prefill_selection;
#[cfg(feature = "direct-mlx")]
mod disk_store_speculative_prefill_target_state;
#[cfg(feature = "direct-mlx")]
mod disk_store_visual_embeddings;
#[cfg(feature = "direct-mlx")]
mod disk_store_write;
mod model_contract;
mod model_contract_error;
mod model_contract_storage_geometry;
pub(crate) mod persistent_safetensors_header;
mod prefill_boundary;
mod prefix_lookup;
#[cfg(feature = "direct-mlx")]
mod retention_policy;
mod speculative_prefill_policy;
mod speculative_prefill_selection;
mod speculative_prefill_selection_metadata;
mod speculative_prefill_target_state;
#[cfg(feature = "direct-mlx")]
mod startup_cleanup_evidence;
mod visual_embedding_format;
mod visual_embedding_key;
mod visual_embedding_model_contract;

pub use block_causal_input::PersistentPromptCacheBlockCausalInput;
pub use block_format::PersistentPromptCacheBlockHeader;
pub use block_format_error::PersistentPromptCacheBlockError;
pub use block_key::{PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError};
pub use counters::PersistentPromptCacheCounters;
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
pub use disk_store_speculative_prefill_policy_purge::PersistentSpeculativePrefillPolicyPurgeOutcome;
#[cfg(feature = "direct-mlx")]
pub use disk_store_write::PersistentPromptCachePublicationOutcome;
pub use model_contract::PersistentPromptCacheModelContract;
pub use model_contract_error::PersistentPromptCacheModelContractError;
pub use prefill_boundary::{
    persistent_prompt_cache_boundary_clamped_prefill_chunck_end,
    persistent_prompt_cache_boundary_completed_prefill_chunck_tokens,
};
pub use prefix_lookup::{
    PersistentPromptCacheLookupDiagnostics, PersistentPromptCacheMissReason,
    PersistentPromptCachePrefixLookup, PersistentPromptCachePrefixLookupResult,
};
pub use speculative_prefill_policy::PersistentSpeculativePrefillPolicyIdentity;
pub use speculative_prefill_selection::{
    PERSISTENT_SPECULATIVE_PREFILL_SELECTION_FORMAT_VERSION,
    PersistentSpeculativePrefillSelectionContract,
};
#[cfg(feature = "direct-mlx")]
pub use speculative_prefill_target_state::RestoredSpeculativePrefillTargetState;
pub use speculative_prefill_target_state::{
    PERSISTENT_SPECULATIVE_PREFILL_TARGET_STATE_FORMAT_VERSION,
    PersistentSpeculativePrefillTargetStateContract,
    longest_reusable_speculative_prefill_target_prefix,
};
#[cfg(feature = "direct-mlx")]
pub use startup_cleanup_evidence::{
    PersistentPromptCacheStartupCleanupCategory, PersistentPromptCacheStartupCleanupEvidence,
};
pub use visual_embedding_format::{
    PersistentVisualEmbeddingFileError, PersistentVisualEmbeddingFileHeader,
};
pub use visual_embedding_key::{
    PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION, PersistentVisualEmbeddingKey,
};
pub use visual_embedding_model_contract::PersistentVisualEmbeddingModelContract;
