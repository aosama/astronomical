//! Cumulative persistent prompt-cache observability counters and stats snapshot.
//!
//! These counters track how often the persistent prompt cache restored a prompt
//! prefix (hit) versus fell back to cold prefill (miss), how many prompt tokens
//! were saved across all hits, and how often persisted visual embeddings were
//! loaded instead of recomputed. They are pure Rust so they can be tested without
//! the `direct-mlx` feature and without MLX hardware.

#[cfg(feature = "direct-mlx")]
use astronomical_ipc_protocol::WorkerEvent;

/// Cumulative persistent prompt-cache hit/miss/token-saved counters.
///
/// The engine increments these on each `restore_persistent_prompt_cache_prefix`
/// call. The worker reads them alongside the disk store's block counts to emit
/// `WorkerEvent::PersistentPromptCacheStats`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PersistentPromptCacheCounters {
    persistent_prompt_cache_hits: u64,
    persistent_prompt_cache_misses: u64,
    persistent_prompt_cache_tokens_saved: u64,
    persistent_prompt_cache_visual_embedding_hits: u64,
    persistent_prompt_cache_visual_embedding_misses: u64,
    persistent_prompt_cache_visual_embedding_rows_loaded: u64,
}

impl PersistentPromptCacheCounters {
    /// Records one successful persistent prompt-cache prefix restore.
    ///
    /// `restored_token_count` is the number of prompt tokens the cache supplied,
    /// which is accumulated into `persistent_prompt_cache_tokens_saved`.
    pub fn record_cache_hit(&mut self, restored_token_count: usize) {
        self.persistent_prompt_cache_hits = self.persistent_prompt_cache_hits.saturating_add(1);
        self.persistent_prompt_cache_tokens_saved = self
            .persistent_prompt_cache_tokens_saved
            .saturating_add(u64::try_from(restored_token_count).unwrap_or(u64::MAX));
    }

    /// Records one persistent prompt-cache miss (cold prefill fallback).
    pub fn record_cache_miss(&mut self) {
        self.persistent_prompt_cache_misses = self.persistent_prompt_cache_misses.saturating_add(1);
    }

    /// Records one successful persistent visual-embedding file restore.
    pub fn record_persistent_prompt_cache_visual_embedding_hit(
        &mut self,
        visual_embedding_row_count: usize,
    ) {
        self.persistent_prompt_cache_visual_embedding_hits = self
            .persistent_prompt_cache_visual_embedding_hits
            .saturating_add(1);
        self.persistent_prompt_cache_visual_embedding_rows_loaded = self
            .persistent_prompt_cache_visual_embedding_rows_loaded
            .saturating_add(u64::try_from(visual_embedding_row_count).unwrap_or(u64::MAX));
    }

    /// Records one persistent visual-embedding cache miss.
    pub fn record_persistent_prompt_cache_visual_embedding_miss(&mut self) {
        self.persistent_prompt_cache_visual_embedding_misses = self
            .persistent_prompt_cache_visual_embedding_misses
            .saturating_add(1);
    }

    /// Returns the cumulative number of successful persistent prompt-cache restores.
    #[must_use]
    pub const fn persistent_prompt_cache_hits(&self) -> u64 {
        self.persistent_prompt_cache_hits
    }

    /// Returns the cumulative number of persistent prompt-cache misses.
    #[must_use]
    pub const fn persistent_prompt_cache_misses(&self) -> u64 {
        self.persistent_prompt_cache_misses
    }

    /// Returns the cumulative number of prompt tokens restored from the persistent prompt cache.
    #[must_use]
    pub const fn persistent_prompt_cache_tokens_saved(&self) -> u64 {
        self.persistent_prompt_cache_tokens_saved
    }

    /// Returns how many visual embedding files were loaded from the persistent prompt cache.
    #[must_use]
    pub const fn persistent_prompt_cache_visual_embedding_hits(&self) -> u64 {
        self.persistent_prompt_cache_visual_embedding_hits
    }

    /// Returns how many visual embedding files were absent, invalid, or had to be recomputed.
    #[must_use]
    pub const fn persistent_prompt_cache_visual_embedding_misses(&self) -> u64 {
        self.persistent_prompt_cache_visual_embedding_misses
    }

    /// Returns how many visual embedding rows were loaded from SSD-backed files.
    #[must_use]
    pub const fn persistent_prompt_cache_visual_embedding_rows_loaded(&self) -> u64 {
        self.persistent_prompt_cache_visual_embedding_rows_loaded
    }

    /// Returns the hit rate as `hits / (hits + misses)`, rounded to 4 decimals.
    ///
    /// Returns `0.0` when no queries have occurred, avoiding division by zero.
    #[must_use]
    pub fn persistent_prompt_cache_hit_rate(&self) -> f64 {
        let total_persistent_prompt_cache_queries = self
            .persistent_prompt_cache_hits
            .saturating_add(self.persistent_prompt_cache_misses);
        if total_persistent_prompt_cache_queries == 0 {
            return 0.0;
        }
        let hit_rate =
            self.persistent_prompt_cache_hits as f64 / total_persistent_prompt_cache_queries as f64;
        (hit_rate * 10_000.0).round() / 10_000.0
    }
}

/// Builds the IPC event carrying persistent prompt-cache observability data.
///
/// Combines cumulative counters and active-model file counts with global-root
/// byte totals in the `WorkerEvent::PersistentPromptCacheStats` variant emitted
/// after `Ready` and after each `Completed`.
#[cfg(feature = "direct-mlx")]
#[must_use]
pub fn build_persistent_prompt_cache_stats_event(
    persistent_prompt_cache_counters: &PersistentPromptCacheCounters,
    persistent_prompt_cache_block_token_count: u64,
    persistent_prompt_cache_sequence_state_block_count: u64,
    persistent_prompt_cache_boundary_state_snapshot_count: u64,
    persistent_prompt_cache_visual_embedding_count: u64,
    global_prompt_cache_total_size_bytes: u64,
    global_prompt_cache_visual_embedding_total_size_bytes: u64,
    global_prompt_cache_maximum_size_bytes: u64,
) -> WorkerEvent {
    WorkerEvent::PersistentPromptCacheStats {
        persistent_prompt_cache_hits: persistent_prompt_cache_counters
            .persistent_prompt_cache_hits(),
        persistent_prompt_cache_misses: persistent_prompt_cache_counters
            .persistent_prompt_cache_misses(),
        persistent_prompt_cache_tokens_saved: persistent_prompt_cache_counters
            .persistent_prompt_cache_tokens_saved(),
        persistent_prompt_cache_block_token_count,
        persistent_prompt_cache_sequence_state_block_count,
        persistent_prompt_cache_boundary_state_snapshot_count,
        persistent_prompt_cache_visual_embedding_count,
        persistent_prompt_cache_total_size_bytes: global_prompt_cache_total_size_bytes,
        persistent_prompt_cache_visual_embedding_total_size_bytes:
            global_prompt_cache_visual_embedding_total_size_bytes,
        persistent_prompt_cache_maximum_size_bytes: global_prompt_cache_maximum_size_bytes,
        persistent_prompt_cache_visual_embedding_hits: persistent_prompt_cache_counters
            .persistent_prompt_cache_visual_embedding_hits(),
        persistent_prompt_cache_visual_embedding_misses: persistent_prompt_cache_counters
            .persistent_prompt_cache_visual_embedding_misses(),
        persistent_prompt_cache_visual_embedding_rows_loaded: persistent_prompt_cache_counters
            .persistent_prompt_cache_visual_embedding_rows_loaded(),
    }
}
