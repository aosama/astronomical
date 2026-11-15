//! Cumulative persistent astro-ssd-prompt-cache observability counters and stats snapshot.
//!
//! These counters track how often the astro SSD astro SSD prompt cache restored a prompt
//! prefix (hit) versus fell back to cold prefill (miss), how many prompt tokens
//! were saved across all hits, and how often persisted visual embeddings were
//! loaded instead of recomputed. They are pure Rust so they can be tested without
//! the `direct-mlx` feature and without MLX hardware.

#[cfg(test)]
mod tests;

#[cfg(feature = "direct-mlx")]
use astronomical_ipc_protocol::WorkerEvent;

/// Cumulative persistent astro-ssd-prompt-cache hit/miss/token-saved counters.
///
/// The engine increments these on each `restore_astro_ssd_prompt_cache_prefix`
/// call. The worker reads them alongside the disk store's block counts to emit
/// `WorkerEvent::AstroSsdPromptCacheStats`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AstroSsdPromptCacheCounters {
    astro_ssd_prompt_cache_hits: u64,
    astro_ssd_prompt_cache_misses: u64,
    astro_ssd_prompt_cache_tokens_saved: u64,
    astro_ssd_prompt_cache_partial_tail_hits: u64,
    astro_ssd_prompt_cache_visual_embedding_hits: u64,
    astro_ssd_prompt_cache_visual_embedding_misses: u64,
    astro_ssd_prompt_cache_visual_embedding_rows_loaded: u64,
    /// Requests that adopted the retained live decoder state instead of
    /// restoring the prefix KV from the disk store (issue #1167).
    astro_ssd_prompt_cache_in_memory_prefix_adoption_hits: u64,
}

impl AstroSsdPromptCacheCounters {
    #[cfg(any(test, feature = "direct-mlx"))]
    /// Records one successful persistent astro-ssd-prompt-cache prefix restore.
    ///
    /// `restored_token_count` is the number of prompt tokens the astro SSD astro SSD prompt cache supplied,
    /// which is accumulated into `astro_ssd_prompt_cache_tokens_saved`.
    pub fn record_astro_ssd_prompt_cache_hit(&mut self, restored_token_count: usize) {
        self.astro_ssd_prompt_cache_hits = self.astro_ssd_prompt_cache_hits.saturating_add(1);
        self.astro_ssd_prompt_cache_tokens_saved = self
            .astro_ssd_prompt_cache_tokens_saved
            .saturating_add(u64::try_from(restored_token_count).unwrap_or(u64::MAX));
    }

    #[cfg(feature = "direct-mlx")]
    /// Records one persistent astro-ssd-prompt-cache miss (cold prefill fallback).
    /// Records one request that adopted the retained live state.
    pub fn record_in_memory_prefix_adoption_hit(&mut self) {
        self.astro_ssd_prompt_cache_in_memory_prefix_adoption_hits = self
            .astro_ssd_prompt_cache_in_memory_prefix_adoption_hits
            .saturating_add(1);
    }

    pub fn record_astro_ssd_prompt_cache_miss(&mut self) {
        self.astro_ssd_prompt_cache_misses = self.astro_ssd_prompt_cache_misses.saturating_add(1);
    }

    /// Records that one successful restore reused a partial tail block from a
    /// previous turn, so the request only prefilled the uncached suffix.
    pub fn record_partial_tail_hit(&mut self) {
        self.astro_ssd_prompt_cache_partial_tail_hits = self
            .astro_ssd_prompt_cache_partial_tail_hits
            .saturating_add(1);
    }

    /// Records one successful persistent visual-embedding file restore.
    pub fn record_astro_ssd_prompt_cache_visual_embedding_hit(
        &mut self,
        visual_embedding_row_count: usize,
    ) {
        self.astro_ssd_prompt_cache_visual_embedding_hits = self
            .astro_ssd_prompt_cache_visual_embedding_hits
            .saturating_add(1);
        self.astro_ssd_prompt_cache_visual_embedding_rows_loaded = self
            .astro_ssd_prompt_cache_visual_embedding_rows_loaded
            .saturating_add(u64::try_from(visual_embedding_row_count).unwrap_or(u64::MAX));
    }

    /// Records one persistent visual-embedding astro SSD astro SSD prompt cache miss.
    pub fn record_astro_ssd_prompt_cache_visual_embedding_miss(&mut self) {
        self.astro_ssd_prompt_cache_visual_embedding_misses = self
            .astro_ssd_prompt_cache_visual_embedding_misses
            .saturating_add(1);
    }

    /// Returns the cumulative number of successful persistent astro-ssd-prompt-cache restores.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_hits(&self) -> u64 {
        self.astro_ssd_prompt_cache_hits
    }

    /// Returns the cumulative number of persistent astro-ssd-prompt-cache misses.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_misses(&self) -> u64 {
        self.astro_ssd_prompt_cache_misses
    }

    /// Returns the cumulative number of prompt tokens restored from the astro SSD astro SSD prompt cache.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_tokens_saved(&self) -> u64 {
        self.astro_ssd_prompt_cache_tokens_saved
    }

    /// Returns how many restores reused a partial tail block from a previous turn.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_partial_tail_hits(&self) -> u64 {
        self.astro_ssd_prompt_cache_partial_tail_hits
    }

    /// Returns how many visual embedding files were loaded from the astro SSD astro SSD prompt cache.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_visual_embedding_hits(&self) -> u64 {
        self.astro_ssd_prompt_cache_visual_embedding_hits
    }

    /// Returns how many visual embedding files were absent, invalid, or had to be recomputed.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_visual_embedding_misses(&self) -> u64 {
        self.astro_ssd_prompt_cache_visual_embedding_misses
    }

    /// Returns how many visual embedding rows were loaded from SSD-backed files.
    #[must_use]
    pub const fn astro_ssd_prompt_cache_visual_embedding_rows_loaded(&self) -> u64 {
        self.astro_ssd_prompt_cache_visual_embedding_rows_loaded
    }

    /// Returns the hit rate as `hits / (hits + misses)`, rounded to 4 decimals.
    ///
    /// Returns `0.0` when no queries have occurred, avoiding division by zero.
    #[must_use]
    pub fn astro_ssd_prompt_cache_hit_rate(&self) -> f64 {
        let total_astro_ssd_prompt_cache_queries = self
            .astro_ssd_prompt_cache_hits
            .saturating_add(self.astro_ssd_prompt_cache_misses);
        if total_astro_ssd_prompt_cache_queries == 0 {
            return 0.0;
        }
        let hit_rate =
            self.astro_ssd_prompt_cache_hits as f64 / total_astro_ssd_prompt_cache_queries as f64;
        (hit_rate * 10_000.0).round() / 10_000.0
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "direct-mlx")]
/// Builds the IPC event carrying persistent astro-ssd-prompt-cache observability data.
///
/// Combines cumulative counters and active-model file counts with global-root
/// byte totals in the `WorkerEvent::AstroSsdPromptCacheStats` variant emitted
/// after `Ready` and after each `Completed`.
#[cfg(any(test, feature = "direct-mlx"))]
#[must_use]
pub fn build_astro_ssd_prompt_cache_stats_event(
    astro_ssd_prompt_cache_counters: &AstroSsdPromptCacheCounters,
    astro_ssd_prompt_cache_block_token_count: u64,
    astro_ssd_prompt_cache_sequence_state_block_count: u64,
    astro_ssd_prompt_cache_boundary_state_snapshot_count: u64,
    astro_ssd_prompt_cache_visual_embedding_count: u64,
    global_astro_ssd_prompt_cache_total_size_bytes: u64,
    global_astro_ssd_prompt_cache_visual_embedding_total_size_bytes: u64,
    global_astro_ssd_prompt_cache_maximum_size_bytes: u64,
) -> WorkerEvent {
    WorkerEvent::AstroSsdPromptCacheStats {
        astro_ssd_prompt_cache_hits: astro_ssd_prompt_cache_counters.astro_ssd_prompt_cache_hits(),
        astro_ssd_prompt_cache_misses: astro_ssd_prompt_cache_counters
            .astro_ssd_prompt_cache_misses(),
        astro_ssd_prompt_cache_tokens_saved: astro_ssd_prompt_cache_counters
            .astro_ssd_prompt_cache_tokens_saved(),
        astro_ssd_prompt_cache_partial_tail_hits: astro_ssd_prompt_cache_counters
            .astro_ssd_prompt_cache_partial_tail_hits(),
        astro_ssd_prompt_cache_block_token_count,
        astro_ssd_prompt_cache_sequence_state_block_count,
        astro_ssd_prompt_cache_boundary_state_snapshot_count,
        astro_ssd_prompt_cache_visual_embedding_count,
        astro_ssd_prompt_cache_total_size_bytes: global_astro_ssd_prompt_cache_total_size_bytes,
        astro_ssd_prompt_cache_visual_embedding_total_size_bytes:
            global_astro_ssd_prompt_cache_visual_embedding_total_size_bytes,
        astro_ssd_prompt_cache_maximum_size_bytes: global_astro_ssd_prompt_cache_maximum_size_bytes,
        astro_ssd_prompt_cache_visual_embedding_hits: astro_ssd_prompt_cache_counters
            .astro_ssd_prompt_cache_visual_embedding_hits(),
        astro_ssd_prompt_cache_visual_embedding_misses: astro_ssd_prompt_cache_counters
            .astro_ssd_prompt_cache_visual_embedding_misses(),
        astro_ssd_prompt_cache_visual_embedding_rows_loaded: astro_ssd_prompt_cache_counters
            .astro_ssd_prompt_cache_visual_embedding_rows_loaded(),
    }
}
