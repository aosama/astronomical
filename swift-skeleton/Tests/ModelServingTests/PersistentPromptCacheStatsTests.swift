import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the persistent prompt-cache observability counters:
/// miss and hit accounting, accumulated tokens saved, the four-decimal hit
/// rate, visual-embedding hit and row accounting, and partial-tail hits
/// counted separately from full-block hits.
final class PersistentPromptCacheStatsTests {

    @Test
    func should_increment_misses_on_a_cache_miss() {
        var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        counters.recordCacheMiss();
        #expect(counters.persistentPromptCacheMisses == 1);
        #expect(counters.persistentPromptCacheHits == 0);
        #expect(counters.persistentPromptCacheTokensSaved == 0);
    }

    @Test
    func should_accumulate_tokens_saved_across_multiple_hits() {
        var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        counters.recordCacheHit(restoredTokenCount: 2_048);
        counters.recordCacheHit(restoredTokenCount: 4_096);
        #expect(counters.persistentPromptCacheHits == 2);
        #expect(counters.persistentPromptCacheTokensSaved == 6_144);
    }

    @Test
    func should_return_a_hit_rate_of_zero_when_no_cache_queries_have_occurred() {
        let counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        #expect(counters.persistentPromptCacheHitRate() == 0.0);
    }

    @Test
    func should_compute_hit_rate_as_hits_over_total_queries() {
        var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        counters.recordCacheHit(restoredTokenCount: 2_048);
        counters.recordCacheHit(restoredTokenCount: 4_096);
        counters.recordCacheMiss();
        // 2 hits / 3 total = 0.6667 rounded to 4 decimals.
        #expect(counters.persistentPromptCacheHitRate() == 0.6667);
    }

    @Test
    func should_increment_visual_embedding_hits_and_rows_loaded_on_a_visual_embedding_cache_hit() {
        var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        counters.recordVisualEmbeddingHit(visualEmbeddingRowCount: 64);
        counters.recordVisualEmbeddingHit(visualEmbeddingRowCount: 128);

        #expect(counters.persistentPromptCacheVisualEmbeddingHits == 2);
        #expect(counters.persistentPromptCacheVisualEmbeddingRowsLoaded == 192);
        #expect(counters.persistentPromptCacheVisualEmbeddingMisses == 0);
    }

    @Test
    func should_increment_visual_embedding_misses_without_rows_loaded() {
        var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        counters.recordVisualEmbeddingMiss();

        #expect(counters.persistentPromptCacheVisualEmbeddingMisses == 1);
        #expect(counters.persistentPromptCacheVisualEmbeddingHits == 0);
        #expect(counters.persistentPromptCacheVisualEmbeddingRowsLoaded == 0);
    }

    @Test
    func should_count_partial_tail_hits_separately_from_full_block_hits() {
        var counters: PersistentPromptCacheCounters = PersistentPromptCacheCounters();
        counters.recordCacheHit(restoredTokenCount: 2_048);
        counters.recordPartialTailHit();
        counters.recordPartialTailHit();

        #expect(counters.persistentPromptCacheHits == 1);
        #expect(counters.persistentPromptCachePartialTailHits == 2);
    }
}
