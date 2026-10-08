import Foundation;

/// Cumulative persistent prompt-cache observability counters, port of the
/// Rust `PersistentPromptCacheCounters`. They track how often the cache
/// restored a prompt prefix (hit) versus fell back to cold prefill (miss),
/// how many prompt tokens were saved across all hits, and how often
/// persisted visual embeddings were loaded instead of recomputed.
public struct PersistentPromptCacheCounters: Equatable, Sendable {

    private var persistentPromptCacheHitsValue: UInt64;
    private var persistentPromptCacheMissesValue: UInt64;
    private var persistentPromptCacheTokensSavedValue: UInt64;
    private var persistentPromptCachePartialTailHitsValue: UInt64;
    private var persistentPromptCacheVisualEmbeddingHitsValue: UInt64;
    private var persistentPromptCacheVisualEmbeddingMissesValue: UInt64;
    private var persistentPromptCacheVisualEmbeddingRowsLoadedValue: UInt64;

    public init() {
        self.persistentPromptCacheHitsValue = 0;
        self.persistentPromptCacheMissesValue = 0;
        self.persistentPromptCacheTokensSavedValue = 0;
        self.persistentPromptCachePartialTailHitsValue = 0;
        self.persistentPromptCacheVisualEmbeddingHitsValue = 0;
        self.persistentPromptCacheVisualEmbeddingMissesValue = 0;
        self.persistentPromptCacheVisualEmbeddingRowsLoadedValue = 0;
    }

    /// Records one successful persistent prompt-cache prefix restore.
    ///
    /// `restoredTokenCount` is the number of prompt tokens the cache
    /// supplied, which is accumulated into the tokens-saved total.
    public mutating func recordCacheHit(restoredTokenCount: Int) {
        self.persistentPromptCacheHitsValue = self.persistentPromptCacheHitsValue &+ 1;
        self.persistentPromptCacheTokensSavedValue = self.persistentPromptCacheTokensSavedValue
            &+ UInt64(max(restoredTokenCount, 0));
    }

    /// Records one persistent prompt-cache miss (cold prefill fallback).
    public mutating func recordCacheMiss() {
        self.persistentPromptCacheMissesValue = self.persistentPromptCacheMissesValue &+ 1;
    }

    /// Records that one successful restore reused a partial tail block from
    /// a previous turn, so the request only prefilled the uncached suffix.
    public mutating func recordPartialTailHit() {
        self.persistentPromptCachePartialTailHitsValue =
            self.persistentPromptCachePartialTailHitsValue &+ 1;
    }

    /// Records one successful persistent visual-embedding file restore.
    public mutating func recordVisualEmbeddingHit(visualEmbeddingRowCount: Int) {
        self.persistentPromptCacheVisualEmbeddingHitsValue =
            self.persistentPromptCacheVisualEmbeddingHitsValue &+ 1;
        self.persistentPromptCacheVisualEmbeddingRowsLoadedValue =
            self.persistentPromptCacheVisualEmbeddingRowsLoadedValue
                &+ UInt64(max(visualEmbeddingRowCount, 0));
    }

    /// Records one persistent visual-embedding cache miss.
    public mutating func recordVisualEmbeddingMiss() {
        self.persistentPromptCacheVisualEmbeddingMissesValue =
            self.persistentPromptCacheVisualEmbeddingMissesValue &+ 1;
    }

    /// The cumulative number of successful persistent prompt-cache restores.
    public var persistentPromptCacheHits: UInt64 {
        return self.persistentPromptCacheHitsValue;
    }

    /// The cumulative number of persistent prompt-cache misses.
    public var persistentPromptCacheMisses: UInt64 {
        return self.persistentPromptCacheMissesValue;
    }

    /// The cumulative number of prompt tokens restored from the persistent prompt cache.
    public var persistentPromptCacheTokensSaved: UInt64 {
        return self.persistentPromptCacheTokensSavedValue;
    }

    /// How many restores reused a partial tail block from a previous turn.
    public var persistentPromptCachePartialTailHits: UInt64 {
        return self.persistentPromptCachePartialTailHitsValue;
    }

    /// How many visual embedding files were loaded from the persistent prompt cache.
    public var persistentPromptCacheVisualEmbeddingHits: UInt64 {
        return self.persistentPromptCacheVisualEmbeddingHitsValue;
    }

    /// How many visual embedding files were absent, invalid, or had to be recomputed.
    public var persistentPromptCacheVisualEmbeddingMisses: UInt64 {
        return self.persistentPromptCacheVisualEmbeddingMissesValue;
    }

    /// How many visual embedding rows were loaded from SSD-backed files.
    public var persistentPromptCacheVisualEmbeddingRowsLoaded: UInt64 {
        return self.persistentPromptCacheVisualEmbeddingRowsLoadedValue;
    }

    /// The hit rate as `hits / (hits + misses)`, rounded to 4 decimals.
    /// Returns `0.0` when no queries have occurred, avoiding division by zero.
    public func persistentPromptCacheHitRate() -> Double {
        let totalQueries: UInt64 = self.persistentPromptCacheHitsValue
            &+ self.persistentPromptCacheMissesValue;
        if totalQueries == 0 {
            return 0.0;
        }
        let hitRate: Double = Double(self.persistentPromptCacheHitsValue)
            / Double(totalQueries);
        return (hitRate * 10_000.0).rounded() / 10_000.0;
    }
}
