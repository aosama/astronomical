import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Persistent prompt-cache statistics journeys for GET /v1/cache/stats,
 * migrating apps/supervisor/tests/rest_api/cache_stats.rs: a ready worker's
 * latest stats observation forwards unchanged, an unavailable worker answers
 * with the zero state, and the hit rate is hits over total queries rounded
 * to four decimals.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestCacheStatsEndpointTests {

    static let configuredMaximumSizeBytes: UInt64 = 123_456_789;

    @Test
    func should_return_populated_cache_stats_for_a_ready_worker_with_cache() throws {
        let statusJourney: CacheStatsJourney = try CacheStatsJourney.launch(
            healthSnapshot: CacheStatsJourney.readyHealthSnapshotWithStats(
                CacheStatsJourney.populatedStats()));

        let statsDocument: [String: Any] = try statusJourney.getCacheStatsDocument();

        #expect(statsDocument["persistent_prompt_cache_hits"] as? UInt64 == 12);
        #expect(statsDocument["persistent_prompt_cache_misses"] as? UInt64 == 3);
        #expect(statsDocument["persistent_prompt_cache_tokens_saved"] as? UInt64 == 95_000);
        #expect(statsDocument["persistent_prompt_cache_block_token_count"] as? UInt64 == 2_048);
        #expect(statsDocument["persistent_prompt_cache_sequence_state_block_count"] as? UInt64 == 87);
        #expect(statsDocument["persistent_prompt_cache_boundary_state_snapshot_count"] as? UInt64 == 1);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_count"] as? UInt64 == 5);
        #expect(statsDocument["persistent_prompt_cache_total_size_bytes"] as? UInt64 == 1_073_741_824);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_total_size_bytes"] as? UInt64 == 222_222);
        #expect(
            statsDocument["persistent_prompt_cache_maximum_size_bytes"] as? UInt64
                == RestCacheStatsEndpointTests.configuredMaximumSizeBytes);
        #expect(statsDocument["persistent_prompt_cache_hit_rate"] as? Double == 0.8);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_hits"] as? UInt64 == 4);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_misses"] as? UInt64 == 2);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_rows_loaded"] as? UInt64 == 256);
        #expect(statsDocument["persistent_prompt_cache_partial_tail_hits"] as? UInt64 == 7);
        #expect(statsDocument["pending_cache_clear"] is NSNull);
    }

    @Test
    func should_return_zeroed_cache_stats_when_worker_is_unavailable() throws {
        let statusJourney: CacheStatsJourney = try CacheStatsJourney.launch(
            healthSnapshot: WorkerHealthSnapshot.unavailable(.unavailable));

        let statsDocument: [String: Any] = try statusJourney.getCacheStatsDocument();

        #expect(statsDocument["persistent_prompt_cache_hits"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_misses"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_tokens_saved"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_sequence_state_block_count"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_boundary_state_snapshot_count"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_count"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_total_size_bytes"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_total_size_bytes"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_maximum_size_bytes"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_hit_rate"] as? Double == 0);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_hits"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_misses"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_visual_embedding_rows_loaded"] as? UInt64 == 0);
        #expect(statsDocument["persistent_prompt_cache_partial_tail_hits"] as? UInt64 == 0);
    }

    @Test
    func should_return_200_ok_with_json_content_type_for_cache_stats() throws {
        let statusJourney: CacheStatsJourney = try CacheStatsJourney.launch(
            healthSnapshot: WorkerHealthSnapshot.unavailable(.unavailable));

        let statsResponse: RestHttpResponse = try statusJourney.getCacheStatsResponse();

        #expect(statsResponse.statusCode == 200);
        #expect(statsResponse.contentType.hasPrefix("application/json"));
    }

    @Test
    func should_compute_hit_rate_as_hits_over_total_queries() throws {
        let twoOfThreeStats: WorkerPersistentPromptCacheStats = CacheStatsJourney.populatedStats();
        let statusJourney: CacheStatsJourney = try CacheStatsJourney.launch(
            healthSnapshot: CacheStatsJourney.readyHealthSnapshotWithStats(
                WorkerPersistentPromptCacheStats(
                    persistentPromptCacheHits: 2,
                    persistentPromptCacheMisses: 1,
                    persistentPromptCacheTokensSaved: twoOfThreeStats.persistentPromptCacheTokensSaved,
                    persistentPromptCachePartialTailHits: twoOfThreeStats.persistentPromptCachePartialTailHits,
                    persistentPromptCacheBlockTokenCount: twoOfThreeStats.persistentPromptCacheBlockTokenCount,
                    persistentPromptCacheSequenceStateBlockCount: twoOfThreeStats.persistentPromptCacheSequenceStateBlockCount,
                    persistentPromptCacheBoundaryStateSnapshotCount: twoOfThreeStats.persistentPromptCacheBoundaryStateSnapshotCount,
                    persistentPromptCacheVisualEmbeddingCount: twoOfThreeStats.persistentPromptCacheVisualEmbeddingCount,
                    persistentPromptCacheTotalSizeBytes: twoOfThreeStats.persistentPromptCacheTotalSizeBytes,
                    persistentPromptCacheVisualEmbeddingTotalSizeBytes: twoOfThreeStats.persistentPromptCacheVisualEmbeddingTotalSizeBytes,
                    persistentPromptCacheMaximumSizeBytes: twoOfThreeStats.persistentPromptCacheMaximumSizeBytes,
                    persistentPromptCacheVisualEmbeddingHits: twoOfThreeStats.persistentPromptCacheVisualEmbeddingHits,
                    persistentPromptCacheVisualEmbeddingMisses: twoOfThreeStats.persistentPromptCacheVisualEmbeddingMisses,
                    persistentPromptCacheVisualEmbeddingRowsLoaded: twoOfThreeStats.persistentPromptCacheVisualEmbeddingRowsLoaded)));

        let statsDocument: [String: Any] = try statusJourney.getCacheStatsDocument();

        #expect(statsDocument["persistent_prompt_cache_hit_rate"] as? Double == 0.6667);
    }
}

/// One cache-stats journey: a serving route table over a health state the
/// journey publishes directly.
final class CacheStatsJourney {

    let routeTable: RestRouteTable;
    private let homeDirectoryUrl: URL;

    private init(routeTable: RestRouteTable, homeDirectoryUrl: URL) {
        self.routeTable = routeTable;
        self.homeDirectoryUrl = homeDirectoryUrl;
    }

    static func launch(healthSnapshot: WorkerHealthSnapshot) throws -> CacheStatsJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-cache-stats-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(healthSnapshot);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            workerHealthState: workerHealthState,
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity());
        return CacheStatsJourney(routeTable: routeTable, homeDirectoryUrl: homeDirectoryUrl);
    }

    func getCacheStatsResponse() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: "/v1/cache/stats");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw CacheStatsJourneyFailure.routeMissing;
        }
        return try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/cache/stats"));
    }

    func getCacheStatsDocument() throws -> [String: Any] {
        return try ConfigReloadJourney.decodeObject(try self.getCacheStatsResponse());
    }

    /// A ready snapshot carrying one published stats observation.
    static func readyHealthSnapshotWithStats(
        _ persistentPromptCacheStats: WorkerPersistentPromptCacheStats
    ) -> WorkerHealthSnapshot {
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.persistentPromptCacheStats = persistentPromptCacheStats;
        return healthSnapshot;
    }

    /// The populated observation the Rust journey pins.
    static func populatedStats() -> WorkerPersistentPromptCacheStats {
        return WorkerPersistentPromptCacheStats(
            persistentPromptCacheHits: 12,
            persistentPromptCacheMisses: 3,
            persistentPromptCacheTokensSaved: 95_000,
            persistentPromptCachePartialTailHits: 7,
            persistentPromptCacheBlockTokenCount: 2_048,
            persistentPromptCacheSequenceStateBlockCount: 87,
            persistentPromptCacheBoundaryStateSnapshotCount: 1,
            persistentPromptCacheVisualEmbeddingCount: 5,
            persistentPromptCacheTotalSizeBytes: 1_073_741_824,
            persistentPromptCacheVisualEmbeddingTotalSizeBytes: 222_222,
            persistentPromptCacheMaximumSizeBytes: RestCacheStatsEndpointTests.configuredMaximumSizeBytes,
            persistentPromptCacheVisualEmbeddingHits: 4,
            persistentPromptCacheVisualEmbeddingMisses: 2,
            persistentPromptCacheVisualEmbeddingRowsLoaded: 256);
    }
}

/// Typed failures of the cache-stats journeys.
enum CacheStatsJourneyFailure: Error {

    case routeMissing;
}
