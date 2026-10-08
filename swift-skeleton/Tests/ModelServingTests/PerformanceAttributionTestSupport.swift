import Foundation

import Testing

import ModelServing

/// Shared evidence factories for the performance-attribution suites, port of
/// crates/model-serving/tests/hermetic/performance_attribution/support.rs.
enum PerformanceAttributionTestSupport {

    static let attributedPrefillTransientHighWaterBytes = UInt64(1_234_567);

    static func generationMetadata(
        outcome: PerformanceAttributionOutcome
    ) -> GenerationPerformanceAttributionMetadata {
        GenerationPerformanceAttributionMetadata(
            outcome: outcome,
            modelId: "Qwen3.6-35B-A3B-OptiQ-4bit",
            modelRevision: "revision",
            prefillTransientObservationCompleted: true,
            prefillObservedTransientHighWaterBytes: attributedPrefillTransientHighWaterBytes,
            requestId: 42,
            configuredMaximumOutputTokens: 512,
            mlxActiveMemoryBytes: 1,
            mlxAllocatorCacheMemoryBytes: 2,
            mlxPeakMemoryBytes: 3,
            failureDescription: "simulated generation failure");
    }

    static func modelLoadingMetadata(
        outcome: PerformanceAttributionOutcome
    ) -> ModelLoadingPerformanceAttributionMetadata {
        ModelLoadingPerformanceAttributionMetadata(
            outcome: outcome,
            modelId: "Qwen3.6-35B-A3B-OptiQ-4bit",
            modelRevision: "revision",
            prefillTransientObservationCompleted: false,
            prefillObservedTransientHighWaterBytes: 0,
            totalArtifactPayloadBytes: 22_135_339_264,
            residentModelPayloadBytes: 2_539_550_976,
            modelShardCount: 5,
            mlxActiveMemoryBytes: 2_539_550_976,
            mlxAllocatorCacheMemoryBytes: 0,
            mlxPeakMemoryBytes: 2_539_550_976,
            failureDescription: nil);
    }

    static func serializeGenerationReport(
        _ performanceAttribution: PerformanceAttribution,
        _ performanceAttributionOutcome: PerformanceAttributionOutcome
    ) throws -> [String: Any] {
        let performanceAttributionReport = try #require(
            performanceAttribution.finishGeneration(
                generationMetadata(outcome: performanceAttributionOutcome)),
            "enabled attribution should produce one generation report");
        return try serialize(performanceAttributionReport);
    }

    static func serializeModelLoadingReport(
        _ performanceAttribution: PerformanceAttribution,
        _ performanceAttributionOutcome: PerformanceAttributionOutcome
    ) throws -> [String: Any] {
        let performanceAttributionReport = try #require(
            performanceAttribution.finishModelLoading(
                modelLoadingMetadata(outcome: performanceAttributionOutcome)),
            "enabled attribution should produce one model-loading report");
        return try serialize(performanceAttributionReport);
    }

    static func serialize(
        _ performanceAttributionReport: PerformanceAttributionReport
    ) throws -> [String: Any] {
        let serializedReport = try JSONEncoder().encode(performanceAttributionReport);
        let jsonObject = try JSONSerialization.jsonObject(with: serializedReport);
        return try #require(jsonObject as? [String: Any]);
    }

    static func serializedText(
        _ performanceAttributionReport: PerformanceAttributionReport
    ) throws -> String {
        let serializedReport = try JSONEncoder().encode(performanceAttributionReport);
        return String(decoding: serializedReport, as: UTF8.self);
    }

    static func integer(
        _ json: [String: Any],
        _ key: String
    ) -> Int? {
        (json[key] as? NSNumber)?.intValue;
    }

    static func text(
        _ json: [String: Any],
        _ key: String
    ) -> String? {
        json[key] as? String;
    }

    static func boolean(
        _ json: [String: Any],
        _ key: String
    ) -> Bool? {
        json[key] as? Bool;
    }

    static func rows(
        _ json: [String: Any],
        _ key: String
    ) -> [[String: Any]] {
        (json[key] as? [[String: Any]]) ?? [];
    }

    static func makeTemporaryDirectory() throws -> URL {
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true);
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true);
        return temporaryDirectory;
    }
}
