import Foundation;

/// Outcome recorded when one bounded attribution report ends.
public enum PerformanceAttributionOutcome: String, Sendable {

    case success;
    case rejected;
    case cancelled;
    case failed;
}

/// Immutable metadata supplied when a model-loading report finishes.
public struct ModelLoadingPerformanceAttributionMetadata: Sendable {

    public let outcome: PerformanceAttributionOutcome;
    public let modelId: String?;
    public let modelRevision: String?;
    public let prefillTransientObservationCompleted: Bool;
    public let prefillObservedTransientHighWaterBytes: UInt64;
    public let totalArtifactPayloadBytes: UInt64?;
    public let residentModelPayloadBytes: UInt64?;
    public let modelShardCount: Int?;
    public let mlxActiveMemoryBytes: UInt64?;
    public let mlxAllocatorCacheMemoryBytes: UInt64?;
    public let mlxPeakMemoryBytes: UInt64?;
    public let failureDescription: String?;

    public init(
        outcome: PerformanceAttributionOutcome,
        modelId: String?,
        modelRevision: String?,
        prefillTransientObservationCompleted: Bool,
        prefillObservedTransientHighWaterBytes: UInt64,
        totalArtifactPayloadBytes: UInt64?,
        residentModelPayloadBytes: UInt64?,
        modelShardCount: Int?,
        mlxActiveMemoryBytes: UInt64?,
        mlxAllocatorCacheMemoryBytes: UInt64?,
        mlxPeakMemoryBytes: UInt64?,
        failureDescription: String?
    ) {
        self.outcome = outcome;
        self.modelId = modelId;
        self.modelRevision = modelRevision;
        self.prefillTransientObservationCompleted = prefillTransientObservationCompleted;
        self.prefillObservedTransientHighWaterBytes = prefillObservedTransientHighWaterBytes;
        self.totalArtifactPayloadBytes = totalArtifactPayloadBytes;
        self.residentModelPayloadBytes = residentModelPayloadBytes;
        self.modelShardCount = modelShardCount;
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes;
        self.mlxAllocatorCacheMemoryBytes = mlxAllocatorCacheMemoryBytes;
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes;
        self.failureDescription = failureDescription;
    }
}

/// Immutable metadata supplied when a generation report finishes.
public struct GenerationPerformanceAttributionMetadata: Sendable {

    public let outcome: PerformanceAttributionOutcome;
    public let modelId: String;
    public let modelRevision: String;
    public let prefillTransientObservationCompleted: Bool;
    public let prefillObservedTransientHighWaterBytes: UInt64;
    public let requestId: UInt64;
    public let configuredMaximumOutputTokens: UInt16;
    public let mlxActiveMemoryBytes: UInt64?;
    public let mlxAllocatorCacheMemoryBytes: UInt64?;
    public let mlxPeakMemoryBytes: UInt64?;
    public let failureDescription: String?;

    public init(
        outcome: PerformanceAttributionOutcome,
        modelId: String,
        modelRevision: String,
        prefillTransientObservationCompleted: Bool,
        prefillObservedTransientHighWaterBytes: UInt64,
        requestId: UInt64,
        configuredMaximumOutputTokens: UInt16,
        mlxActiveMemoryBytes: UInt64?,
        mlxAllocatorCacheMemoryBytes: UInt64?,
        mlxPeakMemoryBytes: UInt64?,
        failureDescription: String?
    ) {
        self.outcome = outcome;
        self.modelId = modelId;
        self.modelRevision = modelRevision;
        self.prefillTransientObservationCompleted = prefillTransientObservationCompleted;
        self.prefillObservedTransientHighWaterBytes = prefillObservedTransientHighWaterBytes;
        self.requestId = requestId;
        self.configuredMaximumOutputTokens = configuredMaximumOutputTokens;
        self.mlxActiveMemoryBytes = mlxActiveMemoryBytes;
        self.mlxAllocatorCacheMemoryBytes = mlxAllocatorCacheMemoryBytes;
        self.mlxPeakMemoryBytes = mlxPeakMemoryBytes;
        self.failureDescription = failureDescription;
    }
}
