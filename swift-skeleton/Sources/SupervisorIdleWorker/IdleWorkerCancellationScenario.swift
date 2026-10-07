import Foundation

import IpcProtocol

/**
 * The scripted cancellation behaviors of the supervisor test worker,
 * migrating the cancellation arms of
 * apps/supervisor/tests/fixtures/scripted_worker.rs: chat models whose
 * requests never complete on their own, and the exact event sequence each
 * one produces when the supervisor asks to cancel.
 */
enum IdleWorkerCancellationScenario {

    static let UNACKNOWLEDGED_CANCELLATION_MODEL_ID: String =
        "astronomical/unacknowledged-cancellation-fixture"
    static let DELAYED_CANCELLATION_ACKNOWLEDGEMENT_MODEL_ID: String =
        "astronomical/delayed-cancellation-acknowledgement-fixture"
    static let UNEXPECTED_CANCELLATION_EVENT_MODEL_ID: String =
        "astronomical/unexpected-cancellation-event-fixture"
    static let CACHE_STATS_DURING_CANCELLATION_MODEL_ID: String =
        "astronomical/cache-stats-during-cancellation-fixture"
    static let MLX_MEMORY_DURING_CANCELLATION_MODEL_ID: String =
        "astronomical/mlx-memory-during-cancellation-fixture"
    static let MLX_MEMORY_CLEAR_DURING_CANCELLATION_MODEL_ID: String =
        "astronomical/mlx-memory-clear-during-cancellation-fixture"
    static let CANCELLATION_ACKNOWLEDGEMENT_DELAY_SECONDS: TimeInterval = 4

    /// One in-flight request whose completion waits for the supervisor's
    /// cancel command; the kind selects the scripted acknowledgement.
    final class PendingCancellation {

        let cancellationKind: IdleWorkerCancellationScenario.CancellationKind;

        init(cancellationKind: IdleWorkerCancellationScenario.CancellationKind) {
            self.cancellationKind = cancellationKind;
        }
    }

    enum CancellationKind {

        /// Ignores the cancel command outright, forcing the bounded
        /// acknowledgement timeout and worker replacement.
        case unacknowledged;

        /// Acknowledges only after a delay longer than the default bound.
        case delayedAcknowledgement;

        /// Acknowledges a foreign request, breaching the cancellation
        /// protocol.
        case unexpectedEvent;

        /// Publishes prompt-cache telemetry first, then acknowledges.
        case cacheStatsThenAcknowledgement;

        /// Publishes an MLX memory sample first, then acknowledges.
        case publishMemoryThenAcknowledgement;

        /// Clears previously published memory telemetry, then acknowledges.
        case clearMemoryThenAcknowledgement;
    }

    /// Maps one chat command's model onto its scripted pending
    /// cancellation, or nil for models that complete normally.
    static func pendingCancellation(for generationCommand: ChatGenerationCommand) -> PendingCancellation? {
        let cancellationKind: IdleWorkerCancellationScenario.CancellationKind?;
        switch (generationCommand.model) {
        case IdleWorkerCancellationScenario.UNACKNOWLEDGED_CANCELLATION_MODEL_ID:
            cancellationKind = .unacknowledged;
        case IdleWorkerCancellationScenario.DELAYED_CANCELLATION_ACKNOWLEDGEMENT_MODEL_ID:
            cancellationKind = .delayedAcknowledgement;
        case IdleWorkerCancellationScenario.UNEXPECTED_CANCELLATION_EVENT_MODEL_ID:
            cancellationKind = .unexpectedEvent;
        case IdleWorkerCancellationScenario.CACHE_STATS_DURING_CANCELLATION_MODEL_ID:
            cancellationKind = .cacheStatsThenAcknowledgement;
        case IdleWorkerCancellationScenario.MLX_MEMORY_DURING_CANCELLATION_MODEL_ID:
            cancellationKind = .publishMemoryThenAcknowledgement;
        case IdleWorkerCancellationScenario.MLX_MEMORY_CLEAR_DURING_CANCELLATION_MODEL_ID:
            cancellationKind = .clearMemoryThenAcknowledgement;
        default:
            cancellationKind = nil;
        }
        guard let cancellationKind = cancellationKind else {
            return nil;
        }
        return PendingCancellation(cancellationKind: cancellationKind);
    }

    /// The events the scripted model emits when its generation command
    /// arrives: everything except the clear-memory fixture stays silent so
    /// the request remains cancellable; the clear-memory fixture seeds a
    /// visible snapshot the later cancellation clears.
    static func emitOnGenerate(
        _ pendingCancellation: PendingCancellation,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        if (pendingCancellation.cancellationKind == .clearMemoryThenAcknowledgement) {
            try eventWriter.sendEvent(.mlxMemorySample(
                mlxMemorySnapshot: IdleWorkerCancellationScenario.memorySnapshot(
                    activeMemoryBytes: 33_000),
                expertResidency: nil));
        }
    }

    /// Drives the scripted acknowledgement for one cancel command; the
    /// default acknowledgement is an immediate cancelled completion.
    static func acknowledgeCancellation(
        _ pendingCancellation: PendingCancellation?,
        requestId: RequestId,
        eventWriter: ProtocolWriter
    ) throws -> Void {
        guard let pendingCancellation = pendingCancellation else {
            try eventWriter.sendEvent(IdleWorkerCancellationScenario.cancelledCompletion(requestId));
            return;
        }
        switch (pendingCancellation.cancellationKind) {
        case .unacknowledged:
            return;
        case .delayedAcknowledgement:
            Thread.sleep(forTimeInterval: IdleWorkerCancellationScenario.CANCELLATION_ACKNOWLEDGEMENT_DELAY_SECONDS);
            try eventWriter.sendEvent(IdleWorkerCancellationScenario.cancelledCompletion(requestId));
        case .unexpectedEvent:
            try eventWriter.sendEvent(IdleWorkerCancellationScenario.cancelledCompletion(
                RequestId(rawRequestId: 2)));
        case .cacheStatsThenAcknowledgement:
            try eventWriter.sendEvent(.persistentPromptCacheStats(
                WorkerPersistentPromptCacheStats(
                    persistentPromptCacheHits: 1,
                    persistentPromptCacheMisses: 0,
                    persistentPromptCacheTokensSaved: 2_048,
                    persistentPromptCachePartialTailHits: 0,
                    persistentPromptCacheBlockTokenCount: 2_048,
                    persistentPromptCacheSequenceStateBlockCount: 1,
                    persistentPromptCacheBoundaryStateSnapshotCount: 1,
                    persistentPromptCacheVisualEmbeddingCount: 0,
                    persistentPromptCacheTotalSizeBytes: 4_096,
                    persistentPromptCacheVisualEmbeddingTotalSizeBytes: 0,
                    persistentPromptCacheMaximumSizeBytes: 50_000_000_000,
                    persistentPromptCacheVisualEmbeddingHits: 0,
                    persistentPromptCacheVisualEmbeddingMisses: 0,
                    persistentPromptCacheVisualEmbeddingRowsLoaded: 0)));
            try eventWriter.sendEvent(IdleWorkerCancellationScenario.cancelledCompletion(requestId));
        case .publishMemoryThenAcknowledgement:
            try eventWriter.sendEvent(.mlxMemorySample(
                mlxMemorySnapshot: IdleWorkerCancellationScenario.memorySnapshot(
                    activeMemoryBytes: 44_000),
                expertResidency: nil));
            try eventWriter.sendEvent(IdleWorkerCancellationScenario.cancelledCompletion(requestId));
        case .clearMemoryThenAcknowledgement:
            try eventWriter.sendEvent(.mlxMemorySample(
                mlxMemorySnapshot: nil,
                expertResidency: nil));
            try eventWriter.sendEvent(IdleWorkerCancellationScenario.cancelledCompletion(requestId));
        }
    }

    /// Builds deterministic memory telemetry for cancellation journeys,
    /// mirroring the Rust cancellation_memory_snapshot.
    private static func memorySnapshot(activeMemoryBytes: UInt64) -> WorkerMlxMemorySnapshot {
        return WorkerMlxMemorySnapshot(
            source: .finalized,
            activeMemoryBytes: activeMemoryBytes,
            allocatorCacheMemoryBytes: 2_000,
            peakMemoryBytes: activeMemoryBytes + 1_000,
            expertPayloadBytes: 20_000,
            modelCorePayloadBytes: 10_000,
            contextStatePayloadBytes: 5_000,
            memoryCeilingUtilization: nil);
    }

    private static func cancelledCompletion(_ requestId: RequestId) -> WorkerEvent {
        return .completed(
            requestId: requestId,
            promptTokenCount: 1,
            generatedTokenCount: 0,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            persistentPromptCacheDiagnostics: nil,
            reason: ChatGenerationCompletionReason.cancelled);
    }
}
