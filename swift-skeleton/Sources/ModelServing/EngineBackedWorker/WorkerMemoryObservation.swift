import Foundation;

import IpcProtocol;

/// Wire-safety helpers translating engine and processor state into worker
/// events: memory snapshot sourcing and bounded failure reasons.
///
/// Mirrors crates/model-serving/src/engine_backed_worker/output.rs snapshot
/// mapping plus fatal.rs reason bounding.
enum WorkerMemoryObservation {

    /// Builds the wire snapshot for one engine observation at `source`.
    static func mlxMemorySnapshot(
        source: MlxMemorySnapshotSource,
        observedSnapshot: WorkerMlxMemorySnapshot?
    ) -> WorkerMlxMemorySnapshot? {
        guard let observedSnapshot = observedSnapshot else {
            return nil;
        }
        return WorkerMlxMemorySnapshot(
            source: source,
            activeMemoryBytes: observedSnapshot.activeMemoryBytes,
            allocatorCacheMemoryBytes: observedSnapshot.allocatorCacheMemoryBytes,
            peakMemoryBytes: observedSnapshot.peakMemoryBytes,
            expertPayloadBytes: observedSnapshot.expertPayloadBytes,
            modelCorePayloadBytes: observedSnapshot.modelCorePayloadBytes,
            contextStatePayloadBytes: observedSnapshot.contextStatePayloadBytes,
            memoryCeilingUtilization: observedSnapshot.memoryCeilingUtilization);
    }

    /// Builds the wire residency snapshot for one engine observation.
    static func expertResidencySnapshot(
        residencyTelemetry: ExpertResidencyTelemetry?
    ) -> WorkerExpertResidencySnapshot? {
        guard let residencyTelemetry = residencyTelemetry else {
            return nil;
        }
        return WorkerExpertResidencySnapshot(
            totalLayerCount: residencyTelemetry.totalLayerCount,
            residentExpertCount: residencyTelemetry.residentExpertCount,
            residentExpertPayloadBytes: residencyTelemetry.residentExpertPayloadBytes);
    }
}

/// A fatal runtime failure that ends the worker process after the wire
/// failure event reaches the supervisor.
enum WorkerRuntimeError: Error, Equatable {

    /// The engine reported a fatal execution failure; the wire reason is
    /// bounded, native detail stays in stderr.
    case inferenceEngineGenerationFailed(reason: String);
    /// The model swap path failed in a way the worker cannot contain.
    case modelSwapFailed(modelLoadFailureReason: String);

    /// Bounds any thrown failure into a wire-safe reason string; native
    /// detail travels in logs and stderr, never on the wire.
    static func boundedModelLoadFailureReason(_ failure: any Error) -> String {
        if let boundedFailure = failure as? InferenceEngineError {
            return boundedFailure.publicFailureReason;
        }
        let failureDescription: String = String(describing: failure);
        return failureDescription.count <= maximumBoundedReasonBytes
            ? failureDescription
            : String(failureDescription.prefix(maximumBoundedReasonBytes));
    }

    private static let maximumBoundedReasonBytes: Int = 512;
}
