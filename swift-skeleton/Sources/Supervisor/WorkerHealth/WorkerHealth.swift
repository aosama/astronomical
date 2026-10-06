import Foundation;

import IpcProtocol;

/// Coarse worker availability state exposed by the supervisor readiness
/// endpoint.
///
/// Migrates the shared state of apps/supervisor/src/worker_health.rs.
public enum WorkerHealthStatus: Equatable {

    /// The worker process is responsive but its inference engine is still loading.
    case loading;

    /// The worker is available for new requests.
    case ready;

    /// The worker is absent or otherwise unavailable.
    case unavailable;

    /// The daemon-protocol spelling of this availability state.
    public func daemonWorkerStatus() -> DaemonWorkerStatus {
        switch (self) {
        case .loading: return .loading;
        case .ready: return .ready;
        case .unavailable: return .unavailable;
        }
    }

    /// Whether the worker should currently receive new requests.
    public func isReady() -> Bool {
        switch (self) {
        case .ready: return true;
        case .loading, .unavailable: return false;
        }
    }
}

/// Snapshot of the supervisor's current worker health assessment.
///
/// Migrates the process-scoped slice of apps/supervisor/src/worker_health.rs
/// `WorkerHealthSnapshot`: lifecycle availability plus the worker
/// acknowledgements and MLX observations that stay valid regardless of request
/// phase. Per-request progress joins with the generation slices; health
/// publication stays push-based from worker events, never inferred from
/// silence.
public struct WorkerHealthSnapshot: Equatable {

    /// Coarse worker availability state.
    public var status: WorkerHealthStatus;
    /// Exact worker-reported model identity when the worker is ready.
    public var readyModelId: String?;
    /// Worker-reported typed model capabilities when the worker is ready.
    public var readyModelCapabilities: WorkerModelCapabilities?;
    /// Feature settings explicitly acknowledged by the currently running worker.
    public var workerRuntimeFeatureConfiguration: WorkerRuntimeFeatureConfiguration?;
    /// Immutable maximum reported by the worker's macOS/MLX startup probe.
    public var machineMlxMemoryCeilingBytes: UInt64;
    /// Safe ceiling the worker currently enforces.
    public var effectiveMlxMemoryCeilingBytes: UInt64;
    /// Safe idle lower bound reported by the loaded model or idle worker.
    public var minimumMlxMemoryCeilingBytes: UInt64;
    /// Latest worker-owned MLX allocator observation.
    public var latestMlxMemorySnapshot: WorkerMlxMemorySnapshot?;
    /// Latest worker-reported retained-expert topology.
    public var expertResidency: WorkerExpertResidencySnapshot?;

    /// Builds a ready snapshot from the worker's enriched readiness event.
    public static func readyWithModel(
        modelId: String,
        capabilities: WorkerModelCapabilities
    ) -> WorkerHealthSnapshot {
        return WorkerHealthSnapshot(
            status: .ready,
            readyModelId: modelId,
            readyModelCapabilities: capabilities,
            workerRuntimeFeatureConfiguration: nil,
            machineMlxMemoryCeilingBytes: 0,
            effectiveMlxMemoryCeilingBytes: 0,
            minimumMlxMemoryCeilingBytes: 1,
            latestMlxMemorySnapshot: nil,
            expertResidency: nil);
    }

    /// Builds a ready snapshot for an idle worker that has no resident model.
    public static func readyWithoutModel(
        machineMlxMemoryCeilingBytes: UInt64,
        effectiveMlxMemoryCeilingBytes: UInt64,
        minimumMlxMemoryCeilingBytes: UInt64
    ) -> WorkerHealthSnapshot {
        return WorkerHealthSnapshot(
            status: .ready,
            readyModelId: nil,
            readyModelCapabilities: nil,
            workerRuntimeFeatureConfiguration: nil,
            machineMlxMemoryCeilingBytes: machineMlxMemoryCeilingBytes,
            effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes,
            minimumMlxMemoryCeilingBytes: minimumMlxMemoryCeilingBytes,
            latestMlxMemorySnapshot: nil,
            expertResidency: nil);
    }

    /// Builds a non-ready snapshot.
    public static func unavailable(_ status: WorkerHealthStatus) -> WorkerHealthSnapshot {
        return WorkerHealthSnapshot(
            status: status,
            readyModelId: nil,
            readyModelCapabilities: nil,
            workerRuntimeFeatureConfiguration: nil,
            machineMlxMemoryCeilingBytes: 0,
            effectiveMlxMemoryCeilingBytes: 0,
            minimumMlxMemoryCeilingBytes: 1,
            latestMlxMemorySnapshot: nil,
            expertResidency: nil);
    }
}
