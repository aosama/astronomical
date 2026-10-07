import Foundation;

/// Typed failures of supervisor-side worker control and event processing.
///
/// Mirrors apps/supervisor/src/worker_control_error.rs for the lifecycle
/// surface the Swift supervisor drives today; generation-command variants
/// join as their slices land.
public indirect enum WorkerControlError: Error, Equatable {

    /// The worker executable could not be started.
    case startWorker(underlyingDescription: String);

    /// The worker process ended before it was asked to.
    case workerExitedUnexpectedly(exitStatus: Int32);

    /// An operation failed and its best-effort cleanup failed too.
    case operationAndCleanupFailed(operationDescription: String, cleanupDescription: String);

    /// The worker event stream violated request expectations or event order.
    case workerProtocolViolation(description: String);

    /// The worker stayed responsive but did not finish loading in time.
    case modelLoadTimeout(modelLoadTimeoutMillis: UInt64);

    /// A replacement candidate did not complete its two-part startup
    /// acknowledgement in time.
    case candidateAcknowledgementTimeout(acknowledgementTimeoutMillis: UInt64);

    /// The worker closed its event stream before the awaited state arrived.
    case workerEventStreamClosed;

    /// No living worker process exists to receive the control command.
    case missingActiveWorker;

    /// The worker stayed responsive but never acknowledged the requested
    /// memory-ceiling change in time.
    case mlxMemoryLimitUpdateTimeout(memoryLimitUpdateTimeoutMillis: UInt64);

    /// The worker stayed responsive but never acknowledged the requested
    /// prompt-cache deletion in time.
    case promptCacheClearTimeout(cacheClearTimeoutMillis: UInt64);

    public var errorDescription: String? {
        switch (self) {
        case let .startWorker(underlyingDescription):
            return "worker could not be started: \(underlyingDescription)";
        case let .workerExitedUnexpectedly(exitStatus):
            return "worker exited unexpectedly with status \(exitStatus)";
        case let .operationAndCleanupFailed(operationDescription, cleanupDescription):
            return "worker operation failed: \(operationDescription); cleanup also failed: \(cleanupDescription)";
        case let .workerProtocolViolation(description):
            return "worker protocol event violated request expectations: \(description)";
        case let .modelLoadTimeout(modelLoadTimeoutMillis):
            return "worker did not finish loading the inference engine within the \(modelLoadTimeoutMillis)-millisecond timeout";
        case let .candidateAcknowledgementTimeout(acknowledgementTimeoutMillis):
            return "candidate worker did not acknowledge readiness and runtime configuration within the \(acknowledgementTimeoutMillis)-millisecond timeout";
        case .workerEventStreamClosed:
            return "worker event stream closed before the awaited state arrived";
        case .missingActiveWorker:
            return "no living worker process exists to receive the control command";
        case let .mlxMemoryLimitUpdateTimeout(memoryLimitUpdateTimeoutMillis):
            return "worker did not acknowledge the memory-ceiling change within the \(memoryLimitUpdateTimeoutMillis)-millisecond timeout";
        case let .promptCacheClearTimeout(cacheClearTimeoutMillis):
            return "worker did not acknowledge the prompt-cache deletion within the \(cacheClearTimeoutMillis)-millisecond timeout";
        }
    }
}
