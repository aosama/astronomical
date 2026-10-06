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

    /// The worker closed its event stream before the awaited state arrived.
    case workerEventStreamClosed;

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
        case .workerEventStreamClosed:
            return "worker event stream closed before the awaited state arrived";
        }
    }
}
