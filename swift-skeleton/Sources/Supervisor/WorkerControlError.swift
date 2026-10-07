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

    /// A replacement candidate acknowledged a semantic configuration other
    /// than the requested one.
    case candidateConfigurationGenerationMismatch;

    /// A replacement candidate violated the initial readiness handshake.
    case candidateProtocolViolation(description: String);

    /// A replacement candidate emitted an event that is invalid before
    /// serving begins.
    case unexpectedCandidateEvent(unexpectedWorkerEventSummary: String);

    /// The worker emitted an unexpected event while a cancellation was in
    /// flight, so the worker can no longer be trusted.
    case unexpectedCancellationEvent(requestId: UInt64, unexpectedWorkerEventSummary: String);

    /// The worker never acknowledged an in-flight cancellation in time.
    case cancellationAckTimeout(cancellationTimeoutMillis: UInt64);

    /// The worker closed its event stream before the awaited state arrived.
    case workerEventStreamClosed;

    /// The worker process closed IPC and supplied bounded process diagnostics.
    case workerProcessExited(processExitStatus: String, workerLifetimeMillis: UInt64, stderrTail: String);

    /// No living worker process exists to receive the control command.
    case missingActiveWorker;

    /// A generation is active or queued, so control actions that would
    /// disturb serving must wait.
    case generationBusy;

    /// The worker stayed responsive but never acknowledged the requested
    /// memory-ceiling change in time.
    case mlxMemoryLimitUpdateTimeout(memoryLimitUpdateTimeoutMillis: UInt64);

    /// The worker stayed responsive but never acknowledged the requested
    /// prompt-cache deletion in time.
    case promptCacheClearTimeout(cacheClearTimeoutMillis: UInt64);

    /// One line rendering of any control error, preferring the typed
    /// description so nested errors keep their diagnostic text when they
    /// are embedded into another error or a log line.
    public static func describe(_ error: Error) -> String {
        if let workerControlError: WorkerControlError = error as? WorkerControlError {
            return workerControlError.errorDescription ?? String(describing: error);
        }
        if let localizedError: LocalizedError = error as? LocalizedError,
           let errorDescription: String = localizedError.errorDescription {
            return errorDescription;
        }
        return String(describing: error);
    }

    public var errorDescription: String? {
        switch (self) {
        case let .startWorker(underlyingDescription):
            return "failed to start worker process: \(underlyingDescription)";
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
        case .candidateConfigurationGenerationMismatch:
            return "candidate worker acknowledged a different configuration generation";
        case let .candidateProtocolViolation(description):
            return "candidate worker protocol violation: \(description)";
        case let .unexpectedCandidateEvent(unexpectedWorkerEventSummary):
            return "candidate worker emitted an invalid startup event: \(unexpectedWorkerEventSummary)";
        case let .unexpectedCancellationEvent(requestId, unexpectedWorkerEventSummary):
            return "worker emitted an unexpected event while cancelling request \(requestId): \(unexpectedWorkerEventSummary)";
        case let .cancellationAckTimeout(cancellationTimeoutMillis):
            return "worker cancellation acknowledgement did not arrive within the \(cancellationTimeoutMillis)-millisecond cancellation timeout";
        case .workerEventStreamClosed:
            return "worker event stream closed before the awaited state arrived";
        case let .workerProcessExited(processExitStatus, workerLifetimeMillis, stderrTail):
            return "worker process exited after closing its event stream (\(processExitStatus)) "
                + "after \(workerLifetimeMillis) milliseconds; worker stderr tail: \(stderrTail)";
        case .missingActiveWorker:
            return "no living worker process exists to receive the control command";
        case .generationBusy:
            return "a generation is active or queued; the control action must wait";
        case let .mlxMemoryLimitUpdateTimeout(memoryLimitUpdateTimeoutMillis):
            return "worker did not acknowledge the memory-ceiling change within the \(memoryLimitUpdateTimeoutMillis)-millisecond timeout";
        case let .promptCacheClearTimeout(cacheClearTimeoutMillis):
            return "worker did not acknowledge the prompt-cache deletion within the \(cacheClearTimeoutMillis)-millisecond timeout";
        }
    }
}
