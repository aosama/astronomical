import Foundation;

/// How a worker process ended when the supervisor closed it.
public enum WorkerTerminationOutcome: Equatable {

    /// Closing worker input was sufficient to terminate and reap the process.
    case graceful(processExitSuccessful: Bool);

    /// The supervisor had to send a termination signal before reaping.
    case forced(processExitSuccessful: Bool);
}
