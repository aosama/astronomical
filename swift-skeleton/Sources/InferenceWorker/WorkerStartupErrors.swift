import Foundation;

/// Typed startup failures of the model-less worker process.
///
/// Mirrors apps/inference-worker/src/worker_startup_error.rs. The variants
/// are process-boundary facts an operator sees on the worker's stderr; the
/// engine slices extend this set when model loading lands.
public enum WorkerStartupError: Error, Equatable {
    /// The first framed command was not InitializeWorker (or the stream ended
    /// before one arrived).
    case initializeWorkerNotFirst(description: String);
    /// The GPU wired-memory sysctl output was not a non-negative integer, or
    /// the byte derivation overflowed.
    case invalidGpuWiredMemoryLimit(description: String);
    /// The sysctl sample did not finish inside its bounded wait.
    case gpuWiredMemoryLimitSampleTimedOut;
    /// The sysctl process exited with a failure status.
    case gpuWiredMemoryLimitSampleFailed;
    /// Reading the sysctl output failed.
    case sampleGpuWiredMemoryLimit(description: String);
    /// The system-default GPU wired-memory policy could not be resolved.
    case readMlxRecommendedGpuWorkingSet(description: String);
}

/// Typed failures of the worker process lifecycle.
///
/// Mirrors apps/inference-worker/src/worker_startup_error.rs's
/// WorkerProcessError: a startup failure before the serving loop, or a
/// runtime failure while serving commands.
public enum WorkerProcessError: Error, Equatable, CustomStringConvertible {
    case startup(WorkerStartupError);
    /// A serving-loop failure; the description is the protocol-level reason.
    case runtime(description: String);

    public var description: String {
        switch self {
        case let .startup(startupError):
            return "worker startup failed: \(startupError)";
        case let .runtime(description):
            return description;
        }
    }
}

extension WorkerStartupError: CustomStringConvertible {

    public var description: String {
        switch self {
        case let .initializeWorkerNotFirst(description):
            return description;
        case let .invalidGpuWiredMemoryLimit(description):
            return description;
        case .gpuWiredMemoryLimitSampleTimedOut:
            return "the GPU wired-memory sysctl sample timed out";
        case .gpuWiredMemoryLimitSampleFailed:
            return "the GPU wired-memory sysctl sample failed";
        case let .sampleGpuWiredMemoryLimit(description):
            return description;
        case let .readMlxRecommendedGpuWorkingSet(description):
            return description;
        }
    }
}
