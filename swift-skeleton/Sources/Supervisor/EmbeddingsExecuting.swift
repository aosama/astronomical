import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// The supervisor-side answer to one embeddings request: one completed
/// vector per input in request order, plus the per-input token counts the
/// usage line aggregates. Mirrors the Rust embeddings output the executor
/// hands back.
public struct EmbeddingsOutput: Equatable {

    public let embeddings: Array<Array<Float>>;
    public let inputTokenCounts: Array<UInt32>;

    public init(embeddings: Array<Array<Float>>, inputTokenCounts: Array<UInt32>) {
        self.embeddings = embeddings;
        self.inputTokenCounts = inputTokenCounts;
    }
}

/// The embeddings-facing face the daemon serves embeddings through.
///
/// Mirrors the embeddings half of apps/supervisor/src/embeddings_executor.rs:
/// the REST surface sees this boundary, never the worker process directly.
/// The synchronous Swift serving model returns the completed output instead
/// of a result channel.
public protocol EmbeddingsExecuting: Sendable {

    /// Runs one bounded embeddings request to its completed output. Throws
    /// GenerationStartError when the request cannot start and
    /// EmbeddingsExecutionError when the worker fails it mid-flight.
    func startEmbeddingsGeneration(
        _ embeddingsCommand: EmbeddingsCommand
    ) throws -> EmbeddingsOutput;

    /// One consistent worker health snapshot for gating and status.
    func workerHealthSnapshot() -> WorkerHealthSnapshot;
}

/// A worker-side failure of one admitted embeddings request, kept apart from
/// the start failures so the endpoint can map the two lifetimes differently.
public enum EmbeddingsExecutionError: Error, Equatable {

    /// The worker rejected or failed the request but stays responsive.
    case workerFailure(EmbeddingsFailureReason);
    /// The worker died or its event stream ended during the request.
    case workerUnavailable;
}
