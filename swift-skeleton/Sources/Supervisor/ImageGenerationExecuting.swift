import Foundation;

import IpcProtocol;

/// The supervisor-side answer to one image request: the encoded image plus
/// the reproducibility metadata the endpoint forwards. Mirrors the Rust
/// image executor's output.
public struct ImageGenerationOutput: Equatable {

    public let generatedImage: GeneratedImage;
    public let resultMetadata: ImageGenerationResultMetadata;

    public init(
        generatedImage: GeneratedImage,
        resultMetadata: ImageGenerationResultMetadata
    ) {
        self.generatedImage = generatedImage;
        self.resultMetadata = resultMetadata;
    }
}

/// Supervisor-owned bounds for image execution and lack of forward progress.
/// Mirrors ImageGenerationTimeouts in apps/supervisor/src/image_generation_executor.rs.
public struct ImageGenerationTimeouts: Equatable, Sendable {

    public static let `default`: ImageGenerationTimeouts = ImageGenerationTimeouts(
        executionTimeoutSeconds: 15 * 60,
        progressStallTimeoutSeconds: 3 * 60);

    public let executionTimeoutSeconds: TimeInterval;
    public let progressStallTimeoutSeconds: TimeInterval;

    public init(executionTimeoutSeconds: TimeInterval, progressStallTimeoutSeconds: TimeInterval) {
        self.executionTimeoutSeconds = executionTimeoutSeconds;
        self.progressStallTimeoutSeconds = progressStallTimeoutSeconds;
    }
}

/// A worker-side failure of one admitted image request, kept apart from the
/// start failures so the endpoint can map the two lifetimes differently.
public enum ImageGenerationExecutionError: Error, Equatable {

    /// The worker rejected or failed the request but stays responsive.
    case workerFailure(ImageGenerationFailureReason);
    /// The worker died or its event stream ended during the request.
    case workerUnavailable;
    /// The execution or forward-progress bound ran out before completion.
    case deadlineExceeded;
}

/// The image-generation face the daemon serves text-to-image through.
///
/// Mirrors the image half of apps/supervisor/src/image_generation_executor.rs:
/// the REST surface sees this boundary, never the worker process directly.
public protocol ImageGenerationExecuting: Sendable {

    /// Runs one bounded image request to its completed output. Throws
    /// GenerationStartError when the request cannot start and
    /// ImageGenerationExecutionError when the worker fails it mid-flight.
    func startImageGeneration(
        _ imageGenerationCommand: ImageGenerationCommand
    ) throws -> ImageGenerationOutput;

    /// One consistent worker health snapshot for gating and status.
    func workerHealthSnapshot() -> WorkerHealthSnapshot;
}
