import Foundation;

import IpcProtocol;

/// The serving activity of the supervised worker, mirroring the Rust
/// `WorkerActivity`: one enum for the request-phase the status endpoint
/// reports, published from the generation event flow and reset to idle when
/// a request terminates.
public enum WorkerActivity: Equatable {

    /// No generation request is active.
    case idle;

    /// A request is active but no output has reached the supervisor yet.
    case promptProcessing;

    /// Prompt processing finished and expert ownership is being prepared for
    /// decode.
    case generationPreparation;

    /// At least one generated output has reached the supervisor.
    case generating;

    /// Native image conditioning, denoising, decoding, or encoding is active.
    case imageGeneration;

    /// Stable text used by the status document.
    public func activityText() -> String {
        switch (self) {
        case .idle: return "idle";
        case .promptProcessing: return "prompt_processing";
        case .generationPreparation: return "generation_preparation";
        case .generating: return "generating";
        case .imageGeneration: return "image_generation";
        }
    }
}

/// The latest progress observation for the active request, mirroring the
/// Rust `ActiveRequestProgress`. The started-at timestamps let the status
/// endpoint present live elapsed time that keeps advancing between worker
/// frames.
public enum ActiveRequestProgress: Equatable {

    /// Accumulated prompt-processing progress for the active request.
    case prefill(
        promptProcessingPhase: WorkerPromptProcessingPhase,
        processedTokens: UInt32,
        totalTokens: UInt32,
        requestStartedAt: Date,
        elapsedMillis: UInt64,
        completedPrefillChunkTokens: UInt32?);

    /// Prefill completed and the worker is preparing the first decode
    /// forward.
    case generationPreparation(
        requestStartedAt: Date,
        preparationStartedAt: Date,
        totalLayerCount: UInt32,
        residentExpertCount: UInt32,
        residentExpertPayloadBytes: UInt64);

    /// Accumulated generation progress for the active request.
    case generation(
        generatedTokenCount: UInt32,
        maximumOutputTokens: UInt32,
        elapsedMillis: UInt64);

    /// Current phase and denoising progress for one image request.
    case imageGeneration(
        phase: ImageGenerationPhase,
        completedSteps: UInt16,
        totalSteps: UInt16,
        elapsedMillis: UInt64);
}
