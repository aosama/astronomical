import Foundation;

import IpcProtocol;

/// One bounded progress boundary from the engine.
///
/// Mirrors crates/model-serving/src/inference_engine/contract.rs
/// `GeneratedToken`. The worker loop advances one case per engine step and
/// translates it into worker events; the engine stays protocol-agnostic.
public enum GeneratedToken: Equatable {

    /// A generated token ID the active chat processor must decode.
    case tokenId(
        generatedTokenId: UInt32,
        isReasoningToken: Bool,
        expertMemoryMode: ExpertMemoryMode?,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        firstDecodeForwardElapsedMillis: UInt64?,
        generationFinalization: GenerationFinalization?);
    /// One bounded native prefill chunk completed without producing a token.
    case prefillProgress(
        processedTokenCount: UInt32,
        elapsedMillis: UInt64,
        forwardPrefillChunkElapsedMillis: UInt64,
        completedPrefillChunkTokens: UInt32,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?,
        expertResidencyTelemetry: ExpertResidencyTelemetry?,
        expertMemoryMode: ExpertMemoryMode?,
        promptWorkReuse: WorkerPromptWorkReuse);
    /// A confirmed prompt phase is about to begin before its blocking work.
    case promptProcessingPhaseStarted(
        promptProcessingPhase: WorkerPromptProcessingPhase,
        totalTokenCount: UInt32);
    /// Prefill is complete and the engine reconciles ownership before decode.
    case generationPreparationStarted(
        totalLayerCount: UInt32,
        residentExpertCount: UInt32,
        residentExpertPayloadBytes: UInt64,
        mlxMemorySnapshot: WorkerMlxMemorySnapshot?);
    /// Engine-side end-of-sequence without an explicit token ID.
    case endOfSequence;
}
