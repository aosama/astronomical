import Foundation;

import IpcProtocol;

/// Engine-side metadata reported when one generation request starts.
///
/// Mirrors crates/model-serving/src/inference_engine/contract.rs
/// `EngineGenerationStart`: how much prompt work the engine restored from
/// reusable state, which model processes the next prompt phase, and the
/// sparse-expert mode the request opens under.
public struct EngineGenerationStart: Equatable {

    public let cachedTokenCount: UInt32;
    public let restoredPromptPrefixTokenCount: UInt32;
    public let expertMemoryMode: ExpertMemoryMode?;
    public let promptProcessingPhase: WorkerPromptProcessingPhase?;

    public init(
        cachedTokenCount: UInt32,
        restoredPromptPrefixTokenCount: UInt32? = nil,
        expertMemoryMode: ExpertMemoryMode? = nil,
        promptProcessingPhase: WorkerPromptProcessingPhase? = .target
    ) {
        self.cachedTokenCount = cachedTokenCount;
        self.restoredPromptPrefixTokenCount = restoredPromptPrefixTokenCount ?? cachedTokenCount;
        self.expertMemoryMode = expertMemoryMode;
        self.promptProcessingPhase = promptProcessingPhase;
    }
}
