import Foundation;

import IpcProtocol;

/// One active chat generation the engine-backed worker advances stepwise.
///
/// Mirrors crates/model-serving/src/engine_backed_worker/support.rs
/// `ActiveEngineGeneration`: counters, reporting watermarks, and the
/// request-local output translator the advance loop consults.
struct ActiveEngineGeneration {

    let generationCommand: ChatGenerationCommand;
    let activeGeneration: any ActiveChatGeneration;
    /// The prompt tokens the engine still has to process, after restored
    /// prefix work is subtracted; prompt-work reuse retracts it mid-request.
    var requiredPromptProcessingTokenCount: UInt32;
    /// The request's output budget from the validated settings.
    let maximumOutputTokens: UInt16;

    var generatedTokenCount: UInt16 = 0;
    var reasoningTokenCount: UInt16 = 0;
    var prefillProcessedTokens: UInt32 = 0;
    var prefillElapsedMillis: UInt64 = 0;
    var nextSequenceNumber: UInt16 = 0;
    var nextToolCallIndex: UInt16 = 0;
    var hasEmittedToolCall: Bool = false;
    /// Set once the engine attached finalization to a terminal token, so the
    /// finish path must not cancel the engine a second time.
    var engineHasFinalizedGeneration: Bool = false;
    var promptWorkReuse: WorkerPromptWorkReuse = WorkerPromptWorkReuse(
        targetEligibleTokenCount: 0, targetRestoredTokenCount: 0);

    var requestId: RequestId {
        return self.generationCommand.requestId;
    }

    init(
        generationCommand: ChatGenerationCommand,
        activeGeneration: any ActiveChatGeneration,
        restoredPromptPrefixTokenCount: UInt32
    ) {
        self.generationCommand = generationCommand;
        self.activeGeneration = activeGeneration;
        self.maximumOutputTokens = generationCommand.settings.maxOutputTokens;
        let promptTokenCount: UInt32 = UInt32(clamping: activeGeneration.promptTokenCount);
        self.requiredPromptProcessingTokenCount = promptTokenCount
            > restoredPromptPrefixTokenCount
            ? promptTokenCount - restoredPromptPrefixTokenCount : 0;
    }

    /// Sets the decode-phase clock on the first produced token boundary.
    var generationStartedAt: ContinuousClock.Instant?;

    mutating func recordGeneratedToken(isReasoningToken: Bool) -> Void {
        self.generatedTokenCount = self.generatedTokenCount &+ 1;
        if isReasoningToken {
            self.reasoningTokenCount = self.reasoningTokenCount &+ 1;
        }
        if self.generationStartedAt == nil {
            self.generationStartedAt = ContinuousClock.now;
        }
    }
}
