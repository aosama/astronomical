import Foundation;

import IpcProtocol;

/// Supervisor-side reason a chat stream cannot start.
///
/// Mirrors apps/supervisor/src/chat_generation_executor.rs's
/// GenerationStartError. The daemon IPC and REST surfaces map these to their
/// own user-facing rejection texts, so the enum stays reason-only.
public enum GenerationStartError: Error, Equatable {
    /// One generation is active and the bounded queue is full.
    case capacityUnavailable;
    /// The on-demand model load was rejected by the worker.
    case modelLoadFailed(modelLoadFailureReason: String);
    /// The framed command exceeds the worker IPC message bound.
    case requestTooLarge(actualIpcMessageBytes: Int, maximumIpcMessageBytes: Int);
    /// No living worker can take the request.
    case workerUnavailable;
}

/// Ordered application event produced by one chat request.
///
/// Mirrors apps/supervisor/src/chat_generation_executor.rs's
/// ChatGenerationStreamEvent: the worker's framed events folded into the
/// presentation order the daemon relays.
public enum ChatGenerationStreamEvent: Equatable {
    case reasoningFragment(String);
    case textFragment(String);
    case toolCall(toolCallIndex: UInt16, functionName: String, argumentsJson: String);
    case prefillProgress(
        processedTokens: UInt32,
        totalTokens: UInt32,
        elapsedMillis: UInt64,
        forwardPrefillChunkElapsedMillis: UInt64?,
        completedPrefillChunkTokens: UInt32?,
        mlxActiveMemoryBytes: UInt64?,
        mlxAllocatorCacheMemoryBytes: UInt64?,
        mlxPeakMemoryBytes: UInt64?);
    case completed(
        promptTokenCount: UInt32,
        generatedTokenCount: UInt16,
        reasoningTokenCount: UInt16,
        cachedTokenCount: UInt32,
        reason: ChatGenerationCompletionReason);
    case failed(reason: ChatGenerationFailureReason);
    case streamError(ChatGenerationStreamErrorCode);

    /// Folds one worker output frame into its ordered stream events.
    public static func fromWorkerOutput(
        _ workerOutput: ChatGenerationOutput
    ) -> ChatGenerationStreamEvent {
        switch (workerOutput) {
        case let .text(text):
            return .textFragment(text);
        case let .reasoning(text):
            return .reasoningFragment(text);
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            return .toolCall(
                toolCallIndex: toolCallIndex,
                functionName: functionName,
                argumentsJson: argumentsJson);
        }
    }

    /// Whether this event ends the chat stream.
    public var isTerminal: Bool {
        switch (self) {
        case .completed, .failed:
            return true;
        case .reasoningFragment, .textFragment, .toolCall, .prefillProgress, .streamError:
            return false;
        }
    }
}

/// Supervisor-side reason a chat stream cannot continue.
public enum ChatGenerationStreamErrorCode: Equatable {
    case workerUnavailable;
}
