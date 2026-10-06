import Foundation;

import IpcProtocol;

/// The model-specific chat translation seam between worker protocol and
/// engine: command validation, template application, and generated-token
/// decoding.
///
/// Mirrors crates/model-serving/src/model_generation_processor.rs. The
/// processor never touches MLX; token math belongs to the paired engine.
public protocol ModelGenerationProcessor: AnyObject {

    /// Reports the exact loaded model identity and output capabilities.
    func readyEvent() -> WorkerEvent;

    /// Validates the command and prepares one request with its translator;
    /// a rejected request throws `ChatPreparationRejection`.
    func prepareChatGeneration(
        _ chatGenerationCommand: ChatGenerationCommand
    ) throws -> any ActiveChatGeneration;
}

/// A request-scoped chat rejection the worker answers with `failed` while
/// the worker stays responsive; the reason is the bounded wire value.
struct ChatPreparationRejection: Error, @unchecked Sendable {
    let reason: ChatGenerationFailureReason;
}
