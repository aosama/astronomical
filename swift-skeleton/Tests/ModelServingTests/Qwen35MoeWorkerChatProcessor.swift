import Foundation;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * The chat processor of the MoE worker journey runtime: reports the MoE
 * model identity and capabilities, and prepares each request as a real
 * Qwen3.5 prepared inference request over the Romeo and Juliet fixture
 * bytes, so the paired resident MoE engine runs actual token ids through
 * the worker seam.
 */
final class Qwen35MoeWorkerChatProcessor: ModelGenerationProcessor {

    static let MODEL_IDENTIFIER: String = "journey-qwen3.5-moe";

    init() {
    }

    func readyEvent() -> WorkerEvent {
        return .ready(
            modelId: Self.MODEL_IDENTIFIER,
            capabilities: Self.chatCapabilities());
    }

    func prepareChatGeneration(
        _ chatGenerationCommand: ChatGenerationCommand
    ) throws -> any ActiveChatGeneration {
        return Qwen35MoeWorkerActiveGeneration(chatGenerationCommand: chatGenerationCommand);
    }

    static func chatCapabilities() -> WorkerModelCapabilities {
        return WorkerModelCapabilities(
            chat: ChatModelCapabilities(
                supportsReasoning: false,
                supportsToolCalls: false,
                hasVision: false,
                maxInputTokens: 3072,
                maxOutputTokens: 1024,
                contextWindow: 4096),
            imageGeneration: nil,
            embeddings: nil);
    }
}
