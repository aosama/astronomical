import Foundation;

import IpcProtocol;
import RestContract;

/// Encodes bounded supervisor chat events into OpenAI-compatible SSE frames.
///
/// Port of apps/supervisor/src/openai_chat_stream.rs's OpenAiChatStreamEncoder.
/// The synchronous Swift serving model has the full ordered event list before
/// the response starts, so the encoder returns the complete frame sequence as
/// text and the transport writes it as one event-stream body.
struct OpenAiChatStreamEncoder {

    private static let doneFrame: String = "data: [DONE]\n\n";

    private let requestId: UInt64;
    private let completionId: String;
    private let createdUnixSeconds: UInt64;
    private let modelId: String;
    private let includesUsage: Bool;
    private let reasoningExcluded: Bool;

    init(
        requestId: UInt64,
        completionId: String,
        createdUnixSeconds: UInt64,
        modelId: String,
        includesUsage: Bool,
        reasoningExcluded: Bool
    ) {
        self.requestId = requestId;
        self.completionId = completionId;
        self.createdUnixSeconds = createdUnixSeconds;
        self.modelId = modelId;
        self.includesUsage = includesUsage;
        self.reasoningExcluded = reasoningExcluded;
    }

    /// The leading assistant-role chunk every OpenAI-compatible client expects.
    func initialFrame() throws -> String {
        return try OpenAiChatStreamEncoder.dataFrame(OpenAiChatCompletionChunk.assistantRole(
            id: self.completionId,
            created: self.createdUnixSeconds,
            model: self.modelId).wireValue());
    }

    /// Encodes one ordered worker event into its SSE frames. A delta yields
    /// one frame, a terminal completion yields the finish chunk plus the
    /// `[DONE]` terminator, and a failure yields one error frame that ends
    /// the stream without a terminator.
    func encode(_ streamEvent: ChatGenerationStreamEvent) throws -> Array<String> {
        switch (streamEvent) {
        case let .reasoningFragment(reasoning):
            if self.reasoningExcluded {
                // The model still thought; only the client-visible deltas are withheld.
                return Array();
            }
            return [try OpenAiChatStreamEncoder.dataFrame(OpenAiChatCompletionChunk.reasoningDelta(
                id: self.completionId,
                created: self.createdUnixSeconds,
                model: self.modelId,
                reasoningContent: reasoning).wireValue())];
        case let .textFragment(text):
            return [try OpenAiChatStreamEncoder.dataFrame(OpenAiChatCompletionChunk.textDelta(
                id: self.completionId,
                created: self.createdUnixSeconds,
                model: self.modelId,
                text: text).wireValue())];
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            return [try OpenAiChatStreamEncoder.dataFrame(OpenAiChatCompletionChunk.toolCallDelta(
                id: self.completionId,
                created: self.createdUnixSeconds,
                model: self.modelId,
                toolCallIndex: toolCallIndex,
                toolCallId: "call_\(self.completionId)_\(toolCallIndex)",
                functionName: functionName,
                functionArguments: argumentsJson).wireValue())];
        case .prefillProgress:
            return Array();
        case let .completed(promptTokenCount, generatedTokenCount, _, cachedTokenCount, completionReason):
            return try self.completedFrames(
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                cachedTokenCount: cachedTokenCount,
                completionReason: completionReason);
        case let .failed(failureReason):
            return [try OpenAiChatStreamEncoder.dataFrame(
                OpenAiChatCompletionCollector.failureEnvelope(failureReason).wireValue())];
        case .streamError:
            return [try OpenAiChatStreamEncoder.dataFrame(OpenAiErrorResponse.serviceUnavailable(
                message: "the local worker became unavailable while processing the chat request",
                code: "chat_worker_unavailable").wireValue())];
        }
    }

    private func completedFrames(
        promptTokenCount: UInt32,
        generatedTokenCount: UInt16,
        cachedTokenCount: UInt32,
        completionReason: ChatGenerationCompletionReason
    ) throws -> Array<String> {
        let finishReason: OpenAiFinishReason;
        switch (completionReason) {
        case .endOfSequence, .cancelled:
            finishReason = .stop;
        case .maximumOutputTokens:
            finishReason = .length;
        case .toolCalls:
            finishReason = .toolCalls;
        }
        var completionChunk: OpenAiChatCompletionChunk = OpenAiChatCompletionChunk.finished(
            id: self.completionId,
            created: self.createdUnixSeconds,
            model: self.modelId,
            finishReason: finishReason);
        if self.includesUsage,
           let tokenUsage: OpenAiTokenUsage = OpenAiTokenUsage.new(
            promptTokens: promptTokenCount,
            completionTokens: UInt32(generatedTokenCount)) {
            completionChunk = completionChunk.withUsage(
                usage: tokenUsage.withCachedTokens(cachedTokens: cachedTokenCount));
        }
        return [
            try OpenAiChatStreamEncoder.dataFrame(completionChunk.wireValue()),
            OpenAiChatStreamEncoder.doneFrame,
        ];
    }

    private static func dataFrame(_ wireValue: JsonWireValue) throws -> String {
        return "data: \(try wireValue.serializedText)\n\n";
    }
}
