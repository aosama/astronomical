import Foundation;

import IpcProtocol;
import RestContract;

/// Internal assembly failure for one non-streaming chat completion.
enum OpenAiChatCompletionCollectorError: Error, CustomStringConvertible {

    case tokenUsageOverflow(promptTokenCount: UInt32, generatedTokenCount: UInt16);

    var description: String {
        switch (self) {
        case let .tokenUsageOverflow(promptTokenCount, generatedTokenCount):
            return "token usage overflowed: prompt_tokens=\(promptTokenCount), "
                + "completion_tokens=\(generatedTokenCount)";
        }
    }
}

/// Collects ordered worker chat events into one OpenAI-compatible
/// non-streaming response.
///
/// Port of apps/supervisor/src/openai_chat_completion.rs's
/// OpenAiChatCompletionCollector. Mirrors OpenAiChatStreamEncoder so
/// streaming and non-streaming modes stay behaviorally congruent: same
/// finish-reason mapping, same tool-call id format, same error code/message
/// pairs. Only the wire shape differs (one JSON body instead of an SSE frame
/// sequence).
struct OpenAiChatCompletionCollector {

    private let completionId: String;
    private let createdUnixSeconds: UInt64;
    private let modelId: String;
    private var textContent: String;
    private var reasoningContent: String;
    private let reasoningExcluded: Bool;
    private var toolCalls: Array<OpenAiResponseToolCall>;

    init(
        completionId: String,
        createdUnixSeconds: UInt64,
        modelId: String,
        reasoningExcluded: Bool
    ) {
        self.completionId = completionId;
        self.createdUnixSeconds = createdUnixSeconds;
        self.modelId = modelId;
        self.textContent = String();
        self.reasoningContent = String();
        self.reasoningExcluded = reasoningExcluded;
        self.toolCalls = Array();
    }

    /// Replaces accumulated visible text with the compact JSON it contains
    /// when structured output was requested: the prompt hint makes the model
    /// answer as JSON, so the client sees clean JSON instead of prose around
    /// it. Tool-call answers carry no visible JSON payload and stay as-is.
    mutating func replaceVisibleTextWithExtractedJson() -> Void {
        if self.toolCalls.isEmpty == false {
            return;
        }
        if let compactJsonText: String = OpenAiStructuredOutput.compact_extracted_json_text(self.textContent) {
            self.textContent = compactJsonText;
        }
    }

    /// Ingests one ordered worker event.
    ///
    /// Returns nil for collectable output events. Returns the OpenAI error
    /// envelope for Failed/streamError events so the caller can short-circuit
    /// with that body. The caller handles Completed before calling this.
    mutating func ingestEvent(_ streamEvent: ChatGenerationStreamEvent) -> OpenAiErrorResponse? {
        switch (streamEvent) {
        case let .reasoningFragment(reasoning):
            if self.reasoningExcluded {
                // The model still thought; only the client-visible text is withheld.
                return nil;
            }
            self.reasoningContent += reasoning;
            return nil;
        case let .textFragment(text):
            self.textContent += text;
            return nil;
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            let toolCallId: String = "call_\(self.completionId)_\(toolCallIndex)";
            self.toolCalls.append(OpenAiResponseToolCall.function(
                id: toolCallId, name: functionName, arguments: argumentsJson));
            return nil;
        case .prefillProgress:
            return nil;
        case .completed:
            return nil;
        case let .failed(failureReason):
            return OpenAiChatCompletionCollector.failureEnvelope(failureReason);
        case .streamError:
            return OpenAiErrorResponse.serviceUnavailable(
                message: "the local worker became unavailable while processing the chat request",
                code: "chat_worker_unavailable");
        }
    }

    /// The shared worker-failure envelope used by both the streaming encoder
    /// and this collector, so one rejection always reads the same way.
    static func failureEnvelope(
        _ failureReason: ChatGenerationFailureReason
    ) -> OpenAiErrorResponse {
        switch (failureReason) {
        case let .invalidRequest(reason):
            return OpenAiErrorResponse.serviceUnavailable(
                message: "the local worker rejected the chat request: \(reason)",
                code: "chat_invalid_request");
        case let .fatalExecution(reason):
            return OpenAiErrorResponse.serviceUnavailable(
                message: "the local worker stopped after a fatal model execution error: \(reason)",
                code: "chat_worker_unavailable");
        case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens):
            return OpenAiErrorResponse.invalidRequest(
                message: "requested context uses \(actualTotalContextTokens) tokens, exceeding the "
                    + "\(maximumContextTokens)-token model context window",
                parameter: "messages",
                code: "context_length_exceeded");
        case .engineBusy:
            return OpenAiErrorResponse.serviceUnavailable(
                message: "the local inference engine is already processing another request",
                code: "chat_engine_busy");
        case .malformedModelOutput:
            return OpenAiErrorResponse.serviceUnavailable(
                message: "the model produced malformed structured output",
                code: "chat_malformed_model_output");
        }
    }

    /// Builds the final non-streaming response after a Completed event.
    func intoResponse(
        promptTokenCount: UInt32,
        generatedTokenCount: UInt16,
        cachedTokenCount: UInt32,
        completionReason: ChatGenerationCompletionReason
    ) throws -> OpenAiChatCompletionResponse {
        let finishReason: OpenAiFinishReason;
        switch (completionReason) {
        case .endOfSequence, .cancelled:
            finishReason = .stop;
        case .maximumOutputTokens:
            finishReason = .length;
        case .toolCalls:
            finishReason = .toolCalls;
        }
        let visibleText: String? = self.textContent.isEmpty ? nil : self.textContent;
        let visibleReasoning: String? = self.reasoningContent.isEmpty ? nil : self.reasoningContent;
        let assistantMessage: OpenAiAssistantMessage = OpenAiAssistantMessage(
            content: visibleText,
            reasoningContent: visibleReasoning,
            toolCalls: self.toolCalls);
        guard let tokenUsage: OpenAiTokenUsage = OpenAiTokenUsage.new(
            promptTokens: promptTokenCount,
            completionTokens: UInt32(generatedTokenCount)) else {
            throw OpenAiChatCompletionCollectorError.tokenUsageOverflow(
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount);
        }
        return OpenAiChatCompletionResponse(
            id: self.completionId,
            created: self.createdUnixSeconds,
            model: self.modelId,
            message: assistantMessage,
            finishReason: finishReason,
            usage: tokenUsage.withCachedTokens(cachedTokens: cachedTokenCount));
    }
}
