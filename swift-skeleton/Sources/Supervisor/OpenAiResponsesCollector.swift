import Foundation;

import IpcProtocol;
import RestContract;

/// One assembly failure while collecting worker events into a Responses
/// object. Port of apps/supervisor/src/openai_responses_assembly.rs's
/// OpenAiResponsesAssemblyError.
enum OpenAiResponsesAssemblyError: Error, CustomStringConvertible {

    case unexpectedCompletionEvent;
    case workerFailure(failureReason: ChatGenerationFailureReason);
    case workerUnavailable;
    case cancelled;
    case systemClockBeforeUnixEpoch;
    case tokenUsageOverflow(inputTokenCount: UInt32, outputTokenCount: UInt16);

    var description: String {
        switch (self) {
        case .unexpectedCompletionEvent:
            return "received a completion event through the non-terminal ingestion path";
        case let .workerFailure(failureReason):
            return "the local worker rejected the response request: \(failureReason)";
        case .workerUnavailable:
            return "the local worker became unavailable while generating a response";
        case .cancelled:
            return "the response request was cancelled";
        case .systemClockBeforeUnixEpoch:
            return "the system clock predates the Unix epoch";
        case let .tokenUsageOverflow(inputTokenCount, outputTokenCount):
            return "token usage overflowed: input_tokens=\(inputTokenCount), "
                + "output_tokens=\(outputTokenCount)";
        }
    }
}

/// One function call collected from the worker tool-call events.
struct CollectedFunctionCall: Equatable {

    let toolCallIndex: UInt16;
    let functionName: String;
    let argumentsJson: String;
}

/// Collects ordered worker events into one terminal Responses object.
///
/// Port of apps/supervisor/src/openai_responses_assembly.rs's
/// OpenAiResponsesCollector. Mirrors OpenAiResponsesStreamEncoder so
/// streaming and non-streaming modes assemble the same output items: the
/// reasoning summary first, the visible message second, then the function
/// calls in arrival order.
struct OpenAiResponsesCollector {

    private let responseId: String;
    private let createdAtUnixSeconds: UInt64;
    private let modelId: String;
    private let instructions: String?;
    private let requestConfiguration: OpenAiResponseRequestConfiguration;
    private var reasoningText: String;
    private var outputText: String;
    private var functionCalls: Array<CollectedFunctionCall>;

    init(
        responseId: String,
        createdAtUnixSeconds: UInt64,
        modelId: String,
        instructions: String?,
        requestConfiguration: OpenAiResponseRequestConfiguration
    ) {
        self.responseId = responseId;
        self.createdAtUnixSeconds = createdAtUnixSeconds;
        self.modelId = modelId;
        self.instructions = instructions;
        self.requestConfiguration = requestConfiguration;
        self.reasoningText = String();
        self.outputText = String();
        self.functionCalls = Array();
    }

    /// Replaces accumulated visible text with the compact JSON it contains
    /// when structured output was requested. Tool-call answers carry no
    /// visible JSON payload and stay as-is.
    mutating func replaceOutputTextWithExtractedJson() -> Void {
        if self.functionCalls.isEmpty == false {
            return;
        }
        if let compactJsonText: String = OpenAiStructuredOutput.compact_extracted_json_text(self.outputText) {
            self.outputText = compactJsonText;
        }
    }

    /// Ingests one ordered worker event. The endpoint handles Failed and
    /// Completed before ingest; this path accepts only the output events.
    mutating func ingestEvent(_ streamEvent: ChatGenerationStreamEvent) throws -> Void {
        switch (streamEvent) {
        case let .reasoningFragment(reasoning):
            self.reasoningText += reasoning;
        case let .textFragment(text):
            self.outputText += text;
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            self.functionCalls.append(CollectedFunctionCall(
                toolCallIndex: toolCallIndex,
                functionName: functionName,
                argumentsJson: argumentsJson));
        case .prefillProgress:
            break;
        case .completed:
            throw OpenAiResponsesAssemblyError.unexpectedCompletionEvent;
        case let .failed(failureReason):
            throw OpenAiResponsesAssemblyError.workerFailure(failureReason: failureReason);
        case .streamError:
            throw OpenAiResponsesAssemblyError.workerUnavailable;
        }
    }

    var collectedReasoningText: String {
        return self.reasoningText;
    }

    var collectedOutputText: String {
        return self.outputText;
    }

    /** Assembles the terminal response after the Completed event.

    - Throws: OpenAiResponsesAssemblyError for a cancelled generation or a
      token-usage overflow.
     */
    func intoResponse(
        inputTokenCount: UInt32,
        outputTokenCount: UInt16,
        cachedInputTokenCount: UInt32,
        reasoningTokenCount: UInt16,
        completionReason: ChatGenerationCompletionReason
    ) throws -> OpenAiResponse {
        if completionReason == .cancelled {
            throw OpenAiResponsesAssemblyError.cancelled;
        }
        guard let completedAtUnixSeconds: UInt64 = RestResponsesTimestamp.currentUnixSeconds() else {
            throw OpenAiResponsesAssemblyError.systemClockBeforeUnixEpoch;
        }
        let outputItems: Array<OpenAiResponseOutputItem> = self.outputItems();
        guard let responseUsage: OpenAiResponseUsage = OpenAiResponseUsage.new(
            inputTokens: inputTokenCount,
            outputTokens: UInt32(outputTokenCount),
            cachedTokens: cachedInputTokenCount,
            reasoningTokens: UInt32(reasoningTokenCount)) else {
            throw OpenAiResponsesAssemblyError.tokenUsageOverflow(
                inputTokenCount: inputTokenCount,
                outputTokenCount: outputTokenCount);
        }
        if completionReason == .maximumOutputTokens {
            return OpenAiResponse.incompleteAtOutputTokenLimit(
                responseId: self.responseId,
                createdAt: self.createdAtUnixSeconds,
                modelId: self.modelId,
                instructions: self.instructions,
                output: outputItems,
                usage: responseUsage).withRequestConfiguration(
                    requestConfiguration: self.requestConfiguration);
        }
        return OpenAiResponse.completed(
            responseId: self.responseId,
            createdAt: self.createdAtUnixSeconds,
            completedAt: completedAtUnixSeconds,
            modelId: self.modelId,
            instructions: self.instructions,
            output: outputItems,
            usage: responseUsage).withRequestConfiguration(
                requestConfiguration: self.requestConfiguration);
    }

    /** Assembles the failed response carrying the worker-reported reason. */
    func intoFailedResponse(
        failureReason: ChatGenerationFailureReason
    ) -> OpenAiResponse {
        let failureDetails: (errorCode: String, errorMessage: String) =
            OpenAiResponsesCollector.failureDetails(failureReason);
        return OpenAiResponse.failed(
            responseId: self.responseId,
            createdAt: self.createdAtUnixSeconds,
            modelId: self.modelId,
            instructions: self.instructions,
            output: self.outputItems(),
            errorCode: failureDetails.errorCode,
            errorMessage: failureDetails.errorMessage).withRequestConfiguration(
                requestConfiguration: self.requestConfiguration);
    }

    private func outputItems() -> Array<OpenAiResponseOutputItem> {
        var identifierSuffix: String = self.responseId;
        if identifierSuffix.hasPrefix("resp_") {
            identifierSuffix = String(identifierSuffix.dropFirst("resp_".count));
        }
        var assembledOutputItems: Array<OpenAiResponseOutputItem> = Array();
        if self.reasoningText.isEmpty == false {
            assembledOutputItems.append(OpenAiResponseOutputItem.reasoning(
                id: "rs_\(identifierSuffix)",
                reasoningText: self.reasoningText));
        }
        if self.outputText.isEmpty == false {
            assembledOutputItems.append(OpenAiResponseOutputItem.message(
                id: "msg_\(identifierSuffix)",
                outputText: self.outputText));
        }
        for collectedFunctionCall: CollectedFunctionCall in self.functionCalls {
            assembledOutputItems.append(OpenAiResponseOutputItem.functionCall(
                id: "fc_\(identifierSuffix)-\(collectedFunctionCall.toolCallIndex)",
                callId: "call_\(identifierSuffix)-\(collectedFunctionCall.toolCallIndex)",
                functionName: collectedFunctionCall.functionName,
                argumentsJson: collectedFunctionCall.argumentsJson));
        }
        return assembledOutputItems;
    }

    private static func failureDetails(
        _ failureReason: ChatGenerationFailureReason
    ) -> (errorCode: String, errorMessage: String) {
        if case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens) = failureReason {
            return (
                "context_length_exceeded",
                "requested context uses \(actualTotalContextTokens) tokens, exceeding the "
                    + "\(maximumContextTokens)-token model context window");
        }
        return (
            "response_generation_failed",
            "the local worker could not generate the response: \(failureReason)");
    }
}

/// Shared timestamp helper for the Responses serving path: Unix seconds with
/// an explicit nil for a pre-epoch system clock.
enum RestResponsesTimestamp {

    static func currentUnixSeconds() -> UInt64? {
        let secondsSinceEpoch: TimeInterval = Date().timeIntervalSince1970;
        guard secondsSinceEpoch >= 0 else {
            return nil;
        }
        return UInt64(secondsSinceEpoch);
    }
}
