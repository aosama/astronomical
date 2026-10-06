import Foundation;

import IpcProtocol;
import RestContract;

/// One encoding failure inside the Responses stream encoder.
enum OpenAiResponsesStreamEncodingError: Error, CustomStringConvertible {

    case alreadyCompleted;
    case assembly(OpenAiResponsesAssemblyError);

    var description: String {
        switch (self) {
        case .alreadyCompleted:
            return "the Responses stream has already completed";
        case let .assembly(assemblyError):
            return "failed to assemble the Responses stream: \(assemblyError)";
        }
    }
}

/// Encodes supervisor generation events into semantic Responses events.
///
/// Port of apps/supervisor/src/openai_responses_stream.rs's
/// OpenAiResponsesStreamEncoder. The synchronous Swift serving model has the
/// full ordered event list before the response starts, so encode returns
/// semantic events and the transport serializes them as named SSE frames.
/// Unlike the chat surface there is no `[DONE]` sentinel: the terminal event
/// itself (completed, incomplete, failed, or error) ends the stream.
struct OpenAiResponsesStreamEncoder {

    private let responseId: String;
    private let createdAtUnixSeconds: UInt64;
    private let modelId: String;
    private let instructions: String?;
    private let requestConfiguration: OpenAiResponseRequestConfiguration;
    private var nextSequenceNumber: UInt64;
    private var nextOutputIndex: Int;
    private var reasoningOutputIndex: Int?;
    private var textOutputIndex: Int?;
    private let reasoningExcluded: Bool;
    private var collector: OpenAiResponsesCollector?;

    init(
        responseId: String,
        createdAtUnixSeconds: UInt64,
        modelId: String,
        instructions: String?,
        requestConfiguration: OpenAiResponseRequestConfiguration,
        reasoningExcluded: Bool
    ) {
        self.responseId = responseId;
        self.createdAtUnixSeconds = createdAtUnixSeconds;
        self.modelId = modelId;
        self.instructions = instructions;
        self.requestConfiguration = requestConfiguration;
        self.nextSequenceNumber = 0;
        self.nextOutputIndex = 0;
        self.reasoningOutputIndex = nil;
        self.textOutputIndex = nil;
        self.reasoningExcluded = reasoningExcluded;
        self.collector = OpenAiResponsesCollector(
            responseId: responseId,
            createdAtUnixSeconds: createdAtUnixSeconds,
            modelId: modelId,
            instructions: instructions,
            requestConfiguration: requestConfiguration);
    }

    /** The leading created plus in-progress lifecycle pair. */
    mutating func initialEvents() -> Array<OpenAiResponseStreamEvent> {
        let inProgressResponse: OpenAiResponse = self.inProgressResponse();
        let createdSequenceNumber: UInt64 = self.takeSequenceNumber();
        let inProgressSequenceNumber: UInt64 = self.takeSequenceNumber();
        return [
            .created(sequenceNumber: createdSequenceNumber, response: inProgressResponse),
            .inProgress(sequenceNumber: inProgressSequenceNumber, response: inProgressResponse),
        ];
    }

    /** Encodes one ordered worker event into its semantic stream events. */
    mutating func encode(
        _ streamEvent: ChatGenerationStreamEvent
    ) throws -> Array<OpenAiResponseStreamEvent> {
        switch (streamEvent) {
        case let .completed(
            promptTokenCount, generatedTokenCount, reasoningTokenCount, cachedTokenCount,
            completionReason):
            return try self.completedEvents(
                promptTokenCount: promptTokenCount,
                generatedTokenCount: generatedTokenCount,
                reasoningTokenCount: reasoningTokenCount,
                cachedTokenCount: cachedTokenCount,
                completionReason: completionReason);
        case let .failed(failureReason):
            return [self.workerFailureEvent(failureReason)];
        case .streamError:
            return [self.errorEvent(
                code: "worker_unavailable",
                message: "the local worker became unavailable while generating the response",
                param: nil)];
        case .prefillProgress:
            return Array();
        case let .reasoningFragment(reasoningText):
            if self.reasoningExcluded {
                // The model still thought and the token count still lands in
                // usage; only the client-visible reasoning item is withheld.
                return Array();
            }
            try self.ingestIntoCollector(.reasoningFragment(reasoningText));
            return self.reasoningDeltaEvents(reasoningText);
        case let .textFragment(outputText):
            try self.ingestIntoCollector(.textFragment(outputText));
            return self.outputTextDeltaEvents(outputText);
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            try self.ingestIntoCollector(.toolCall(
                toolCallIndex: toolCallIndex,
                functionName: functionName,
                argumentsJson: argumentsJson));
            return self.functionCallEvents(
                toolCallIndex: toolCallIndex,
                functionName: functionName,
                argumentsJson: argumentsJson);
        }
    }

    /** Whether a terminal event has already been produced. */
    func isTerminal() -> Bool {
        return self.collector == nil;
    }

    private mutating func reasoningDeltaEvents(
        _ reasoningDelta: String
    ) -> Array<OpenAiResponseStreamEvent> {
        var encodedEvents: Array<OpenAiResponseStreamEvent> = Array();
        let resolvedReasoningOutputIndex: Int;
        if let existingReasoningOutputIndex: Int = self.reasoningOutputIndex {
            resolvedReasoningOutputIndex = existingReasoningOutputIndex;
        } else {
            let freshReasoningOutputIndex: Int = self.takeOutputIndex();
            self.reasoningOutputIndex = freshReasoningOutputIndex;
            let itemAddedSequenceNumber: UInt64 = self.takeSequenceNumber();
            encodedEvents.append(.outputItemAdded(
                sequenceNumber: itemAddedSequenceNumber,
                outputIndex: freshReasoningOutputIndex,
                item: OpenAiResponseOutputItem.reasoningInProgress(id: self.reasoningItemId())));
            resolvedReasoningOutputIndex = freshReasoningOutputIndex;
        }
        let deltaSequenceNumber: UInt64 = self.takeSequenceNumber();
        encodedEvents.append(.reasoningSummaryTextDelta(
            sequenceNumber: deltaSequenceNumber,
            itemId: self.reasoningItemId(),
            outputIndex: resolvedReasoningOutputIndex,
            summaryIndex: 0,
            delta: reasoningDelta));
        return encodedEvents;
    }

    private mutating func outputTextDeltaEvents(
        _ outputTextDelta: String
    ) -> Array<OpenAiResponseStreamEvent> {
        var encodedEvents: Array<OpenAiResponseStreamEvent> = self.closeReasoningEvents();
        let resolvedTextOutputIndex: Int;
        if let existingTextOutputIndex: Int = self.textOutputIndex {
            resolvedTextOutputIndex = existingTextOutputIndex;
        } else {
            let freshTextOutputIndex: Int = self.takeOutputIndex();
            self.textOutputIndex = freshTextOutputIndex;
            let itemAddedSequenceNumber: UInt64 = self.takeSequenceNumber();
            encodedEvents.append(.outputItemAdded(
                sequenceNumber: itemAddedSequenceNumber,
                outputIndex: freshTextOutputIndex,
                item: OpenAiResponseOutputItem.messageInProgress(id: self.messageItemId())));
            let partAddedSequenceNumber: UInt64 = self.takeSequenceNumber();
            encodedEvents.append(.contentPartAdded(
                sequenceNumber: partAddedSequenceNumber,
                itemId: self.messageItemId(),
                outputIndex: freshTextOutputIndex,
                contentIndex: 0,
                part: OpenAiResponseOutputContent.outputText(outputText: "")));
            resolvedTextOutputIndex = freshTextOutputIndex;
        }
        let deltaSequenceNumber: UInt64 = self.takeSequenceNumber();
        encodedEvents.append(.outputTextDelta(
            sequenceNumber: deltaSequenceNumber,
            itemId: self.messageItemId(),
            outputIndex: resolvedTextOutputIndex,
            contentIndex: 0,
            delta: outputTextDelta,
            logprobs: Array<JsonWireValue>()));
        return encodedEvents;
    }

    private mutating func functionCallEvents(
        toolCallIndex: UInt16,
        functionName: String,
        argumentsJson: String
    ) -> Array<OpenAiResponseStreamEvent> {
        var encodedEvents: Array<OpenAiResponseStreamEvent> = self.closeReasoningEvents();
        encodedEvents.append(contentsOf: self.closeOutputTextEvents());
        let outputIndex: Int = self.takeOutputIndex();
        let functionItemId: String = self.functionItemId(toolCallIndex);
        let functionCallId: String = self.functionCallId(toolCallIndex);
        encodedEvents.append(.outputItemAdded(
            sequenceNumber: self.takeSequenceNumber(),
            outputIndex: outputIndex,
            item: OpenAiResponseOutputItem.functionCallInProgress(
                id: functionItemId,
                callId: functionCallId,
                functionName: functionName)));
        encodedEvents.append(.functionCallArgumentsDelta(
            sequenceNumber: self.takeSequenceNumber(),
            itemId: functionItemId,
            outputIndex: outputIndex,
            delta: argumentsJson));
        encodedEvents.append(.functionCallArgumentsDone(
            sequenceNumber: self.takeSequenceNumber(),
            itemId: functionItemId,
            outputIndex: outputIndex,
            name: functionName,
            arguments: argumentsJson));
        encodedEvents.append(.outputItemDone(
            sequenceNumber: self.takeSequenceNumber(),
            outputIndex: outputIndex,
            item: OpenAiResponseOutputItem.functionCall(
                id: functionItemId,
                callId: functionCallId,
                functionName: functionName,
                argumentsJson: argumentsJson)));
        return encodedEvents;
    }

    private mutating func completedEvents(
        promptTokenCount: UInt32,
        generatedTokenCount: UInt16,
        reasoningTokenCount: UInt16,
        cachedTokenCount: UInt32,
        completionReason: ChatGenerationCompletionReason
    ) throws -> Array<OpenAiResponseStreamEvent> {
        var encodedEvents: Array<OpenAiResponseStreamEvent> = self.closeReasoningEvents();
        encodedEvents.append(contentsOf: self.closeOutputTextEvents());
        guard let terminalCollector: OpenAiResponsesCollector = self.collector else {
            throw OpenAiResponsesStreamEncodingError.alreadyCompleted;
        }
        self.collector = nil;
        let terminalResponse: OpenAiResponse = try terminalCollector.intoResponse(
            inputTokenCount: promptTokenCount,
            outputTokenCount: generatedTokenCount,
            cachedInputTokenCount: cachedTokenCount,
            reasoningTokenCount: reasoningTokenCount,
            completionReason: completionReason);
        let terminalSequenceNumber: UInt64 = self.takeSequenceNumber();
        if completionReason == .maximumOutputTokens {
            encodedEvents.append(.incomplete(
                sequenceNumber: terminalSequenceNumber, response: terminalResponse));
        } else {
            encodedEvents.append(.completed(
                sequenceNumber: terminalSequenceNumber, response: terminalResponse));
        }
        return encodedEvents;
    }

    private mutating func closeReasoningEvents() -> Array<OpenAiResponseStreamEvent> {
        guard let closedReasoningOutputIndex: Int = self.reasoningOutputIndex else {
            return Array();
        }
        self.reasoningOutputIndex = nil;
        let reasoningItemId: String = self.reasoningItemId();
        let closedReasoningText: String = self.collector?.collectedReasoningText ?? "";
        return [
            .reasoningSummaryTextDone(
                sequenceNumber: self.takeSequenceNumber(),
                itemId: reasoningItemId,
                outputIndex: closedReasoningOutputIndex,
                summaryIndex: 0,
                text: closedReasoningText),
            .outputItemDone(
                sequenceNumber: self.takeSequenceNumber(),
                outputIndex: closedReasoningOutputIndex,
                item: OpenAiResponseOutputItem.reasoning(
                    id: reasoningItemId, reasoningText: closedReasoningText)),
        ];
    }

    private mutating func closeOutputTextEvents() -> Array<OpenAiResponseStreamEvent> {
        guard let closedTextOutputIndex: Int = self.textOutputIndex else {
            return Array();
        }
        self.textOutputIndex = nil;
        let messageItemId: String = self.messageItemId();
        let closedOutputText: String = self.collector?.collectedOutputText ?? "";
        return [
            .outputTextDone(
                sequenceNumber: self.takeSequenceNumber(),
                itemId: messageItemId,
                outputIndex: closedTextOutputIndex,
                contentIndex: 0,
                text: closedOutputText,
                logprobs: Array<JsonWireValue>()),
            .contentPartDone(
                sequenceNumber: self.takeSequenceNumber(),
                itemId: messageItemId,
                outputIndex: closedTextOutputIndex,
                contentIndex: 0,
                part: OpenAiResponseOutputContent.outputText(outputText: closedOutputText)),
            .outputItemDone(
                sequenceNumber: self.takeSequenceNumber(),
                outputIndex: closedTextOutputIndex,
                item: OpenAiResponseOutputItem.message(
                    id: messageItemId, outputText: closedOutputText)),
        ];
    }

    private mutating func workerFailureEvent(
        _ failureReason: ChatGenerationFailureReason
    ) -> OpenAiResponseStreamEvent {
        guard let failedCollector: OpenAiResponsesCollector = self.collector else {
            return self.errorEvent(
                code: "response_generation_failed",
                message: "the Responses stream had already terminated",
                param: nil);
        }
        self.collector = nil;
        return .failed(
            sequenceNumber: self.takeSequenceNumber(),
            response: failedCollector.intoFailedResponse(failureReason: failureReason));
    }

    private mutating func errorEvent(
        code: String,
        message: String,
        param: String?
    ) -> OpenAiResponseStreamEvent {
        self.collector = nil;
        return .error(
            sequenceNumber: self.takeSequenceNumber(),
            code: code,
            message: message,
            param: param);
    }

    private mutating func ingestIntoCollector(
        _ streamEvent: ChatGenerationStreamEvent
    ) throws -> Void {
        guard self.collector != nil else {
            throw OpenAiResponsesStreamEncodingError.alreadyCompleted;
        }
        try self.collector?.ingestEvent(streamEvent);
    }

    private func inProgressResponse() -> OpenAiResponse {
        return OpenAiResponse.inProgress(
            responseId: self.responseId,
            createdAt: self.createdAtUnixSeconds,
            modelId: self.modelId,
            instructions: self.instructions).withRequestConfiguration(
                requestConfiguration: self.requestConfiguration);
    }

    private mutating func takeSequenceNumber() -> UInt64 {
        let sequenceNumber: UInt64 = self.nextSequenceNumber;
        let (advancedSequenceNumber, didOverflow) =
            self.nextSequenceNumber.addingReportingOverflow(1);
        self.nextSequenceNumber = didOverflow ? UInt64.max : advancedSequenceNumber;
        return sequenceNumber;
    }

    private mutating func takeOutputIndex() -> Int {
        let outputIndex: Int = self.nextOutputIndex;
        self.nextOutputIndex += 1;
        return outputIndex;
    }

    private func identifierSuffix() -> String {
        if self.responseId.hasPrefix("resp_") {
            return String(self.responseId.dropFirst("resp_".count));
        }
        return self.responseId;
    }

    private func reasoningItemId() -> String {
        return "rs_\(self.identifierSuffix())";
    }

    private func messageItemId() -> String {
        return "msg_\(self.identifierSuffix())";
    }

    private func functionItemId(_ toolCallIndex: UInt16) -> String {
        return "fc_\(self.identifierSuffix())-\(toolCallIndex)";
    }

    private func functionCallId(_ toolCallIndex: UInt16) -> String {
        return "call_\(self.identifierSuffix())-\(toolCallIndex)";
    }
}
