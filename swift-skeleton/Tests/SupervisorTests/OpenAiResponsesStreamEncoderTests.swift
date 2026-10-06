import Foundation;

import Testing;

import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Stream-encoder journeys for the Responses surface: the exact semantic
 * event lifecycle and monotonic sequence numbers, reasoning streamed as
 * summary text without encrypted content, the incomplete and failed
 * terminals, reasoning exclusion, and the absence of any response_id field.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class OpenAiResponsesStreamEncoderTests {

    @Test
    func should_emit_the_exact_fragmented_text_lifecycle_and_skip_prefill_telemetry() throws {
        var encoder: OpenAiResponsesStreamEncoder = Self.newEncoder(
            responseId: "resp_instance-11", reasoningExcluded: false);
        var encodedEvents: Array<OpenAiResponseStreamEvent> = encoder.initialEvents();
        encodedEvents.append(contentsOf: try encoder.encode(.prefillProgress(
            processedTokens: 2_048,
            totalTokens: 4_096,
            elapsedMillis: 1_000,
            forwardPrefillChunkElapsedMillis: 900,
            completedPrefillChunkTokens: 2_048,
            mlxActiveMemoryBytes: 20_000,
            mlxAllocatorCacheMemoryBytes: 0,
            mlxPeakMemoryBytes: 22_000)));
        encodedEvents.append(contentsOf: try encoder.encode(.textFragment("Hel")));
        encodedEvents.append(contentsOf: try encoder.encode(.textFragment("lo")));
        encodedEvents.append(contentsOf: try encoder.encode(.completed(
            promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 0,
            cachedTokenCount: 0, reason: .endOfSequence)));

        #expect(encodedEvents.map({ (encodedEvent: OpenAiResponseStreamEvent) -> String in
            return encodedEvent.eventType();
        }) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]);
        #expect(encodedEvents.map({ (encodedEvent: OpenAiResponseStreamEvent) -> UInt64 in
            return encodedEvent.sequenceNumber();
        }) == Array<UInt64>(0..<10));
        var serializedEvents: String = String();
        for encodedEvent: OpenAiResponseStreamEvent in encodedEvents {
            serializedEvents += try encodedEvent.wireValue().serializedText;
            serializedEvents += "\n";
        }
        #expect(serializedEvents.contains("\"delta\":\"Hel\""));
        #expect(serializedEvents.contains("\"delta\":\"lo\""));
        #expect(serializedEvents.contains("[DONE]") == false);
        try Self.assertNoResponseIdFields(encodedEvents);
    }

    @Test
    func should_stream_raw_reasoning_as_summary_text_without_encrypted_content() throws {
        var encoder: OpenAiResponsesStreamEncoder = Self.newEncoder(
            responseId: "resp_instance-12", reasoningExcluded: false);
        var encodedEvents: Array<OpenAiResponseStreamEvent> = encoder.initialEvents();
        encodedEvents.append(contentsOf: try encoder.encode(.reasoningFragment("Inspect first.")));
        encodedEvents.append(contentsOf: try encoder.encode(.toolCall(
            toolCallIndex: 0, functionName: "read",
            argumentsJson: "{\"filePath\":\"README.md\"}")));
        encodedEvents.append(contentsOf: try encoder.encode(.completed(
            promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 2,
            cachedTokenCount: 0, reason: .toolCalls)));

        #expect(encodedEvents.map({ (encodedEvent: OpenAiResponseStreamEvent) -> String in
            return encodedEvent.eventType();
        }) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]);
        let eventDocuments: Array<[String: Any]> = try encodedEvents.map({
            (encodedEvent: OpenAiResponseStreamEvent) -> [String: Any] in
            return try Self.jsonObject(encodedEvent.wireValue());
        });
        #expect(eventDocuments[3]["summary_index"] as? Int == 0);
        #expect(eventDocuments[3]["delta"] as? String == "Inspect first.");
        #expect(eventDocuments[4]["summary_index"] as? Int == 0);
        #expect(eventDocuments[4]["text"] as? String == "Inspect first.");
        let doneItem: [String: Any] = try Self.requireObject(eventDocuments[5]["item"]);
        let doneSummary: Array<Any> = try Self.requireArray(doneItem["summary"]);
        let doneSummaryEntry: [String: Any] = try Self.requireObject(doneSummary.first);
        #expect(doneSummaryEntry["type"] as? String == "summary_text");
        #expect(doneSummaryEntry["text"] as? String == "Inspect first.");
        var allSerialized: String = String();
        for encodedEvent: OpenAiResponseStreamEvent in encodedEvents {
            allSerialized += try encodedEvent.wireValue().serializedText;
        }
        #expect(allSerialized.contains("encrypted_content") == false);
        #expect(eventDocuments[7]["delta"] as? String == "{\"filePath\":\"README.md\"}");
        #expect(eventDocuments[8]["name"] as? String == "read");
        let terminalResponse: [String: Any] = try Self.requireObject(eventDocuments[10]["response"]);
        #expect(terminalResponse["output_text"] as? String == "");
        try Self.assertNoResponseIdFields(encodedEvents);
    }

    @Test
    func should_close_open_text_before_emitting_an_incomplete_terminal_event() throws {
        var encoder: OpenAiResponsesStreamEncoder = Self.newEncoder(
            responseId: "resp_instance-13", reasoningExcluded: false);
        var encodedEvents: Array<OpenAiResponseStreamEvent> = encoder.initialEvents();
        encodedEvents.append(contentsOf: try encoder.encode(.textFragment("Partial")));
        encodedEvents.append(contentsOf: try encoder.encode(.completed(
            promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 0,
            cachedTokenCount: 0, reason: .maximumOutputTokens)));

        guard let terminalEvent: OpenAiResponseStreamEvent = encodedEvents.last else {
            throw OpenAiResponsesTestFailure.expectedObject;
        }
        let terminalDocument: [String: Any] = try Self.jsonObject(terminalEvent.wireValue());
        #expect(terminalDocument["type"] as? String == "response.incomplete");
        let terminalResponse: [String: Any] = try Self.requireObject(terminalDocument["response"]);
        #expect(terminalResponse["status"] as? String == "incomplete");
        let incompleteDetails: [String: Any] = try Self.requireObject(terminalResponse["incomplete_details"]);
        #expect(incompleteDetails["reason"] as? String == "max_output_tokens");
        #expect(encoder.isTerminal());
    }

    @Test
    func should_emit_a_terminal_failed_response_for_a_worker_reported_context_failure() throws {
        var encoder: OpenAiResponsesStreamEncoder = Self.newEncoder(
            responseId: "resp_instance-14", reasoningExcluded: false);
        var encodedEvents: Array<OpenAiResponseStreamEvent> = encoder.initialEvents();
        encodedEvents.append(contentsOf: try encoder.encode(.failed(
            reason: .contextLengthExceeded(
                actualTotalContextTokens: 262_145, maximumContextTokens: 262_144))));

        guard let terminalEvent: OpenAiResponseStreamEvent = encodedEvents.last else {
            throw OpenAiResponsesTestFailure.expectedObject;
        }
        let terminalDocument: [String: Any] = try Self.jsonObject(terminalEvent.wireValue());
        #expect(terminalDocument["type"] as? String == "response.failed");
        let terminalResponse: [String: Any] = try Self.requireObject(terminalDocument["response"]);
        #expect(terminalResponse["status"] as? String == "failed");
        let terminalError: [String: Any] = try Self.requireObject(terminalResponse["error"]);
        #expect(terminalError["code"] as? String == "context_length_exceeded");
        #expect(encoder.isTerminal());
    }

    @Test
    func should_mark_partially_emitted_output_incomplete_when_the_worker_fails() throws {
        var encoder: OpenAiResponsesStreamEncoder = Self.newEncoder(
            responseId: "resp_instance-15", reasoningExcluded: false);
        var encodedEvents: Array<OpenAiResponseStreamEvent> = encoder.initialEvents();
        encodedEvents.append(contentsOf: try encoder.encode(.textFragment("Partial")));
        encodedEvents.append(contentsOf: try encoder.encode(.failed(
            reason: .invalidRequest(reason: "the tool result was invalid"))));

        guard let terminalEvent: OpenAiResponseStreamEvent = encodedEvents.last else {
            throw OpenAiResponsesTestFailure.expectedObject;
        }
        let terminalDocument: [String: Any] = try Self.jsonObject(terminalEvent.wireValue());
        #expect(terminalDocument["type"] as? String == "response.failed");
        let terminalResponse: [String: Any] = try Self.requireObject(terminalDocument["response"]);
        let outputArray: Array<Any> = try Self.requireArray(terminalResponse["output"]);
        let firstOutputItem: [String: Any] = try Self.requireObject(outputArray.first);
        #expect(firstOutputItem["status"] as? String == "incomplete");
        let terminalError: [String: Any] = try Self.requireObject(terminalResponse["error"]);
        #expect(terminalError["code"] as? String == "response_generation_failed");
    }

    @Test
    func should_withhold_reasoning_items_but_keep_reasoning_usage_when_excluded() throws {
        var encoder: OpenAiResponsesStreamEncoder = Self.newEncoder(
            responseId: "resp_instance-16", reasoningExcluded: true);
        var encodedEvents: Array<OpenAiResponseStreamEvent> = encoder.initialEvents();
        encodedEvents.append(contentsOf: try encoder.encode(.reasoningFragment("internal thought")));
        encodedEvents.append(contentsOf: try encoder.encode(.textFragment("Visible answer")));
        encodedEvents.append(contentsOf: try encoder.encode(.completed(
            promptTokenCount: 10, generatedTokenCount: 7, reasoningTokenCount: 5,
            cachedTokenCount: 0, reason: .endOfSequence)));

        var allSerialized: String = String();
        for encodedEvent: OpenAiResponseStreamEvent in encodedEvents {
            allSerialized += try encodedEvent.wireValue().serializedText;
        }
        #expect(allSerialized.contains("reasoning_summary_text") == false);
        #expect(allSerialized.contains("internal thought") == false);
        guard let terminalDocumentData: OpenAiResponseStreamEvent = encodedEvents.last else {
            throw OpenAiResponsesTestFailure.expectedObject;
        }
        let terminalDocument: [String: Any] = try Self.jsonObject(terminalDocumentData.wireValue());
        #expect(terminalDocument["type"] as? String == "response.completed");
        let terminalResponse: [String: Any] = try Self.requireObject(terminalDocument["response"]);
        let usageObject: [String: Any] = try Self.requireObject(terminalResponse["usage"]);
        let outputDetails: [String: Any] = try Self.requireObject(usageObject["output_tokens_details"]);
        #expect(outputDetails["reasoning_tokens"] as? Int == 5);
        let outputArray: Array<Any> = try Self.requireArray(terminalResponse["output"]);
        var outputItemTypes: Array<String> = Array();
        for outputItem: Any in outputArray {
            let outputItemObject: [String: Any] = try Self.requireObject(outputItem);
            outputItemTypes.append(try Self.requireString(outputItemObject["type"]));
        }
        #expect(outputItemTypes == ["message"]);
        try Self.assertNoResponseIdFields(encodedEvents);
    }

    // MARK: Journey plumbing

    private static func newEncoder(
        responseId: String,
        reasoningExcluded: Bool
    ) -> OpenAiResponsesStreamEncoder {
        return OpenAiResponsesStreamEncoder(
            responseId: responseId,
            createdAtUnixSeconds: 1_753_000_000,
            modelId: "astronomical/fake-mixture-of-experts",
            instructions: nil,
            requestConfiguration: OpenAiResponseRequestConfiguration(),
            reasoningExcluded: reasoningExcluded);
    }

    private static func assertNoResponseIdFields(
        _ encodedEvents: Array<OpenAiResponseStreamEvent>
    ) throws -> Void {
        for encodedEvent: OpenAiResponseStreamEvent in encodedEvents {
            let eventDocument: [String: Any] = try Self.jsonObject(encodedEvent.wireValue());
            #expect(eventDocument["response_id"] == nil, """
                \(encodedEvent.eventType()) must not include response_id
                """);
        }
    }

    private static func jsonObject(_ wireValue: JsonWireValue) throws -> [String: Any] {
        let serializedText: String = try wireValue.serializedText;
        let decodedJson: Any = try JSONSerialization.jsonObject(
            with: Data(serializedText.utf8));
        guard let objectValue: [String: Any] = decodedJson as? [String: Any] else {
            throw OpenAiResponsesTestFailure.expectedObject;
        }
        return objectValue;
    }

    private static func requireObject(_ anyValue: Any?) throws -> [String: Any] {
        guard let objectValue: [String: Any] = anyValue as? [String: Any] else {
            throw OpenAiResponsesTestFailure.expectedObject;
        }
        return objectValue;
    }

    private static func requireArray(_ anyValue: Any?) throws -> Array<Any> {
        guard let arrayValue: Array<Any> = anyValue as? Array<Any> else {
            throw OpenAiResponsesTestFailure.expectedArray;
        }
        return arrayValue;
    }

    private static func requireString(_ anyValue: Any?) throws -> String {
        guard let stringValue: String = anyValue as? String else {
            throw OpenAiResponsesTestFailure.expectedString;
        }
        return stringValue;
    }
}
