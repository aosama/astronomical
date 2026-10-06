import Foundation;

import Testing;

import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Collector journeys for the Responses assembly: ordered reasoning, text,
 * and tool-call events assemble into one terminal Responses object with
 * derived item ids, and the generated-token ceiling marks the response and
 * its items incomplete.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class OpenAiResponsesCollectorTests {

    @Test
    func should_collect_reasoning_summary_text_and_function_calls_into_one_response() throws {
        var collector: OpenAiResponsesCollector = OpenAiResponsesCollector(
            responseId: "resp_instance-9",
            createdAtUnixSeconds: 1_753_000_000,
            modelId: "astronomical/fake-mixture-of-experts",
            instructions: "Be precise.",
            requestConfiguration: OpenAiResponseRequestConfiguration());
        try collector.ingestEvent(.reasoningFragment("Inspect first."));
        try collector.ingestEvent(.prefillProgress(
            processedTokens: 2_048,
            totalTokens: 4_096,
            elapsedMillis: 1_000,
            forwardPrefillChunkElapsedMillis: 900,
            completedPrefillChunkTokens: 2_048,
            mlxActiveMemoryBytes: 20_000,
            mlxAllocatorCacheMemoryBytes: 0,
            mlxPeakMemoryBytes: 22_000));
        try collector.ingestEvent(.textFragment("Done."));
        try collector.ingestEvent(.toolCall(
            toolCallIndex: 0, functionName: "read",
            argumentsJson: "{\"filePath\":\"README.md\"}"));

        let assembledResponse: OpenAiResponse = try collector.intoResponse(
            inputTokenCount: 100,
            outputTokenCount: 20,
            cachedInputTokenCount: 64,
            reasoningTokenCount: 7,
            completionReason: .endOfSequence);
        let responseDocument: [String: Any] = try Self.jsonObject(assembledResponse.wireValue());

        let outputArray: Array<Any> = try Self.requireArray(responseDocument["output"]);
        let reasoningItem: [String: Any] = try Self.requireObject(outputArray.first);
        #expect(reasoningItem["type"] as? String == "reasoning");
        let summaryArray: Array<Any> = try Self.requireArray(reasoningItem["summary"]);
        let summaryEntry: [String: Any] = try Self.requireObject(summaryArray.first);
        #expect(summaryEntry["type"] as? String == "summary_text");
        #expect(summaryEntry["text"] as? String == "Inspect first.");
        let reasoningContent: Array<Any> = try Self.requireArray(reasoningItem["content"]);
        #expect(reasoningContent.isEmpty);
        #expect(reasoningItem["encrypted_content"] == nil
            || reasoningItem["encrypted_content"] is NSNull);
        let messageItem: [String: Any] = try Self.requireObject(outputArray[1]);
        #expect(messageItem["type"] as? String == "message");
        let functionCallItem: [String: Any] = try Self.requireObject(outputArray[2]);
        #expect(functionCallItem["type"] as? String == "function_call");
        #expect(responseDocument["output_text"] as? String == "Done.");
        let completedAt: Double = try Self.requireNumber(responseDocument["completed_at"]);
        let createdAt: Double = try Self.requireNumber(responseDocument["created_at"]);
        #expect(completedAt > createdAt);
        let usageObject: [String: Any] = try Self.requireObject(responseDocument["usage"]);
        let outputDetails: [String: Any] = try Self.requireObject(usageObject["output_tokens_details"]);
        #expect(outputDetails["reasoning_tokens"] as? Int == 7);
        #expect(usageObject["total_tokens"] as? Int == 120);
    }

    @Test
    func should_mark_a_maximum_output_token_response_incomplete() throws {
        var collector: OpenAiResponsesCollector = OpenAiResponsesCollector(
            responseId: "resp_instance-10",
            createdAtUnixSeconds: 1_753_000_000,
            modelId: "astronomical/fake-mixture-of-experts",
            instructions: nil,
            requestConfiguration: OpenAiResponseRequestConfiguration());
        try collector.ingestEvent(.textFragment("Partial answer"));

        let incompleteResponse: OpenAiResponse = try collector.intoResponse(
            inputTokenCount: 100,
            outputTokenCount: 20,
            cachedInputTokenCount: 0,
            reasoningTokenCount: 0,
            completionReason: .maximumOutputTokens);
        let responseDocument: [String: Any] = try Self.jsonObject(incompleteResponse.wireValue());

        #expect(responseDocument["status"] as? String == "incomplete");
        let incompleteDetails: [String: Any] = try Self.requireObject(responseDocument["incomplete_details"]);
        #expect(incompleteDetails["reason"] as? String == "max_output_tokens");
        let outputArray: Array<Any> = try Self.requireArray(responseDocument["output"]);
        let firstOutputItem: [String: Any] = try Self.requireObject(outputArray.first);
        #expect(firstOutputItem["status"] as? String == "incomplete");
    }

    // MARK: JSON helpers

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

    private static func requireNumber(_ anyValue: Any?) throws -> Double {
        guard let numberValue: Double = anyValue as? Double else {
            throw OpenAiResponsesTestFailure.expectedNumber;
        }
        return numberValue;
    }
}

/// Typed failures for the Responses contract unit journeys.
enum OpenAiResponsesTestFailure: Error {
    case expectedObject;
    case expectedArray;
    case expectedNumber;
    case expectedString;
}
