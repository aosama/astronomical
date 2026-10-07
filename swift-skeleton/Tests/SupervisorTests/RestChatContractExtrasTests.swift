import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Application chat-contract journeys without a dedicated earlier home,
 * migrating the streaming half of
 * apps/supervisor/tests/rest_api/application/contracts.rs: chunks carry a
 * current unix timestamp, internal prefill progress keeps the OpenAI stream
 * open, one large user message streams to completion, tool-call ids never
 * repeat across application instances, secret-bearing tool arguments reach
 * the client unredacted, and the removed legacy text route stays gone.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestChatContractExtrasTests {

    @Test
    func should_timestamp_openai_chat_chunks_with_current_unix_time() throws {
        let routeTable: RestRouteTable = try RestChatContractExtrasTests.routeTable([
            .completed(
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence),
        ]);

        let responseBody: String = try RestChatContractExtrasTests.postChat(
            routeTable: routeTable,
            userMessage: "hello");

        #expect(responseBody.contains("\"created\":0") == false);
    }

    @Test
    func should_keep_the_openai_stream_open_after_internal_prefill_progress() throws {
        let routeTable: RestRouteTable = try RestChatContractExtrasTests.routeTable([
            .prefillProgress(
                processedTokens: 2_048,
                totalTokens: 19_485,
                elapsedMillis: 1_300,
                forwardPrefillChunkElapsedMillis: 1_200,
                completedPrefillChunkTokens: 2_048,
                mlxActiveMemoryBytes: 22_164_699_392,
                mlxAllocatorCacheMemoryBytes: 0,
                mlxPeakMemoryBytes: 24_754_436_684),
            .textFragment("still connected"),
            .completed(
                promptTokenCount: 19_485,
                generatedTokenCount: 2,
                reasoningTokenCount: 0,
                cachedTokenCount: 8_192,
                reason: .endOfSequence),
        ]);

        let responseBody: String = try RestChatContractExtrasTests.postChat(
            routeTable: routeTable,
            userMessage: "hello");

        #expect(responseBody.contains("\"content\":\"still connected\""));
        #expect(responseBody.contains("\"finish_reason\":\"stop\""));
        #expect(responseBody.hasSuffix("data: [DONE]\n\n"));
    }

    @Test
    func should_accept_a_streaming_chat_request_with_one_large_user_message() throws {
        let routeTable: RestRouteTable = try RestChatContractExtrasTests.routeTable([
            .completed(
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence),
        ]);
        let largeUserMessage: String = String(repeating: "x", count: 128 * 1_024);

        let responseBody: String = try RestChatContractExtrasTests.postChat(
            routeTable: routeTable,
            userMessage: largeUserMessage);

        #expect(responseBody.contains("\"finish_reason\":\"stop\""));
        #expect(responseBody.hasSuffix("data: [DONE]\n\n"));
    }

    @Test
    func should_not_reuse_tool_call_ids_across_application_restarts() throws {
        let toolCallStream: Array<ChatGenerationStreamEvent> = [
            .toolCall(
                toolCallIndex: 0,
                functionName: "read",
                argumentsJson: "{\"filePath\":\"README.md\"}"),
        ];
        let firstResponseBody: String = try RestChatContractExtrasTests.postChat(
            routeTable: try RestChatContractExtrasTests.routeTable(toolCallStream),
            userMessage: "hello");
        let secondResponseBody: String = try RestChatContractExtrasTests.postChat(
            routeTable: try RestChatContractExtrasTests.routeTable(toolCallStream),
            userMessage: "hello");

        let firstToolCallId: String = try RestChatContractExtrasTests.extractToolCallId(firstResponseBody);
        let secondToolCallId: String = try RestChatContractExtrasTests.extractToolCallId(secondResponseBody);
        #expect(firstToolCallId != secondToolCallId);
    }

    @Test
    func should_send_tool_call_arguments_with_secret_bearing_lines_unredacted_on_the_wire() throws {
        let argumentsWithSecretLine: String = "{\"command\":\"export api_key=sk-secret-123\\nls -la\"}";
        let routeTable: RestRouteTable = try RestChatContractExtrasTests.routeTable([
            .toolCall(
                toolCallIndex: 0,
                functionName: "bash",
                argumentsJson: argumentsWithSecretLine),
            .completed(
                promptTokenCount: 1,
                generatedTokenCount: 1,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .toolCalls),
        ]);

        let responseBody: String = try RestChatContractExtrasTests.postChat(
            routeTable: routeTable,
            userMessage: "hello");

        #expect(
            responseBody.contains("api_key=sk-secret-123"),
            "the actual secret-bearing command line must reach the client unredacted");
        #expect(
            responseBody.contains("[REDACTED") == false,
            "no redaction marker must leak onto the wire");
    }

    @Test
    func should_not_expose_the_removed_legacy_text_route() throws {
        let routeTable: RestRouteTable = try RestChatContractExtrasTests.routeTable([
            .completed(
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence),
        ]);

        let legacyOutcome: RestRouteOutcome = routeTable.outcome(
            method: "POST",
            path: "/v1/text/generations");

        guard case .notFound = legacyOutcome else {
            Issue.record("the removed legacy text route must stay absent");
            return;
        }
    }

    // MARK: Support

    private static func routeTable(
        _ streamEvents: Array<ChatGenerationStreamEvent>
    ) throws -> RestRouteTable {
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: RestChatJourneySupport.nonStreamingModelId,
                capabilities: RestChatJourneySupport.readyChatCapabilities()),
            streamEvents: streamEvents);
        return try RestChatJourneySupport.chatRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            chatExecutor: scriptedExecutor);
    }

    private static func postChat(
        routeTable: RestRouteTable,
        userMessage: String
    ) throws -> String {
        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\","
                + "\"messages\":[{\"role\":\"user\",\"content\":"
                + (try RestChatContractExtrasTests.encodedJsonText(userMessage))
                + "}],\"stream\":true}");
        #expect(chatResponse.statusCode == 200);
        return String(decoding: chatResponse.bodyBytes, as: UTF8.self);
    }

    private static func encodedJsonText(_ text: String) throws -> String {
        let encodedData: Data = try JSONEncoder().encode(text);
        return String(decoding: encodedData, as: UTF8.self);
    }

    private static func extractToolCallId(_ responseBody: String) throws -> String {
        let idPrefix: String = "\"id\":\"call_";
        guard let idStartRange: Range<String.Index> = responseBody.range(of: idPrefix) else {
            throw RestChatContractExtrasFailure.toolCallIdMissing;
        }
        let idStartIndex: String.Index = responseBody.index(
            idStartRange.upperBound,
            offsetBy: "call_".count);
        guard let idEndIndex: String.Index = responseBody[idStartIndex...].firstIndex(of: "\"") else {
            throw RestChatContractExtrasFailure.toolCallIdUnterminated;
        }
        return String(responseBody[idStartIndex..<idEndIndex]);
    }
}

/// Typed failures of the chat-contract extras journeys.
enum RestChatContractExtrasFailure: Error {

    case toolCallIdMissing;
    case toolCallIdUnterminated;
}
