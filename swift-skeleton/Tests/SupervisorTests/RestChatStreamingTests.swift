import Foundation;

import Testing;

import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Acceptance journeys for the streaming POST /v1/chat/completions surface:
 * an OpenAI-compatible SSE frame sequence that starts with the assistant
 * role chunk, carries text/reasoning/tool-call deltas, terminates with the
 * finish chunk plus `data: [DONE]`, attaches usage when requested, withholds
 * excluded reasoning deltas, and answers worker failures with an in-stream
 * error frame. The executor is scripted, so no worker process runs.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestChatStreamingTests {

    @Test
    func should_stream_openai_compatible_text_lifecycle_for_opencode() throws {
        let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(streamEvents: [
            .textFragment("Visible title"),
            RestChatStreamingTests.completedEvent(.endOfSequence),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true,\"stream_options\":{\"include_usage\":true}}");

        #expect(chatResponse.statusCode == 200);
        #expect(chatResponse.contentType.hasPrefix("text/event-stream"));
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(
                RestChatJourneySupport.responseText(chatResponse));
        #expect(parsedStream.sawDone,
            "OpenAI-compatible chat streams must end with data: [DONE]");
        #expect(parsedStream.deltaText(forKey: "content") == "Visible title");
        #expect(parsedStream.deltaText(forKey: "reasoning_content") == "");
        #expect(parsedStream.finishReason() == "stop");
        #expect(parsedStream.payloads.contains { (payload: [String: Any]) -> Bool in
            return (RestChatStreamingTests.deltaObject(payload)?["role"] as? String) == "assistant";
        }, "chat stream should start with an assistant role chunk");
        let terminalPayload: [String: Any] = try #require(
            parsedStream.terminalPayload(finishReason: "stop"));
        let usage: [String: Any] = try #require(terminalPayload["usage"] as? [String: Any]);
        #expect(usage["prompt_tokens"] as? UInt64 == 3);
        #expect(usage["completion_tokens"] as? UInt64 == 2);
    }

    @Test
    func should_not_expose_reasoning_only_stream_as_opencode_visible_text() throws {
        let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(streamEvents: [
            .reasoningFragment("thinking only"),
            RestChatStreamingTests.completedEvent(.endOfSequence),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 200);
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(
                RestChatJourneySupport.responseText(chatResponse));
        #expect(parsedStream.sawDone, "stream should terminate");
        #expect(parsedStream.deltaText(forKey: "reasoning_content") == "thinking only");
        #expect(parsedStream.deltaText(forKey: "content") == "",
            "reasoning_content must not be duplicated into delta.content because OpenCode uses content as visible text/title");
        #expect(parsedStream.finishReason() == "stop");
    }

    @Test
    func should_expose_only_text_fragment_as_visible_content_when_reasoning_precedes_text() throws {
        let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(streamEvents: [
            .reasoningFragment("internal thought"),
            .textFragment("Visible title"),
            RestChatStreamingTests.completedEvent(.endOfSequence),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 200);
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(
                RestChatJourneySupport.responseText(chatResponse));
        #expect(parsedStream.deltaText(forKey: "reasoning_content") == "internal thought");
        #expect(parsedStream.deltaText(forKey: "content") == "Visible title",
            "OpenCode title generation should see only assistant text deltas, not prior reasoning");
        #expect(parsedStream.finishReason() == "stop");
    }

    @Test
    func should_withhold_reasoning_deltas_from_the_stream_when_excluded() throws {
        let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(streamEvents: [
            .reasoningFragment("internal thought"),
            .textFragment("Visible title"),
            RestChatStreamingTests.completedEvent(.endOfSequence),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true,\"reasoning\":{\"max_tokens\":1024,\"exclude\":true}}");

        #expect(chatResponse.statusCode == 200);
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(
                RestChatJourneySupport.responseText(chatResponse));
        #expect(parsedStream.deltaText(forKey: "reasoning_content") == "",
            "excluded reasoning must not emit reasoning deltas");
        #expect(parsedStream.deltaText(forKey: "content") == "Visible title",
            "the visible answer must still stream when reasoning is excluded");
        #expect(parsedStream.finishReason() == "stop");
    }

    @Test
    func should_stream_parallel_tool_calls_in_openai_compatible_chunks_for_opencode() throws {
        let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(streamEvents: [
            .toolCall(
                toolCallIndex: 0,
                functionName: "read",
                argumentsJson: "{\"filePath\":\"README.md\"}"),
            .toolCall(
                toolCallIndex: 1,
                functionName: "glob",
                argumentsJson: "{\"pattern\":\"tests/**/*.rs\"}"),
            RestChatStreamingTests.completedEvent(.toolCalls),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 200);
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(
                RestChatJourneySupport.responseText(chatResponse));
        let toolCallDeltas: Array<[String: Any]> = parsedStream.toolCallDeltas();
        #expect(toolCallDeltas.count == 2);
        let firstToolCallDelta: [String: Any] = toolCallDeltas[0];
        let secondToolCallDelta: [String: Any] = toolCallDeltas[1];

        #expect(firstToolCallDelta["index"] as? UInt64 == 0);
        let firstToolCallId: String = try #require(firstToolCallDelta["id"] as? String);
        #expect(firstToolCallId.hasPrefix("call_"),
            "tool call ids should be stable OpenAI-style call ids");
        #expect(firstToolCallDelta["type"] as? String == "function");
        let firstFunction: [String: Any] = try #require(firstToolCallDelta["function"] as? [String: Any]);
        #expect(firstFunction["name"] as? String == "read");
        #expect(firstFunction["arguments"] as? String == "{\"filePath\":\"README.md\"}");
        #expect(secondToolCallDelta["index"] as? UInt64 == 1);
        let secondFunction: [String: Any] = try #require(secondToolCallDelta["function"] as? [String: Any]);
        #expect(secondFunction["name"] as? String == "glob");
        #expect(secondFunction["arguments"] as? String == "{\"pattern\":\"tests/**/*.rs\"}");
        #expect(parsedStream.finishReason() == "tool_calls");
        #expect(parsedStream.sawDone, "stream should terminate");
    }

    @Test
    func should_answer_a_failed_stream_with_an_error_frame_and_no_done_terminator() throws {
        let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(streamEvents: [
            .failed(reason: .engineBusy),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        // The stream starts (200) and the failure arrives as an in-band frame.
        #expect(chatResponse.statusCode == 200);
        let responseText: String = RestChatJourneySupport.responseText(chatResponse);
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(responseText);
        #expect(parsedStream.sawDone == false,
            "a failed stream ends without the [DONE] terminator");
        let errorPayload: [String: Any] = try #require(
            parsedStream.payloads.first { (payload: [String: Any]) -> Bool in
                return payload["error"] != nil;
            });
        let errorEnvelope: [String: Any] = try RestChatJourneySupport.requireErrorObject(errorPayload);
        #expect(errorEnvelope["code"] as? String == "chat_engine_busy");
        #expect(errorEnvelope["message"] as? String
            == "the local inference engine is already processing another request");
    }

    /// Every stream failure answers with its own meaningful code and message,
    /// and no failure ever carries the [DONE] terminator.
    @Test
    func should_emit_meaningful_openai_errors_for_each_stream_failure() throws {
        let failureCases: Array<(streamEvent: ChatGenerationStreamEvent, expectedCode: String, expectedMessage: String)> =
            [
                (
                    streamEvent: .failed(reason: .contextLengthExceeded(
                        actualTotalContextTokens: 262_145,
                        maximumContextTokens: 262_144)),
                    expectedCode: "context_length_exceeded",
                    expectedMessage: "requested context uses 262145 tokens, exceeding the "
                        + "262144-token model context window"),
                (
                    streamEvent: .failed(reason: .invalidRequest(
                        reason: "rendered prompt exceeds the 262144-byte worker limit")),
                    expectedCode: "chat_invalid_request",
                    expectedMessage: "the local worker rejected the chat request: "
                        + "rendered prompt exceeds the 262144-byte worker limit"),
                (
                    streamEvent: .failed(reason: .engineBusy),
                    expectedCode: "chat_engine_busy",
                    expectedMessage: "the local inference engine is already processing another request"),
                (
                    streamEvent: .failed(reason: .malformedModelOutput),
                    expectedCode: "chat_malformed_model_output",
                    expectedMessage: "the model produced malformed structured output"),
                (
                    streamEvent: .failed(reason: .fatalExecution(
                        reason: "GPU allocation exceeded the platform buffer limit while evaluating the model; "
                            + "reduce the prompt size or configured prefill chunk size")),
                    expectedCode: "chat_worker_unavailable",
                    expectedMessage: "the local worker stopped after a fatal model execution error: "
                        + "GPU allocation exceeded the platform buffer limit while evaluating the model; "
                        + "reduce the prompt size or configured prefill chunk size"),
                (
                    streamEvent: .streamError(.workerUnavailable),
                    expectedCode: "chat_worker_unavailable",
                    expectedMessage: "the local worker became unavailable while processing the chat request"),
            ];

        for failureCase in failureCases {
            let routeTable: RestRouteTable = try RestChatStreamingTests.routeTable(
                streamEvents: [failureCase.streamEvent]);
            let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: routeTable,
                requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

            #expect(chatResponse.statusCode == 200);
            let responseText: String = RestChatJourneySupport.responseText(chatResponse);
            #expect(
                responseText.contains("\"code\":\"\(failureCase.expectedCode)\""),
                "each stream failure must carry its own error code");
            #expect(
                responseText.contains("\"message\":\"\(failureCase.expectedMessage)\""),
                "each stream failure must carry its human-readable message");
            #expect(
                responseText.contains("[DONE]") == false,
                "a failed stream must never end with the done terminator");
        }
    }

    // MARK: Shared fixture helpers

    private static func routeTable(
        streamEvents: Array<ChatGenerationStreamEvent>
    ) throws -> RestRouteTable {
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: RestChatJourneySupport.streamingModelId,
                capabilities: RestChatJourneySupport.readyChatCapabilities()),
            streamEvents: streamEvents);
        return try RestChatJourneySupport.chatRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            chatExecutor: scriptedExecutor);
    }

    private static func completedEvent(
        _ completionReason: ChatGenerationCompletionReason
    ) -> ChatGenerationStreamEvent {
        return .completed(
            promptTokenCount: 3,
            generatedTokenCount: 2,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            reason: completionReason);
    }

    private static func deltaObject(_ payload: [String: Any]) -> [String: Any]? {
        guard let choices: Array<Any> = payload["choices"] as? Array<Any>,
              choices.count > 0,
              let choiceObject: [String: Any] = choices[0] as? [String: Any] else {
            return nil;
        }
        return choiceObject["delta"] as? [String: Any];
    }
}
