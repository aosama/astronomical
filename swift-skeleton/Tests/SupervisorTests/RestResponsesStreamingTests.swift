import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// REST Responses streaming journeys: the semantic SSE frame sequence, the
/// reasoning/text split, the function-call lifecycle, and terminal failure
/// frames. Responses streams carry no [DONE] sentinel.
@Suite(.serialized)
struct RestResponsesStreamingTests {

    @Test
    mutating func should_return_semantic_sse_events_without_a_done_sentinel() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .textFragment("Done."),
            .completed(
                promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 0,
                cachedTokenCount: 0, reason: .endOfSequence),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"stream\":true}");

        #expect(responsesResponse.statusCode == 200);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        #expect(responseText.contains("event: response.created"));
        #expect(responseText.contains("event: response.output_text.delta"));
        #expect(responseText.contains("\"delta\":\"Done.\""));
        #expect(responseText.contains("event: response.completed"));
        #expect(responseText.contains("[DONE]") == false);
    }

    @Test
    mutating func should_stream_responses_visible_text_separately_from_reasoning_for_opencode() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .reasoningFragment("internal thought"),
            .textFragment("Visible answer"),
            .completed(
                promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 1,
                cachedTokenCount: 0, reason: .endOfSequence),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"stream\":true}");

        #expect(responsesResponse.statusCode == 200);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        let parsedStream: RestResponsesJourneySupport.ParsedResponsesSseStream =
            RestResponsesJourneySupport.ParsedResponsesSseStream.parse(responseText);

        #expect(parsedStream.eventTypes() == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]);
        #expect(parsedStream.visibleTextForOpencode() == "Visible answer");
        #expect(parsedStream.reasoningSummaryText() == "internal thought");
        #expect(responseText.contains("[DONE]") == false);
        let completedResponse: [String: Any] = try Self.requireCompletedResponse(parsedStream);
        #expect(completedResponse["status"] as? String == "completed");
        #expect(completedResponse["output_text"] as? String == "Visible answer");
        let outputArray: Array<Any> = try Self.requireArray(completedResponse["output"], field: "output");
        let firstOutputItem: [String: Any] = try Self.requireObject(outputArray.first, field: "output[0]");
        #expect(firstOutputItem["type"] as? String == "reasoning");
        let secondOutputItem: [String: Any] = try Self.requireObject(outputArray[1], field: "output[1]");
        #expect(secondOutputItem["type"] as? String == "message");
        let usageObject: [String: Any] = try Self.requireObject(completedResponse["usage"], field: "usage");
        #expect(usageObject["input_tokens"] as? Int == 10);
        #expect(usageObject["output_tokens"] as? Int == 2);
        let outputDetails: [String: Any] = try Self.requireObject(
            usageObject["output_tokens_details"], field: "usage.output_tokens_details");
        #expect(outputDetails["reasoning_tokens"] as? Int == 1);
    }

    @Test
    mutating func should_stream_responses_parallel_function_call_lifecycle_for_opencode() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .toolCall(
                toolCallIndex: 0, functionName: "read",
                argumentsJson: "{\"filePath\":\"README.md\"}"),
            .toolCall(
                toolCallIndex: 1, functionName: "glob",
                argumentsJson: "{\"pattern\":\"tests/**/*.rs\"}"),
            .completed(
                promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 0,
                cachedTokenCount: 0, reason: .toolCalls),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"stream\":true}");

        #expect(responsesResponse.statusCode == 200);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        let parsedStream: RestResponsesJourneySupport.ParsedResponsesSseStream =
            RestResponsesJourneySupport.ParsedResponsesSseStream.parse(responseText);

        #expect(parsedStream.eventTypes() == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]);
        let functionCallItem: [String: Any] = try Self.requireObject(
            parsedStream.firstPayloadForEventType("response.output_item.added")?["item"],
            field: "output_item.added.item");
        #expect(functionCallItem["type"] as? String == "function_call");
        #expect(functionCallItem["name"] as? String == "read");
        let functionCallId: String = try Self.requireString(functionCallItem["call_id"], field: "item.call_id");
        #expect(functionCallId.hasPrefix("call_"));
        let argumentsDelta: [String: Any] = try Self.requireObject(
            parsedStream.firstPayloadForEventType("response.function_call_arguments.delta"),
            field: "function_call_arguments.delta");
        #expect(argumentsDelta["delta"] as? String == "{\"filePath\":\"README.md\"}");
        let completedResponse: [String: Any] = try Self.requireCompletedResponse(parsedStream);
        #expect(completedResponse["output_text"] as? String == "");
        let outputArray: Array<Any> = try Self.requireArray(completedResponse["output"], field: "output");
        let firstOutputItem: [String: Any] = try Self.requireObject(outputArray.first, field: "output[0]");
        #expect(firstOutputItem["type"] as? String == "function_call");
        let secondOutputItem: [String: Any] = try Self.requireObject(outputArray[1], field: "output[1]");
        #expect(secondOutputItem["type"] as? String == "function_call");
        #expect(secondOutputItem["name"] as? String == "glob");
    }

    @Test
    mutating func should_stream_the_context_overflow_signal_without_assistant_output() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .failed(reason: .contextLengthExceeded(
                actualTotalContextTokens: 262_145, maximumContextTokens: 262_144)),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"stream\":true}");

        #expect(responsesResponse.statusCode == 200);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        #expect(responseText.contains("event: response.failed"));
        #expect(responseText.contains("\"code\":\"context_length_exceeded\""));
        #expect(responseText.contains("\"status\":\"failed\""));
        #expect(responseText.contains("response.output_text.delta") == false);
        #expect(responseText.contains("[DONE]") == false);
    }

    @Test
    mutating func should_end_a_partially_emitted_stream_with_an_error_if_the_worker_closes() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .textFragment("Partial"),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"stream\":true}");

        #expect(responsesResponse.statusCode == 200);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        #expect(responseText.contains("event: response.output_text.delta"));
        #expect(responseText.contains("event: error"));
        #expect(responseText.contains("\"code\":\"worker_unavailable\""));
        #expect(responseText.contains("event: response.completed") == false);
    }

    // MARK: Journey plumbing

    private static func routeTable(
        streamEvents: Array<ChatGenerationStreamEvent>
    ) throws -> RestRouteTable {
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: RestResponsesJourneySupport.responsesModelId,
                capabilities: RestResponsesJourneySupport.readyResponsesCapabilities()),
            streamEvents: streamEvents);
        return try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(),
            responsesExecutor: scriptedExecutor);
    }

    private static func requireCompletedResponse(
        _ parsedStream: RestResponsesJourneySupport.ParsedResponsesSseStream
    ) throws -> [String: Any] {
        guard let completedResponse: [String: Any] = parsedStream.completedResponse() else {
            throw RestChatJourneyFailure.expectedObject("response.completed.response");
        }
        return completedResponse;
    }

    private static func requireObject(_ anyValue: Any?, field: String) throws -> [String: Any] {
        guard let objectValue: [String: Any] = anyValue as? [String: Any] else {
            throw RestChatJourneyFailure.expectedObject(field);
        }
        return objectValue;
    }

    private static func requireArray(_ anyValue: Any?, field: String) throws -> Array<Any> {
        guard let arrayValue: Array<Any> = anyValue as? Array<Any> else {
            throw RestChatJourneyFailure.expectedArray(field);
        }
        return arrayValue;
    }

    private static func requireString(_ anyValue: Any?, field: String) throws -> String {
        guard let stringValue: String = anyValue as? String else {
            throw RestChatJourneyFailure.expectedString(field);
        }
        return stringValue;
    }
}
