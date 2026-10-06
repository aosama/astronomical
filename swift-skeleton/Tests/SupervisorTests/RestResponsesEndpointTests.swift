import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// REST Responses endpoint journeys: admission rejections before the worker,
/// generated answers after it, and the request-echo contract.
@Suite(.serialized)
struct RestResponsesEndpointTests {

    @Test
    mutating func should_reject_invalid_json_before_worker_admission() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: Array());

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"broken\"");

        #expect(responsesResponse.statusCode == 400);
        #expect(try Self.errorMessage(responsesResponse)["code"] as? String == "invalid_json");
    }

    @Test
    mutating func should_reject_a_model_that_is_not_loaded() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: Array());

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"another/model\",\"input\":\"hello\"}");

        #expect(responsesResponse.statusCode == 400);
        #expect(try Self.errorMessage(responsesResponse)["code"] as? String == "model_not_found");
    }

    @Test
    mutating func should_reject_stateful_response_storage_before_worker_admission() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: Array());

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"store\":true}");

        #expect(responsesResponse.statusCode == 400);
        #expect(try Self.errorMessage(responsesResponse)["code"] as? String == "invalid_request");
    }

    @Test
    mutating func should_return_too_many_requests_when_generation_capacity_is_active() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(
            streamEvents: Array(), startError: .capacityUnavailable);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: Self.defaultBody(stream: false));

        #expect(responsesResponse.statusCode == 429);
        #expect(try Self.errorMessage(responsesResponse)["code"] as? String == "server_capacity");
    }

    @Test
    mutating func should_return_payload_too_large_when_a_responses_request_exceeds_the_ipc_frame() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(
            streamEvents: Array(),
            startError: .requestTooLarge(
                actualIpcMessageBytes: 33_554_433, maximumIpcMessageBytes: 33_554_432));

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: Self.defaultBody(stream: false));

        #expect(responsesResponse.statusCode == 413);
        #expect(try Self.errorMessage(responsesResponse)["code"] as? String == "request_too_large");
    }

    @Test
    mutating func should_explain_why_the_requested_responses_model_could_not_be_loaded() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(
            streamEvents: Array(),
            startError: .modelLoadFailed(
                modelLoadFailureReason: "OptiQ metadata uses unsupported 2-bit quantization"));

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: Self.defaultBody(stream: false));

        #expect(responsesResponse.statusCode == 503);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        #expect(responseText.contains("\"code\":\"model_load_failed\""));
        #expect(responseText.contains(
            "\"message\":\"the requested model could not be loaded: "
                + "OptiQ metadata uses unsupported 2-bit quantization\""));
    }

    @Test
    mutating func should_canonicalize_a_provider_prefixed_model_for_an_idle_worker() throws -> Void {
        let discoveredModel: DiscoveryDiscoveredModel =
            RestResponsesJourneySupport.discoveredChatModel(modelId: "requested-model");
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: 1,
                effectiveMlxMemoryCeilingBytes: 1,
                minimumMlxMemoryCeilingBytes: 1),
            startError: .capacityUnavailable);
        let routeTable: RestRouteTable = try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(
                discoveredModels: [discoveredModel]),
            responsesExecutor: scriptedExecutor);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"mlx-community/requested-model\",\"input\":\"hello\"}");

        // The provider prefix canonicalizes onto the discovered leaf model,
        // so the request reaches admission (and its queue-full rejection).
        #expect(responsesResponse.statusCode == 429);
        #expect(scriptedExecutor.receivedCommands.first?.model == "requested-model");
    }

    @Test
    mutating func should_return_context_length_exceeded_for_a_non_streaming_response() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .failed(reason: .contextLengthExceeded(
                actualTotalContextTokens: 262_145, maximumContextTokens: 262_144)),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: Self.defaultBody(stream: false));

        #expect(responsesResponse.statusCode == 400);
        let errorObject: [String: Any] = try Self.errorMessage(responsesResponse);
        #expect(errorObject["code"] as? String == "context_length_exceeded");
        #expect(errorObject["param"] as? String == "input");
    }

    @Test
    mutating func should_return_a_non_streaming_response_from_the_public_endpoint() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .textFragment("Done."),
            .completed(
                promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 0,
                cachedTokenCount: 0, reason: .endOfSequence),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
                + "\"input\":\"hello\",\"stream\":false}");

        #expect(responsesResponse.statusCode == 200);
        let envelope: [String: Any] = try Self.envelope(responsesResponse);
        #expect(envelope["object"] as? String == "response");
        #expect(envelope["status"] as? String == "completed");
        #expect(envelope["output_text"] as? String == "Done.");
        // Response metadata echoes caller intent; runtime defaults are not
        // rewritten as explicit input.
        #expect(envelope["temperature"] is NSNull || envelope["temperature"] == nil);
        #expect(envelope["top_p"] is NSNull || envelope["top_p"] == nil);
        #expect(envelope["max_output_tokens"] is NSNull || envelope["max_output_tokens"] == nil);
    }

    @Test
    mutating func should_echo_validated_request_configuration_in_the_response() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .completed(
                promptTokenCount: 4, generatedTokenCount: 1, reasoningTokenCount: 0,
                cachedTokenCount: 0, reason: .endOfSequence),
        ]);
        let requestBody: String = "{\n"
            + "    \"model\":\"\(RestResponsesJourneySupport.responsesModelId)\",\n"
            + "    \"input\":\"hello\",\n"
            + "    \"metadata\":{\"task\":\"edit\"},\n"
            + "    \"temperature\":0.5,\n"
            + "    \"top_p\":0.8,\n"
            + "    \"max_output_tokens\":64,\n"
            + "    \"tool_choice\":\"none\",\n"
            + "    \"tools\":[{\"type\":\"function\",\"name\":\"read_file\","
            + "\"description\":\"Read one file\",\"parameters\":{\"type\":\"object\"}}]\n"
            + "}";

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: requestBody);

        #expect(responsesResponse.statusCode == 200);
        let envelope: [String: Any] = try Self.envelope(responsesResponse);
        let metadataObject: [String: Any] = try Self.requireObject(envelope["metadata"], field: "metadata");
        #expect(metadataObject["task"] as? String == "edit");
        #expect(envelope["temperature"] as? Double == 0.5);
        #expect(envelope["top_p"] as? Double == 0.8);
        #expect(envelope["max_output_tokens"] as? Int == 64);
        #expect(envelope["tool_choice"] as? String == "none");
        let toolsArray: Array<Any> = try Self.requireArray(envelope["tools"], field: "tools");
        let firstTool: [String: Any] = try Self.requireObject(toolsArray.first, field: "tools[0]");
        #expect(firstTool["type"] as? String == "function");
        #expect(firstTool["name"] as? String == "read_file");
        let toolParameters: [String: Any] = try Self.requireObject(firstTool["parameters"], field: "tools[0].parameters");
        #expect(toolParameters["type"] as? String == "object");
    }

    @Test
    mutating func should_return_non_streaming_responses_reasoning_without_visible_output_text() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .reasoningFragment("internal thought"),
            .completed(
                promptTokenCount: 10, generatedTokenCount: 2, reasoningTokenCount: 2,
                cachedTokenCount: 0, reason: .endOfSequence),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: Self.defaultBody(stream: false));

        #expect(responsesResponse.statusCode == 200);
        let envelope: [String: Any] = try Self.envelope(responsesResponse);
        #expect(envelope["output_text"] as? String == "");
        let outputArray: Array<Any> = try Self.requireArray(envelope["output"], field: "output");
        let firstOutputItem: [String: Any] = try Self.requireObject(outputArray.first, field: "output[0]");
        #expect(firstOutputItem["type"] as? String == "reasoning");
        let summaryArray: Array<Any> = try Self.requireArray(firstOutputItem["summary"], field: "output[0].summary");
        let summaryEntry: [String: Any] = try Self.requireObject(summaryArray.first, field: "output[0].summary[0]");
        #expect(summaryEntry["type"] as? String == "summary_text");
        #expect(summaryEntry["text"] as? String == "internal thought");
        for outputItem: Any in outputArray {
            let outputItemObject: [String: Any] = try Self.requireObject(outputItem, field: "output[]");
            #expect(outputItemObject["type"] as? String != "message");
        }
    }

    @Test
    mutating func should_return_extracted_json_and_a_warning_for_text_format_json_schema() throws -> Void {
        let routeTable: RestRouteTable = try Self.routeTable(streamEvents: [
            .textFragment("```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```"),
            .completed(
                promptTokenCount: 12, generatedTokenCount: 8, reasoningTokenCount: 0,
                cachedTokenCount: 0, reason: .endOfSequence),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\n"
                + "    \"model\":\"\(RestResponsesJourneySupport.responsesModelId)\",\n"
                + "    \"input\":\"O Romeo, Romeo, wherefore art thou Romeo?\",\n"
                + "    \"text\":{\"format\":{\"type\":\"json_schema\",\"name\":\"romeo_line\","
                + "\"schema\":{\"type\":\"object\"}}},\n"
                + "    \"stream\":false\n"
                + "}");

        #expect(responsesResponse.statusCode == 200);
        let warningHeader: String? = RestResponsesJourneySupport.responseHeaderValue(
            responsesResponse, headerName: "Warning");
        #expect(warningHeader == ResponseFormatConstants.UNENFORCED_RESPONSE_FORMAT_WARNING);
        let envelope: [String: Any] = try Self.envelope(responsesResponse);
        let outputText: String = try Self.requireString(envelope["output_text"], field: "output_text");
        let extractedJson: [String: Any] = try Self.requireObject(
            try JSONSerialization.jsonObject(with: Data(outputText.utf8)), field: "output_text");
        #expect(extractedJson["speaker"] as? String == "Juliet");
        #expect(extractedJson["play"] as? String == "Romeo and Juliet");
    }

    // MARK: Journey plumbing

    private static func routeTable(
        streamEvents: Array<ChatGenerationStreamEvent>,
        startError: GenerationStartError? = nil
    ) throws -> RestRouteTable {
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: RestResponsesJourneySupport.responsesModelId,
                capabilities: RestResponsesJourneySupport.readyResponsesCapabilities()),
            streamEvents: streamEvents,
            startError: startError);
        return try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(),
            responsesExecutor: scriptedExecutor);
    }

    private static func defaultBody(stream: Bool) -> String {
        return "{\"model\":\"\(RestResponsesJourneySupport.responsesModelId)\","
            + "\"input\":\"hello\",\"stream\":\(stream)}";
    }

    private static func envelope(_ responsesResponse: RestHttpResponse) throws -> [String: Any] {
        guard let envelopeObject: [String: Any] = try JSONSerialization.jsonObject(
            with: responsesResponse.bodyBytes) as? [String: Any] else {
            throw RestChatJourneyFailure.nonJsonBody;
        }
        return envelopeObject;
    }

    private static func errorMessage(_ responsesResponse: RestHttpResponse) throws -> [String: Any] {
        guard let errorObject: [String: Any] = try Self.envelope(responsesResponse)["error"] as? [String: Any] else {
            throw RestChatJourneyFailure.missingErrorObject;
        }
        return errorObject;
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
