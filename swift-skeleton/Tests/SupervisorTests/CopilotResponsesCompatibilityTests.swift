import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// Copilot-client compatibility journeys over the Responses endpoint: the
/// coding-agent request spellings survive translation, and a finished
/// function-call item replays back into the worker conversation.
@Suite(.serialized)
struct CopilotResponsesCompatibilityTests {

    private static let copilotModelId: String = "astronomical/copilot-responses-compatibility-model";

    @Test
    mutating func should_complete_a_copilot_stream_with_visible_reasoning_and_supported_tool_schemas() throws -> Void {
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: Self.copilotModelId,
                capabilities: RestResponsesJourneySupport.readyResponsesCapabilities()),
            streamEvents: [
                .reasoningFragment("Inspecting locally."),
                .textFragment("ASTRONOMICAL_OK"),
                Self.completedGenerationEvent(.endOfSequence),
            ]);
        let routeTable: RestRouteTable = try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(),
            responsesExecutor: scriptedExecutor);
        let copilotRequestBody: String = "{\n"
            + "    \"model\":\"\(Self.copilotModelId)\",\n"
            + "    \"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\","
            + "\"text\":\"Reply exactly ASTRONOMICAL_OK\"}],\"type\":\"message\"}],\n"
            + "    \"tools\":[\n"
            + "        {\"type\":\"function\",\"name\":\"grep\",\"parameters\":{\"type\":\"object\","
            + "\"properties\":{\"paths\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"array\","
            + "\"items\":{\"type\":\"string\"}}]}}},\"strict\":false},\n"
            + "        {\"type\":\"function\",\"name\":\"glob\",\"parameters\":{\"type\":\"object\","
            + "\"properties\":{\"paths\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"array\","
            + "\"items\":{\"type\":\"string\"}}]}}},\"strict\":false},\n"
            + "        {\"type\":\"function\",\"name\":\"open_canvas\",\"parameters\":{\"type\":\"object\","
            + "\"properties\":{\"input\":{\"description\":\"Canvas open input matching the canvas input schema\"}}},"
            + "\"strict\":false},\n"
            + "        {\"type\":\"function\",\"name\":\"invoke_canvas_action\",\"parameters\":"
            + "{\"type\":\"object\",\"properties\":{\"input\":{\"description\":\"Action input matching "
            + "the action input schema\"}}},\"strict\":false}\n"
            + "    ],\n"
            + "    \"parallel_tool_calls\":true,\n"
            + "    \"reasoning\":{\"effort\":\"medium\"},\n"
            + "    \"prompt_cache_key\":\"copilot-session\",\n"
            + "    \"max_output_tokens\":20480,\n"
            + "    \"store\":false,\n"
            + "    \"stream\":true,\n"
            + "    \"include\":[\"reasoning.encrypted_content\"]\n"
            + "}";

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable, requestBody: copilotRequestBody);

        #expect(responsesResponse.statusCode == 200);
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        #expect(responseText.contains("event: response.reasoning_summary_text.delta"));
        #expect(responseText.contains("\"delta\":\"Inspecting locally.\""));
        #expect(responseText.contains("\"type\":\"summary_text\",\"text\":\"Inspecting locally.\""));
        #expect(responseText.contains("encrypted_content") == false);
        #expect(responseText.contains("\"delta\":\"ASTRONOMICAL_OK\""));
        #expect(responseText.contains("event: response.completed"));
        let admittedCommand: ChatGenerationCommand = try Self.requireAdmittedCommand(scriptedExecutor);
        #expect(admittedCommand.tools.count == 4);
        var sawAnyOfSchema: Bool = false;
        var sawCanvasDescription: Bool = false;
        for toolDefinition: ChatToolDefinition in admittedCommand.tools {
            if toolDefinition.name == "grep"
                && toolDefinition.parametersJson.contains(
                    "\"anyOf\":[{\"type\":\"string\"},{\"items\":{\"type\":\"string\"},\"type\":\"array\"}]") {
                sawAnyOfSchema = true;
            }
            if toolDefinition.name == "invoke_canvas_action"
                && toolDefinition.parametersJson.contains(
                    "\"description\":\"Action input matching the action input schema\"") {
                sawCanvasDescription = true;
            }
        }
        #expect(sawAnyOfSchema);
        #expect(sawCanvasDescription);
    }

    @Test
    mutating func should_replay_a_copilot_function_result_through_the_responses_endpoint() throws -> Void {
        // First turn: the worker asks for one function call.
        let firstExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: Self.copilotModelId,
                capabilities: RestResponsesJourneySupport.readyResponsesCapabilities()),
            streamEvents: [
                .toolCall(
                    toolCallIndex: 0, functionName: "view",
                    argumentsJson: "{\"path\":\"package.json\"}"),
                Self.completedGenerationEvent(.toolCalls),
            ]);
        let firstRouteTable: RestRouteTable = try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(),
            responsesExecutor: firstExecutor);
        let firstResponseBody: String = "{\n"
            + "    \"model\":\"\(Self.copilotModelId)\",\n"
            + "    \"input\":\"Read package.json.\",\n"
            + "    \"tools\":[{\"type\":\"function\",\"name\":\"view\",\"parameters\":"
            + "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
            + "\"required\":[\"path\"]},\"strict\":false}],\n"
            + "    \"stream\":true\n"
            + "}";

        let firstResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: firstRouteTable, requestBody: firstResponseBody);

        #expect(firstResponse.statusCode == 200);
        let functionCallItem: [String: Any] = try Self.requireFunctionCallItem(firstResponse);
        let functionCallId: String = try Self.requireItemString(functionCallItem["call_id"], field: "item.call_id");

        // Replay turn: the finished function-call item returns as input next
        // to its tool output, and the worker conversation reflects both.
        let replayExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: Self.copilotModelId,
                capabilities: RestResponsesJourneySupport.readyResponsesCapabilities()),
            streamEvents: [
                .textFragment("ASTRONOMICAL_TOOL_OK"),
                Self.completedGenerationEvent(.endOfSequence),
            ]);
        let replayRouteTable: RestRouteTable = try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(),
            responsesExecutor: replayExecutor);
        let replayRequestBody: String = "{\n"
            + "    \"model\":\"\(Self.copilotModelId)\",\n"
            + "    \"input\":[\n"
            + "        {\"role\":\"user\",\"content\":\"Read package.json.\"},\n"
            + "        \(Self.serializedJsonObject(functionCallItem)),\n"
            + "        {\"type\":\"function_call_output\",\"call_id\":\"\(functionCallId)\","
            + "\"output\":\"{\\\"name\\\":\\\"copilot-byok\\\"}\"}\n"
            + "    ],\n"
            + "    \"tools\":[{\"type\":\"function\",\"name\":\"view\",\"parameters\":"
            + "{\"type\":\"object\"},\"strict\":false}],\n"
            + "    \"stream\":true\n"
            + "}";

        let replayResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: replayRouteTable, requestBody: replayRequestBody);

        #expect(replayResponse.statusCode == 200);
        let replayResponseText: String = RestResponsesJourneySupport.responseText(replayResponse);
        #expect(replayResponseText.contains("\"delta\":\"ASTRONOMICAL_TOOL_OK\""));
        #expect(replayResponseText.contains("event: response.completed"));
        let replayCommand: ChatGenerationCommand = try Self.requireAdmittedCommand(replayExecutor);
        #expect(replayCommand.messages.count == 3);
        guard case let .assistant(replayAssistantContent, _, replayToolCalls) = replayCommand.messages[1],
              case let .tool(replayToolCallId, replayToolOutput) = replayCommand.messages[2] else {
            throw RestChatJourneyFailure.expectedObject("assistant and tool messages");
        }
        #expect(replayAssistantContent == nil);
        #expect(replayToolCalls.first?.id == replayToolCallId);
        #expect(replayToolOutput == "{\"name\":\"copilot-byok\"}");
    }

    // MARK: Journey plumbing

    private static func completedGenerationEvent(
        _ completionReason: ChatGenerationCompletionReason
    ) -> ChatGenerationStreamEvent {
        return .completed(
            promptTokenCount: 100, generatedTokenCount: 10, reasoningTokenCount: 0,
            cachedTokenCount: 0, reason: completionReason);
    }

    private static func requireAdmittedCommand(
        _ scriptedExecutor: ScriptedChatExecutor
    ) throws -> ChatGenerationCommand {
        guard let admittedCommand: ChatGenerationCommand = scriptedExecutor.receivedCommands.first else {
            throw RestChatJourneyFailure.missingAssistantMessage;
        }
        return admittedCommand;
    }

    private static func requireFunctionCallItem(
        _ responsesResponse: RestHttpResponse
    ) throws -> [String: Any] {
        let responseText: String = RestResponsesJourneySupport.responseText(responsesResponse);
        for responseLine: Substring in responseText.split(separator: "\n") {
            guard responseLine.hasPrefix("data: ") else {
                continue;
            }
            guard let payloadData: Any = try? JSONSerialization.jsonObject(
                with: Data(responseLine.dropFirst("data: ".count).utf8)),
                  let payloadObject: [String: Any] = payloadData as? [String: Any] else {
                continue;
            }
            if payloadObject["type"] as? String == "response.output_item.done",
               let itemObject: [String: Any] = payloadObject["item"] as? [String: Any],
               itemObject["type"] as? String == "function_call" {
                return itemObject;
            }
        }
        throw RestChatJourneyFailure.expectedObject("completed function_call item");
    }

    private static func requireItemString(_ anyValue: Any?, field: String) throws -> String {
        guard let stringValue: String = anyValue as? String else {
            throw RestChatJourneyFailure.expectedString(field);
        }
        return stringValue;
    }

    private static func serializedJsonObject(_ objectValue: [String: Any]) -> String {
        guard let serializedData: Data = try? JSONSerialization.data(withJSONObject: objectValue) else {
            return "{}";
        }
        return String(decoding: serializedData, as: UTF8.self);
    }
}
