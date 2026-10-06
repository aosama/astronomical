import Foundation;

import Testing;

import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Translation journeys for the public Responses request: a compact string
 * input lowers to one user chat message with thousandths sampling, and a
 * summary-reasoning plus function-loop replay lowers to the system, user,
 * assistant tool-call, tool-result conversation the worker expects.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class OpenAiResponsesTranslationTests {

    @Test
    func should_translate_string_input_into_one_user_chat_message() throws {
        let chatCommand: ChatGenerationCommand = try Self.translate(
            requestId: 700,
            requestBody: "{\n"
                + "    \"model\":\"astronomical/fake-mixture-of-experts\",\n"
                + "    \"input\":\"Explain this repository.\",\n"
                + "    \"max_output_tokens\":512,\n"
                + "    \"temperature\":0.6,\n"
                + "    \"top_p\":0.95\n"
                + "}");

        #expect(chatCommand.requestId == RequestId(rawRequestId: 700));
        #expect(chatCommand.model == "astronomical/fake-mixture-of-experts");
        #expect(chatCommand.messages == [
            .user(content: "Explain this repository.", images: []),
        ]);
        #expect(chatCommand.tools == []);
        #expect(chatCommand.toolChoice == .auto);
        #expect(chatCommand.settings.maxOutputTokens == 512);
        #expect(chatCommand.settings.temperatureThousandths == 600);
        #expect(chatCommand.settings.topPThousandths == 950);
        #expect(chatCommand.settings.seed == nil);
        #expect(chatCommand.settings.thinkingBudget == nil);
        #expect(chatCommand.qwenThinkingChannelSeed == nil);
        #expect(chatCommand.structuredGeneration == nil);
    }

    @Test
    func should_translate_summary_reasoning_and_function_loop_replay() throws {
        let chatCommand: ChatGenerationCommand = try Self.translate(
            requestId: 701,
            requestBody: "{\n"
                + "    \"model\":\"astronomical/fake-mixture-of-experts\",\n"
                + "    \"instructions\":\"You are a coding assistant.\",\n"
                + "    \"input\":[\n"
                + "        {\"role\":\"user\",\"content\":\"Inspect files.\"},\n"
                + "        {\"type\":\"reasoning\",\"id\":\"rs_prior\",\"summary\":["
                + "{\"type\":\"summary_text\",\"text\":\"I should list files.\"}],\"content\":[]},\n"
                + "        {\"type\":\"function_call\",\"id\":\"fc_prior\",\"call_id\":\"call_prior\","
                + "\"name\":\"glob\",\"arguments\":\"{\\\"pattern\\\":\\\"**/*.rs\\\"}\","
                + "\"status\":\"completed\"},\n"
                + "        {\"type\":\"function_call_output\",\"call_id\":\"call_prior\","
                + "\"output\":\"src/lib.rs\"},\n"
                + "        {\"role\":\"user\",\"content\":\"Summarize it.\"}\n"
                + "    ],\n"
                + "    \"tools\":[{\"type\":\"function\",\"name\":\"glob\","
                + "\"parameters\":{\"type\":\"object\"},\"strict\":false}],\n"
                + "    \"tool_choice\":\"auto\"\n"
                + "}");

        #expect(chatCommand.messages == [
            .system(content: "You are a coding assistant."),
            .user(content: "Inspect files.", images: []),
            .assistant(
                content: nil,
                reasoningContent: "I should list files.",
                toolCalls: [ChatAssistantToolCall(
                    id: "call_prior",
                    function: ChatAssistantToolFunction(
                        name: "glob",
                        argumentsJson: "{\"pattern\":\"**/*.rs\"}"))]),
            .tool(toolCallId: "call_prior", content: "src/lib.rs"),
            .user(content: "Summarize it.", images: []),
        ]);
        #expect(chatCommand.tools == [ChatToolDefinition(
            name: "glob",
            description: nil,
            parametersJson: "{\"type\":\"object\"}")]);
    }

    // MARK: Journey plumbing

    private static func translate(
        requestId: UInt64,
        requestBody: String
    ) throws -> ChatGenerationCommand {
        let requestWireValue: JsonWireValue = try JsonWireParser.parseDocument(
            documentBytes: Data(requestBody.utf8));
        let responsesRequest: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: requestWireValue);
        return try OpenAiResponsesTranslation.translateRequest(
            responsesRequest,
            requestId: RequestId(rawRequestId: requestId));
    }
}
