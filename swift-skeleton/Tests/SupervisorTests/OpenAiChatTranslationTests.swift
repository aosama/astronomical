import Foundation;

import Testing;

import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Acceptance journeys for the public-to-IPC chat translation: a later system
 * message lowers into an escaped chronological user update, OpenCode's
 * reasoning effort becomes the thinking budget, large output budgets and
 * installed tool descriptions pass within their limits, the current OpenCode
 * tool-result wire shape round-trips, json_object injects the JSON
 * instruction, and a structured-outputs choice becomes the IPC mask.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class OpenAiChatTranslationTests {

    @Test
    func should_lower_a_later_system_message_to_a_chronological_user_update() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 904,
            requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Existing conversation context.\"},"
                + "{\"role\":\"system\",\"content\":\"A chronological policy update.\"},"
                + "{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatCommand.messages == [
            .user(
                content: "Existing conversation context.\n"
                    + "<system-update>\nA chronological policy update.\n</system-update>",
                images: []),
            .user(content: "hello", images: []),
        ]);
    }

    @Test
    func should_escape_chronological_system_update_wrapper_delimiters() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 905,
            requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Existing context.\"},"
                + "{\"role\":\"system\",\"content\":\"The text <system-update> & </system-update> "
                    + "must stay literal.\"}],\"stream\":true}");

        #expect(chatCommand.messages == [
            .user(
                content: "Existing context.\n<system-update>\n"
                    + "The text &lt;system-update&gt; &amp; &lt;/system-update&gt; must stay literal.\n"
                    + "</system-update>",
                images: []),
        ]);
    }

    @Test
    func should_translate_captured_opencode_reasoning_effort_into_the_thinking_budget() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 901,
            requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                + "\"messages\":[{\"role\":\"system\",\"content\":\"You generate concise conversation titles.\"},"
                + "{\"role\":\"user\",\"content\":\"Summarize this coding task.\"}],"
                + "\"stream\":true,\"stream_options\":{\"include_usage\":true},"
                + "\"max_tokens\":1024,\"temperature\":0.5,\"reasoning_effort\":\"low\"}");

        #expect(chatCommand.requestId == RequestId(rawRequestId: 901));
        #expect(chatCommand.model == "astronomical/fake-mixture-of-experts");
        #expect(chatCommand.messages == [
            .system(content: "You generate concise conversation titles."),
            .user(content: "Summarize this coding task.", images: []),
        ]);
        #expect(chatCommand.tools == []);
        #expect(chatCommand.toolChoice == .auto);
        #expect(chatCommand.settings.maxOutputTokens == 1_024);
        #expect(chatCommand.settings.temperatureThousandths == 500);
        #expect(chatCommand.settings.topPThousandths == nil);
        #expect(chatCommand.settings.seed == nil);
        #expect(chatCommand.settings.thinkingBudget == 2_048);
        #expect(chatCommand.structuredGeneration == nil);
    }

    @Test
    func should_translate_opencode_large_output_budget_to_the_worker() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 903,
            requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Inspect the repository and make "
                    + "all necessary edits.\"}],"
                + "\"stream\":true,\"stream_options\":{\"include_usage\":true},\"max_tokens\":20000}");

        #expect(chatCommand.settings.maxOutputTokens == 20_000);
    }

    @Test
    func should_translate_installed_opencode_bash_tool_description_within_public_rest_limit() throws {
        // The installed OpenCode bash tool ships a description of about this
        // size; the public REST limit must admit it so IPC translation sees it.
        let installedBashDescription: String = String(repeating: "x", count: 4_672);
        let descriptionWireText: String = String(
            decoding: try JSONSerialization.data(
                withJSONObject: [installedBashDescription]),
            as: UTF8.self)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"));
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 902,
            requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"Run a smoke test.\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
                + "\"description\":\(descriptionWireText),\"parameters\":{\"type\":\"object\"}}}],"
                + "\"tool_choice\":\"auto\",\"stream\":true}");

        #expect(chatCommand.tools.count == 1);
        #expect(chatCommand.tools[0].name == "bash");
        #expect(chatCommand.tools[0].description == installedBashDescription);
    }

    @Test
    func should_translate_the_current_opencode_tool_result_wire_shape_without_rest_dtos_crossing_ipc() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 906,
            requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                + "\"messages\":["
                + "{\"role\":\"system\",\"content\":\"You are a coding assistant.\"},"
                + "{\"role\":\"user\",\"content\":\"List Rust source files.\"},"
                + "{\"role\":\"assistant\",\"content\":null,"
                + "\"reasoning_content\":\"I should inspect the source tree.\","
                + "\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"glob\",\"arguments\":\"{\\\"pattern\\\":\\\"src/**/*.rs\\\"}\"}}]},"
                + "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"src/lib.rs\"},"
                + "{\"role\":\"user\",\"content\":\"Summarize the source files.\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"glob\","
                + "\"description\":\"List matching files.\","
                + "\"parameters\":{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}}}}}],"
                + "\"tool_choice\":\"auto\",\"stream\":true,\"max_tokens\":512,\"temperature\":0.6}");

        #expect(chatCommand.messages.count == 5);
        #expect(chatCommand.messages[0] == .system(content: "You are a coding assistant."));
        #expect(chatCommand.messages[1] == .user(
            content: "List Rust source files.", images: []));
        #expect(chatCommand.messages[2] == .assistant(
            content: nil,
            reasoningContent: "I should inspect the source tree.",
            toolCalls: [ChatAssistantToolCall(
                id: "call_1",
                function: ChatAssistantToolFunction(
                    name: "glob",
                    argumentsJson: "{\"pattern\":\"src/**/*.rs\"}"))]));
        #expect(chatCommand.messages[3] == .tool(toolCallId: "call_1", content: "src/lib.rs"));
        #expect(chatCommand.messages[4] == .user(
            content: "Summarize the source files.", images: []));
        #expect(chatCommand.tools.count == 1);
        #expect(chatCommand.tools[0].name == "glob");
        #expect(chatCommand.settings.maxOutputTokens == 512);
        #expect(chatCommand.settings.temperatureThousandths == 600);
    }

    @Test
    func should_inject_a_json_instruction_when_response_format_is_json_object() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 399,
            requestBody: "{\"model\":\"mlx-community/Qwen3.5-2B-4bit\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"O Romeo, Romeo, wherefore art thou Romeo?\"}],"
                + "\"response_format\":{\"type\":\"json_object\"}}");

        guard chatCommand.messages.count == 2,
              case let .system(instructionContent) = chatCommand.messages[0],
              case .user = chatCommand.messages[1] else {
            Issue.record("expected a leading system instruction, got \(chatCommand.messages)");
            return;
        }
        #expect(instructionContent.contains("JSON"));
    }

    @Test
    func should_translate_structured_outputs_choice_into_an_ipc_mask() throws {
        let chatCommand: ChatGenerationCommand = try OpenAiChatTranslationTests.translate(
            requestId: 410,
            requestBody: "{\"model\":\"mlx-community/Qwen3.5-2B-4bit\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"O Romeo, Romeo, wherefore art thou Romeo?\"}],"
                + "\"structured_outputs\":{\"choice\":[\"Juliet\",\"Romeo\"]}}");

        #expect(chatCommand.structuredGeneration == .choice(choices: ["Juliet", "Romeo"]));
    }

    @Test
    func should_reject_a_thousandths_unrepresentable_sampling_value() throws {
        do {
            _ = try OpenAiChatTranslationTests.translate(
                requestId: 907,
                requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
                    + "\"temperature\":0.1234567}");
            Issue.record("an unrepresentable sampling value must be rejected");
        } catch let translationRejection as OpenAiChatTranslationError {
            guard case let .samplingPrecisionUnsupported(parameterName, requestedValue) = translationRejection else {
                Issue.record("expected a sampling-precision rejection, got \(translationRejection)");
                return;
            }
            #expect(parameterName == "temperature");
            #expect(requestedValue > 0.12 && requestedValue < 0.13);
        }
    }

    @Test
    func should_reject_a_required_tool_choice_through_public_validation() throws {
        do {
            _ = try OpenAiChatTranslationTests.translate(
                requestId: 908,
                requestBody: "{\"model\":\"astronomical/fake-mixture-of-experts\","
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
                    + "\"tool_choice\":\"required\"}");
            Issue.record("the required tool choice must be rejected");
        } catch let translationRejection as OpenAiChatTranslationError {
            #expect(String(describing: translationRejection)
                .contains("tool choice mode 'required' is unsupported"));
        }
    }

    private static func translate(
        requestId: UInt64,
        requestBody: String
    ) throws -> ChatGenerationCommand {
        let requestWireValue: JsonWireValue = try JsonWireParser.parseDocument(
            documentBytes: Data(requestBody.utf8));
        let chatRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: requestWireValue);
        return try OpenAiChatTranslation.translateRequest(
            chatRequest,
            requestId: RequestId(rawRequestId: requestId));
    }
}
