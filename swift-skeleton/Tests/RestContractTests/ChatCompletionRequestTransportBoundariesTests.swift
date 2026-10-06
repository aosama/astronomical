import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/transport_boundaries.rs.
@Suite(.tags(.hermeticJourney))
final class ChatCompletionRequestTransportBoundariesTests {

    @Test
    func should_accept_opencode_large_output_budget_without_a_public_coding_cap() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "inspect the repository"}],
        "stream": true,
        "max_tokens": 20000
    }
    """));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        #expect(requestParts.maximumOutputTokens == 20_000);
    }

    @Test
    func should_accept_large_opencode_chat_history_without_a_public_message_count_cap() throws {
        let requestJson: String = Self.requestJson(messageCount: 250);
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        #expect(requestParts.messages.count == 250);
    }

    @Test
    func should_accept_many_small_text_content_parts_without_a_public_part_count_cap() throws {
        var contentPartTexts: Array<String> = Array();
        for partNumber in 0..<250 {
            contentPartTexts.append("{\"type\": \"text\", \"text\": \"part \(partNumber) \"}");
        }
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": [\(contentPartTexts.joined(separator: ","))]}]
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        guard requestParts.messages.count == 1,
            case .user(let content, _) = requestParts.messages[0] else {
            Issue.record("expected a single user message part");
            return;
        }
        #expect(content.contains("part 249"));
    }

    @Test
    func should_accept_many_small_tool_definitions_without_a_public_tool_count_cap() throws {
        var toolJsonTexts: Array<String> = Array();
        for toolNumber in 0..<250 {
            toolJsonTexts.append(
                "{\"type\": \"function\", \"function\": {\"name\": \"tool_\(toolNumber)\", \"parameters\": {\"type\": \"object\"}}}");
        }
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": [\(toolJsonTexts.joined(separator: ","))],
        "tool_choice": "auto"
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        #expect(requestParts.tools.count == 250);
    }

    @Test
    func should_accept_large_assistant_tool_call_arguments_without_a_public_field_byte_cap() throws {
        let largeArgumentsJson: String = "{\"payload\":\"\(String(repeating: "x", count: 80 * 1024))\"}";
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_large_arguments",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": "\(Self.escapeForJsonText(largeArgumentsJson))"
                }
            }]
        }]
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        guard requestParts.messages.count == 1,
            case .assistant(_, _, let toolCalls) = requestParts.messages[0] else {
            Issue.record("expected an assistant message part with tool calls");
            return;
        }
        #expect(toolCalls[0].argumentsJson.contains("payload"));
    }

    @Test
    func should_accept_a_single_text_message_larger_than_the_old_public_message_byte_limit() throws {
        let largeMessageContent: String = String(repeating: "x", count: 128 * 1024);
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "\(largeMessageContent)"}],
        "stream": true
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        guard requestParts.messages.count == 1,
            case .user(let content, _) = requestParts.messages[0] else {
            Issue.record("expected a single user message part");
            return;
        }
        #expect(content == largeMessageContent);
    }

    @Test
    func should_reject_an_unknown_oversized_reasoning_effort_label_instead_of_ignoring_it() throws {
        let oversizedReasoningEffort: String = String(repeating: "x", count: 8 * 1024);
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "reasoning_effort": "\(oversizedReasoningEffort)"
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        do {
            try chatCompletionRequest.validate();
            Issue.record("an unrecognized effort label must fail loudly, not get ignored by size");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.thinkingControls(
                        .unknownReasoningEffort(reasoningEffort: oversizedReasoningEffort)));
        }
    }

    private static func requestJson(messageCount: Int) -> String {
        var messageJsonTexts: Array<String> = Array();
        for messageNumber in 0..<messageCount {
            messageJsonTexts.append("{\"role\": \"user\", \"content\": \"short turn \(messageNumber)\"}");
        }
        return """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [\(messageJsonTexts.joined(separator: ","))],
        "stream": true
    }
    """;
    }

    /// Escapes JSON quotes inside a fragment already embedded in a JSON string
    /// literal, matching serde_json::to_string of the inner text.
    private static func escapeForJsonText(_ rawText: String) -> String {
        return rawText.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"");
    }
}
