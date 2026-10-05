import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/transport_boundaries.rs.
final class ChatCompletionRequestTransportBoundariesTests: XCTestCase {

    func testShouldAcceptOpencodeLargeOutputBudgetWithoutAPublicCodingCap() throws {
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
        XCTAssertEqual(requestParts.maximumOutputTokens, 20_000);
    }

    func testShouldAcceptLargeOpencodeChatHistoryWithoutAPublicMessageCountCap() throws {
        let requestJson: String = Self.requestJson(messageCount: 250);
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try chatCompletionRequest.intoParts();
        XCTAssertEqual(requestParts.messages.count, 250);
    }

    func testShouldAcceptManySmallTextContentPartsWithoutAPublicPartCountCap() throws {
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
            XCTFail("expected a single user message part");
            return;
        }
        XCTAssertTrue(content.contains("part 249"));
    }

    func testShouldAcceptManySmallToolDefinitionsWithoutAPublicToolCountCap() throws {
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
        XCTAssertEqual(requestParts.tools.count, 250);
    }

    func testShouldAcceptLargeAssistantToolCallArgumentsWithoutAPublicFieldByteCap() throws {
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
            XCTFail("expected an assistant message part with tool calls");
            return;
        }
        XCTAssertTrue(toolCalls[0].argumentsJson.contains("payload"));
    }

    func testShouldAcceptASingleTextMessageLargerThanTheOldPublicMessageByteLimit() throws {
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
            XCTFail("expected a single user message part");
            return;
        }
        XCTAssertEqual(content, largeMessageContent);
    }

    func testShouldRejectAnUnknownOversizedReasoningEffortLabelInsteadOfIgnoringIt() throws {
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
            XCTFail("an unrecognized effort label must fail loudly, not get ignored by size");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(
                validationError,
                .thinkingControls(.unknownReasoningEffort(reasoningEffort: oversizedReasoningEffort)));
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
