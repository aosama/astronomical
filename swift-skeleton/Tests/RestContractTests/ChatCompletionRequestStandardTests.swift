import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/standard_request.rs.
final class ChatCompletionRequestStandardTests: XCTestCase {

    func testShouldDeserializeAStandardStreamingToolUseRequest() throws {
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "List Rust source files."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "List matching files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": true,
        "stream_options": {"include_usage": true}
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        try chatCompletionRequest.validate();
        XCTAssertEqual(chatCompletionRequest.model(), "astronomical/fake-mixture-of-experts");
        XCTAssertEqual(chatCompletionRequest.messages().count, 2);
        XCTAssertEqual(chatCompletionRequest.tools().count, 1);
        XCTAssertTrue(chatCompletionRequest.stream());
        XCTAssertTrue(chatCompletionRequest.includesUsageInStream());
    }

    func testShouldExposeValidatedRequestPartsWithoutLeakingRestDTOsIntoIpc() throws {        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Inspect "}, {"type": "text", "text": "the repository."}]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "List matching files.",
                    "parameters": {"type": "object"}
                }
            }
        ],
        "tool_choice": "none",
        "max_completion_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 7,
        "stream": true,
        "stream_options": {"include_usage": true}
    }
    """;
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.model, "astronomical/fake-mixture-of-experts");
        XCTAssertEqual(requestParts.maximumOutputTokens, 512);
        XCTAssertEqual(requestParts.requestedMaximumOutputTokens, 512);
        XCTAssertEqual(requestParts.toolChoice, .none);
        XCTAssertEqual(requestParts.temperature, 0.6);
        XCTAssertEqual(requestParts.topP, 0.95);
        XCTAssertEqual(requestParts.seed, 7);
        XCTAssertTrue(requestParts.stream);
        XCTAssertTrue(requestParts.includesUsageInStream);
        XCTAssertEqual(requestParts.tools[0].name, "glob");
        XCTAssertEqual(requestParts.tools[0].parametersJson, #"{"type":"object"}"#);
        guard requestParts.messages.count == 1,
            case .user(let content, _) = requestParts.messages[0] else {
            XCTFail("expected a single user message part");
            return;
        }
        XCTAssertEqual(content, "Inspect the repository.");
    }

    func testShouldPreserveOmittedChatGenerationSettingsForModelDefaults() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(
                #"{"model":"organization/model","messages":[{"role":"user","content":"hello"}]}"#));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        XCTAssertNil(requestParts.requestedMaximumOutputTokens);
        XCTAssertNil(requestParts.temperature);
        XCTAssertNil(requestParts.topP);
    }
}
