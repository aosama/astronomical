import XCTest;
import RestContract;
import IpcProtocol;

/// Contract coverage for replayed history that carries provider-specific fields.
/// Mainstream harnesses resend messages produced by other providers, so unknown
/// fields must degrade to ignored instead of rejecting the whole request (#772).
/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/history_compatibility.rs.
final class ChatCompletionRequestHistoryCompatibilityTests: XCTestCase {

    func testShouldIgnoreUnknownFieldsOnReplayedHistoryMessages() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "system", "content": "Be helpful.", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "What is the weather?", "metadata": {"source": "web"}},
            {
                "role": "assistant",
                "content": "I cannot check the weather.",
                "finish_reason": "content_filter",
                "provider_specific": true
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
                "executed_at": "2026-01-01T00:00:00Z"
            }
        ]
    }
    """)).intoParts();
        guard case .some(.system(let firstContent)) = requestParts.messages.first else {
            XCTFail("expected a leading system message");
            return;
        }
        XCTAssertEqual(firstContent, "Be helpful.");
        guard case .some(.tool(_, let lastContent)) = requestParts.messages.last else {
            XCTFail("expected a trailing tool message");
            return;
        }
        XCTAssertEqual(lastContent, "sunny");
    }

    func testShouldPreserveAnAssistantRefusalAsMessageContent() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Ask the upstream model something"},
            {"role": "assistant", "refusal": "I'm sorry, but I can't help with that.", "finish_reason": "content_filter"}
        ]
    }
    """)).intoParts();
        guard case .some(.assistant(let content, _, _)) = requestParts.messages.last else {
            XCTFail("expected a trailing assistant message");
            return;
        }
        XCTAssertEqual(content, "I'm sorry, but I can't help with that.");
    }

    func testShouldPreferExplicitContentOverRefusalOnAnAssistantMessage() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "the visible answer", "refusal": "a hidden refusal"}
        ]
    }
    """)).intoParts();
        guard case .assistant(let content, _, _) = requestParts.messages.last else {
            XCTFail("expected a trailing assistant message");
            return;
        }
        XCTAssertEqual(content, "the visible answer");
    }

    func testShouldPreserveARefusalContentPartAsMessageText() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "before "},
                    {"type": "refusal", "refusal": "and a refusal"}
                ]
            }
        ]
    }
    """)).intoParts();
        guard case .assistant(let content, _, _) = requestParts.messages.last else {
            XCTFail("expected a trailing assistant message");
            return;
        }
        XCTAssertEqual(content, "before and a refusal");
    }

    func testShouldIgnoreUnknownFieldsInsideStreamOptionsAndHistoryToolCalls() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\\"command\\":\\"ls\\"}"},
                    "provider_marker": true
                }]
            }
        ],
        "stream": true,
        "stream_options": {"include_usage": true, "verbose": false}
    }
    """)).intoParts();
        XCTAssertTrue(requestParts.includesUsageInStream);
        guard case .assistant(_, _, let toolCalls) = requestParts.messages.last,
            let firstToolCall: OpenAiAssistantToolCallParts = toolCalls.first else {
            XCTFail("expected a trailing assistant message with tool calls");
            return;
        }
        XCTAssertEqual(firstToolCall.name, "bash");
    }

    func testShouldIgnoreAnUnknownFieldOnAnImageUrlObject() throws {
        let imageDataUri: String = "data:image/png;base64,iVBORw0KGgo=";
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "\(imageDataUri)", "detail": "auto"}
                }
            ]
        }]
    }
    """;
        let requestParts: OpenAiChatCompletionRequestParts = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson)).intoParts();
        guard case .some(.user(_, let images)) = requestParts.messages.first else {
            XCTFail("expected a leading user message with images");
            return;
        }
        XCTAssertEqual(images.count, 1);
    }

    func testShouldRejectAnUnknownMessageRole() {
        do {
            _ = try OpenAiChatCompletionRequest.decoded(
                wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "critic", "content": "still unknown"}
        ]
    }
    """));
            XCTFail("role is the message discriminator and must stay strict");
        } catch let deserializationError as JsonWireProblem {
            XCTAssertTrue(
                deserializationError.description.contains("unknown variant `critic`"),
                "an unknown role must be reported through the tag error, got: \(deserializationError)");
        } catch {
            XCTFail("unexpected error type: \(error)");
        }
    }

    func testShouldAcceptUnknownTopLevelRequestFields() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "logit_bias": {"13": -100},
        "service_tier": "auto"
    }
    """));
        try request.validate();
    }
}
