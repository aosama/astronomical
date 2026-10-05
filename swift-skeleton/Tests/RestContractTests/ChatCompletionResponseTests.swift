import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_response.rs.
final class ChatCompletionResponseTests: XCTestCase {

    func testShouldSerializeAnOpenaiCompatibleStreamingTextDelta() throws {
        let streamingChunk: OpenAiChatCompletionChunk = OpenAiChatCompletionChunk.textDelta(
            id: "chatcmpl-local-42",
            created: 1_784_231_803,
            model: "astronomical/fake-mixture-of-experts",
            text: "local fragment");
        XCTAssertEqual(
            try streamingChunk.wireValue().serializedText,
            #"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"delta":{"content":"local fragment"},"finish_reason":null}]}"#);
    }

    func testShouldSerializeATerminalToolCallChunkWithRequestedUsage() throws {
        let tokenUsage: OpenAiTokenUsage = try XCTUnwrap(
            OpenAiTokenUsage.new(promptTokens: 31, completionTokens: 7));
        let terminalChunk: OpenAiChatCompletionChunk = OpenAiChatCompletionChunk.finished(
            id: "chatcmpl-local-42",
            created: 1_784_231_803,
            model: "astronomical/fake-mixture-of-experts",
            finishReason: .toolCalls).withUsage(usage: tokenUsage);
        XCTAssertEqual(
            try terminalChunk.wireValue().serializedText,
            #"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#);
    }

    func testShouldSerializeANonStreamingResponseWithReasoningAndText() throws {
        let tokenUsage: OpenAiTokenUsage = try XCTUnwrap(
            OpenAiTokenUsage.new(promptTokens: 31, completionTokens: 7));
        let response: OpenAiChatCompletionResponse = OpenAiChatCompletionResponse(
            id: "chatcmpl-local-42",
            created: 1_784_231_803,
            model: "astronomical/fake-mixture-of-experts",
            message: OpenAiAssistantMessage(
                content: "The repository uses Rust.",
                reasoningContent: "I inspected the source tree.",
                toolCalls: Array<OpenAiResponseToolCall>()),
            finishReason: .stop,
            usage: tokenUsage);
        XCTAssertEqual(
            try response.wireValue().serializedText,
            #"{"id":"chatcmpl-local-42","object":"chat.completion","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"message":{"role":"assistant","content":"The repository uses Rust.","reasoning_content":"I inspected the source tree."},"finish_reason":"stop"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#);
    }

    func testShouldSerializeCachedTokensInUsageWhenNonzero() throws {
        let tokenUsage: OpenAiTokenUsage = try XCTUnwrap(
            OpenAiTokenUsage.new(promptTokens: 4096, completionTokens: 100))
            .withCachedTokens(cachedTokens: 2048);
        let serialized: String = try tokenUsage.wireValue().serializedText;
        XCTAssertTrue(serialized.contains("\"prompt_tokens\":4096"),
            "expected prompt_tokens in serialized usage: \(serialized)");
        XCTAssertTrue(serialized.contains("\"completion_tokens\":100"),
            "expected completion_tokens in serialized usage: \(serialized)");
        XCTAssertTrue(serialized.contains("\"prompt_tokens_details\":{\"cached_tokens\":2048}"),
            "expected prompt_tokens_details.cached_tokens in serialized usage: \(serialized)");
    }

    func testShouldOmitCachedTokensFromUsageWhenZero() throws {
        let tokenUsage: OpenAiTokenUsage = try XCTUnwrap(
            OpenAiTokenUsage.new(promptTokens: 31, completionTokens: 7));
        let serialized: String = try tokenUsage.wireValue().serializedText;
        XCTAssertFalse(serialized.contains("cached_tokens"),
            "expected no cached_tokens in serialized usage: \(serialized)");
        XCTAssertFalse(serialized.contains("prompt_tokens_details"),
            "expected no prompt_tokens_details in serialized usage: \(serialized)");
    }
}
