import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_response.rs.
@Suite(.tags(.hermeticJourney))
final class ChatCompletionResponseTests {

    @Test
    func should_serialize_an_openai_compatible_streaming_text_delta() throws {
        let streamingChunk: OpenAiChatCompletionChunk = OpenAiChatCompletionChunk.textDelta(
            id: "chatcmpl-local-42",
            created: 1_784_231_803,
            model: "astronomical/fake-mixture-of-experts",
            text: "local fragment");
        #expect(
            try streamingChunk.wireValue().serializedText
                == #"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"delta":{"content":"local fragment"},"finish_reason":null}]}"#);
    }

    @Test
    func should_serialize_a_terminal_tool_call_chunk_with_requested_usage() throws {
        let tokenUsage: OpenAiTokenUsage = try #require(
            OpenAiTokenUsage.new(promptTokens: 31, completionTokens: 7));
        let terminalChunk: OpenAiChatCompletionChunk = OpenAiChatCompletionChunk.finished(
            id: "chatcmpl-local-42",
            created: 1_784_231_803,
            model: "astronomical/fake-mixture-of-experts",
            finishReason: .toolCalls).withUsage(usage: tokenUsage);
        #expect(
            try terminalChunk.wireValue().serializedText
                == #"{"id":"chatcmpl-local-42","object":"chat.completion.chunk","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#);
    }

    @Test
    func should_serialize_a_non_streaming_response_with_reasoning_and_text() throws {
        let tokenUsage: OpenAiTokenUsage = try #require(
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
        #expect(
            try response.wireValue().serializedText
                == #"{"id":"chatcmpl-local-42","object":"chat.completion","created":1784231803,"model":"astronomical/fake-mixture-of-experts","choices":[{"index":0,"message":{"role":"assistant","content":"The repository uses Rust.","reasoning_content":"I inspected the source tree."},"finish_reason":"stop"}],"usage":{"prompt_tokens":31,"completion_tokens":7,"total_tokens":38}}"#);
    }

    @Test
    func should_serialize_cached_tokens_in_usage_when_nonzero() throws {
        let tokenUsage: OpenAiTokenUsage = try #require(
            OpenAiTokenUsage.new(promptTokens: 4096, completionTokens: 100))
            .withCachedTokens(cachedTokens: 2048);
        let serialized: String = try tokenUsage.wireValue().serializedText;
        #expect(serialized.contains("\"prompt_tokens\":4096"),
            "expected prompt_tokens in serialized usage: \(serialized)");
        #expect(serialized.contains("\"completion_tokens\":100"),
            "expected completion_tokens in serialized usage: \(serialized)");
        #expect(serialized.contains("\"prompt_tokens_details\":{\"cached_tokens\":2048}"),
            "expected prompt_tokens_details.cached_tokens in serialized usage: \(serialized)");
    }

    @Test
    func should_omit_cached_tokens_from_usage_when_zero() throws {
        let tokenUsage: OpenAiTokenUsage = try #require(
            OpenAiTokenUsage.new(promptTokens: 31, completionTokens: 7));
        let serialized: String = try tokenUsage.wireValue().serializedText;
        #expect(serialized.contains("cached_tokens") == false,
            "expected no cached_tokens in serialized usage: \(serialized)");
        #expect(serialized.contains("prompt_tokens_details") == false,
            "expected no prompt_tokens_details in serialized usage: \(serialized)");
    }
}
