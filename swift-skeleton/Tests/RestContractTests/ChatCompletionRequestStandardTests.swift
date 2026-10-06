import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/standard_request.rs.
@Suite(.tags(.hermeticJourney))
final class ChatCompletionRequestStandardTests {

    @Test
    func should_deserialize_a_standard_streaming_tool_use_request() throws {
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
        #expect(chatCompletionRequest.model() == "astronomical/fake-mixture-of-experts");
        #expect(chatCompletionRequest.messages().count == 2);
        #expect(chatCompletionRequest.tools().count == 1);
        #expect(chatCompletionRequest.stream());
        #expect(chatCompletionRequest.includesUsageInStream());
    }

    @Test
    func should_expose_validated_request_parts_without_leaking_rest_dtos_into_ipc() throws {
        let requestJson: String = """
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
        #expect(requestParts.model == "astronomical/fake-mixture-of-experts");
        #expect(requestParts.maximumOutputTokens == 512);
        #expect(requestParts.requestedMaximumOutputTokens == 512);
        #expect(requestParts.toolChoice == .none);
        #expect(requestParts.temperature == 0.6);
        #expect(requestParts.topP == 0.95);
        #expect(requestParts.seed == 7);
        #expect(requestParts.stream);
        #expect(requestParts.includesUsageInStream);
        #expect(requestParts.tools[0].name == "glob");
        #expect(requestParts.tools[0].parametersJson == #"{"type":"object"}"#);
        guard requestParts.messages.count == 1,
            case .user(let content, _) = requestParts.messages[0] else {
            Issue.record("expected a single user message part");
            return;
        }
        #expect(content == "Inspect the repository.");
    }

    @Test
    func should_preserve_omitted_chat_generation_settings_for_model_defaults() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(
                #"{"model":"organization/model","messages":[{"role":"user","content":"hello"}]}"#));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        #expect(requestParts.requestedMaximumOutputTokens == nil);
        #expect(requestParts.temperature == nil);
        #expect(requestParts.topP == nil);
    }
}
