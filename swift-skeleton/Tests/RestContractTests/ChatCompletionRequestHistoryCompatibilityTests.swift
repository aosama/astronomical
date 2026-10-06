import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/**
 * Contract coverage for replayed history that carries provider-specific fields.
 * Mainstream harnesses resend messages produced by other providers, so unknown
 * fields must degrade to ignored instead of rejecting the whole request (#772).
 * Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/history_compatibility.rs.
 */
@Suite(.tags(.hermeticJourney))
final class ChatCompletionRequestHistoryCompatibilityTests {

    @Test
    func should_ignore_unknown_fields_on_replayed_history_messages() throws {
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
            Issue.record("expected a leading system message");
            return;
        }
        #expect(firstContent == "Be helpful.");
        guard case .some(.tool(_, let lastContent)) = requestParts.messages.last else {
            Issue.record("expected a trailing tool message");
            return;
        }
        #expect(lastContent == "sunny");
    }

    @Test
    func should_preserve_an_assistant_refusal_as_message_content() throws {
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
            Issue.record("expected a trailing assistant message");
            return;
        }
        #expect(content == "I'm sorry, but I can't help with that.");
    }

    @Test
    func should_prefer_explicit_content_over_refusal_on_an_assistant_message() throws {
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
            Issue.record("expected a trailing assistant message");
            return;
        }
        #expect(content == "the visible answer");
    }

    @Test
    func should_preserve_a_refusal_content_part_as_message_text() throws {
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
            Issue.record("expected a trailing assistant message");
            return;
        }
        #expect(content == "before and a refusal");
    }

    @Test
    func should_ignore_unknown_fields_inside_stream_options_and_history_tool_calls() throws {
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
        #expect(requestParts.includesUsageInStream);
        guard case .assistant(_, _, let toolCalls) = requestParts.messages.last,
            let firstToolCall: OpenAiAssistantToolCallParts = toolCalls.first else {
            Issue.record("expected a trailing assistant message with tool calls");
            return;
        }
        #expect(firstToolCall.name == "bash");
    }

    @Test
    func should_ignore_an_unknown_field_on_an_image_url_object() throws {
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
            Issue.record("expected a leading user message with images");
            return;
        }
        #expect(images.count == 1);
    }

    @Test
    func should_reject_an_unknown_message_role() {
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
            Issue.record("role is the message discriminator and must stay strict");
        } catch let deserializationError as JsonWireProblem {
            #expect(
                deserializationError.description.contains("unknown variant `critic`"),
                "an unknown role must be reported through the tag error, got: \(deserializationError)");
        } catch {
            Issue.record("unexpected error type: \(error)");
        }
    }

    @Test
    func should_accept_unknown_top_level_request_fields() throws {
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
