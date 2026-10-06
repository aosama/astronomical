import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/option_validation.rs.
@Suite(.tags(.hermeticJourney))
final class ChatCompletionRequestOptionValidationTests {

    @Test
    func should_reject_an_output_budget_above_the_worker_representation_limit() throws {
        let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "max_tokens": \(ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS + 1)
    }
    """;
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        do {
            try chatCompletionRequest.validate();
            Issue.record("the public endpoint must reject an oversized output budget");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.outputTokenCountOutOfRange(
                        actualOutputTokens: ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS + 1,
                        maximumOutputTokens: ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS));
        }
    }

    @Test
    func should_reject_caller_supplied_stop_sequences_before_worker_admission() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "stop": ["</tool_call>"]
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("caller-supplied stop sequences must not be accepted and ignored");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError == OpenAiChatCompletionValidationError.unsupportedStopSequences);
        }
    }

    @Test
    func should_reject_required_tool_choice_before_worker_admission() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "glob", "parameters": {"type": "object"}}
            }
        ],
        "tool_choice": "required"
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("required tool choice must not rely on unenforced prompt hints");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.unsupportedToolChoice(mode: "required"));
        }
    }

    @Test
    func should_reject_a_named_forced_function_before_worker_admission() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "glob", "parameters": {"type": "object"}}
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "glob"}}
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("named function choices must not be accepted without deterministic enforcement");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.unsupportedForcedToolChoice(functionName: "glob"));
        }
    }

    @Test
    func should_reject_a_known_but_unsupported_opencode_option_explicitly() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "Inspect the repository."}],
        "store": true
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("known but unsupported options must fail closed");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.unsupportedOption(optionName: "store"));
        }
    }

    @Test
    func should_accept_history_tool_calls_with_model_invented_names() throws {
        // The Qwen output parser deliberately fail-opens closed tool-call envelopes
        // with unknown or malformed names to the harness. A follow-up request
        // echoes that model output as assistant history; rejecting the replayed
        // name would strand the whole conversation (issue #430 field report:
        // a model-invented "r=bash" tool call returned invalid_request on the
        // next turn).
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Inspect the repository."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-model-invented-1",
                "type": "function",
                "function": {"name": "r=bash", "arguments": "{\\"command\\": \\"git status\\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call-model-invented-1", "content": "clean working tree"},
            {"role": "user", "content": "Summarize what you learned."}
        ],
        "tools": [{
            "type": "function",
            "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}
        }]
    }
    """));
        try chatCompletionRequest.validate();
    }

    @Test
    func should_still_reject_invalid_tool_definition_names() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "tools": [{
            "type": "function",
            "function": {"name": "r=bash", "parameters": {"type": "object", "properties": {}}}
        }]
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("caller-declared tool definitions must keep the portable name grammar");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.invalidToolName(toolName: "r=bash"));
        }
    }

    @Test
    func should_reject_empty_history_tool_call_names() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Inspect the repository."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "", "arguments": "{}"}
            }]}
        ]
    }
    """));
        do {
            try chatCompletionRequest.validate();
            Issue.record("an empty history tool-call name carries no round-trip identity");
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .emptyString = validationError else {
                Issue.record("expected EmptyString, got \(validationError)");
                return;
            }
        }
    }
}
