import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/option_validation.rs.
final class ChatCompletionRequestOptionValidationTests: XCTestCase {

    func testShouldRejectAnOutputBudgetAboveTheWorkerRepresentationLimit() throws {
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
            XCTFail("the public endpoint must reject an oversized output budget");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(
                validationError,
                .outputTokenCountOutOfRange(
                    actualOutputTokens: ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS + 1,
                    maximumOutputTokens: ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS));
        }
    }

    func testShouldRejectCallerSuppliedStopSequencesBeforeWorkerAdmission() throws {
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
            XCTFail("caller-supplied stop sequences must not be accepted and ignored");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .unsupportedStopSequences);
        }
    }

    func testShouldRejectRequiredToolChoiceBeforeWorkerAdmission() throws {
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
            XCTFail("required tool choice must not rely on unenforced prompt hints");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .unsupportedToolChoice(mode: "required"));
        }
    }

    func testShouldRejectANamedForcedFunctionBeforeWorkerAdmission() throws {
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
            XCTFail("named function choices must not be accepted without deterministic enforcement");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .unsupportedForcedToolChoice(functionName: "glob"));
        }
    }

    func testShouldRejectAKnownButUnsupportedOpencodeOptionExplicitly() throws {
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
            XCTFail("known but unsupported options must fail closed");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .unsupportedOption(optionName: "store"));
        }
    }

    func testShouldAcceptHistoryToolCallsWithModelInventedNames() throws {
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

    func testShouldStillRejectInvalidToolDefinitionNames() throws {
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
            XCTFail("caller-declared tool definitions must keep the portable name grammar");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(validationError, .invalidToolName(toolName: "r=bash"));
        }
    }

    func testShouldRejectEmptyHistoryToolCallNames() throws {
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
            XCTFail("an empty history tool-call name carries no round-trip identity");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .emptyString = validationError else {
                XCTFail("expected EmptyString, got \(validationError)");
                return;
            }
        }
    }
}
