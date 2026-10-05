import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/thinking_budget_spellings.rs.
final class ChatCompletionRequestThinkingBudgetSpellingsTests: XCTestCase {

    func testShouldResolveTheCodingAgentThinkingBudgetSpellingAlone() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_token_budget": 96
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 96);
    }

    func testShouldResolveTheDocumentedThinkingBudgetAliasAlone() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget_tokens": 64
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 64);
    }

    func testShouldAcceptAgreeingThinkingBudgetSpellings() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 32,
        "thinking_token_budget": 32,
        "thinking_budget_tokens": 32
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 32);
    }

    func testShouldRejectDisagreeingThinkingBudgetSpellings() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 32,
        "thinking_token_budget": 96
    }
    """));
        do {
            try chatCompletionRequest.validate();
            XCTFail("disagreeing budget spellings must fail loudly");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(
                validationError,
                .thinkingControls(.conflictingNumericThinkingBudgets(
                    thinkingBudget: 32,
                    thinkingTokenBudget: 96,
                    thinkingBudgetTokens: nil,
                    reasoningMaxTokens: nil)));
        }
    }

    func testShouldMapTheReasoningEffortLevelsToThinkingBudgets() throws {
        let effortLevels: Array<(reasoningEffort: String, expectedBudget: UInt32)> = [
            ("minimal", 1024),
            ("low", 2048),
            ("medium", 8192),
            ("high", 16384),
            ("xhigh", 16384),
            ("max", 16384),
        ];
        for effortLevel in effortLevels {
            let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning_effort": "\(effortLevel.reasoningEffort)"
    }
    """);
            XCTAssertEqual(
                requestParts.thinkingBudget,
                effortLevel.expectedBudget,
                "reasoning_effort \(effortLevel.reasoningEffort) must enforce its token budget");
        }
    }

    func testShouldPreferTheExplicitThinkingBudgetOverReasoningEffort() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 300,
        "reasoning_effort": "high"
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 300);
    }

    func testShouldRejectAnUnknownReasoningEffortLabel() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning_effort": "turbo"
    }
    """));
        do {
            try chatCompletionRequest.validate();
            XCTFail("an unknown reasoning effort must fail loudly instead of being dropped");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(
                validationError,
                .thinkingControls(.unknownReasoningEffort(reasoningEffort: "turbo")));
        }
    }

    func testShouldDisableThinkingForOffReasoningEffort() throws {
        for reasoningEffort in ["off", "none"] {
            let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning_effort": "\(reasoningEffort)"
    }
    """);
            XCTAssertEqual(
                requestParts.thinkingBudget,
                0,
                "reasoning_effort \(reasoningEffort) must close the thinking channel");
        }
    }

    func testShouldResolveTheReasoningObjectEffortAndMaxTokens() throws {
        XCTAssertEqual(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "low"}
    }
    """).thinkingBudget,
            2048);
        XCTAssertEqual(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"max_tokens": 5000}
    }
    """).thinkingBudget,
            5000);
        // reasoning.max_tokens must agree with the top-level numeric spellings.
        XCTAssertEqual(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 5000,
        "reasoning": {"max_tokens": 5000}
    }
    """).thinkingBudget,
            5000);
        let disagreeingRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 96,
        "reasoning": {"max_tokens": 5000}
    }
    """));
        do {
            try disagreeingRequest.validate();
            XCTFail("a numeric disagreement must fail loudly");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.conflictingNumericThinkingBudgets) = validationError else {
                XCTFail("expected ConflictingNumericThinkingBudgets, got \(validationError)");
                return;
            }
        }
    }

    func testShouldDisableThinkingFromTheReasoningObjectEnabledFlag() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"enabled": false}
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 0);
    }

    func testShouldFlagReasoningExclusionWithoutChangingTheBudget() throws {
        let excludedParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"max_tokens": 5000, "exclude": true}
    }
    """);
        XCTAssertEqual(excludedParts.thinkingBudget, 5000);
        XCTAssertTrue(excludedParts.reasoningExcluded);
        let plainParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}]
    }
    """);
        XCTAssertFalse(plainParts.reasoningExcluded);
    }

    func testShouldAcceptTheFlatEnableThinkingFlagAndChatTemplateKwargs() throws {
        XCTAssertEqual(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": false
    }
    """).thinkingBudget,
            0);
        XCTAssertEqual(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": true,
        "chat_template_kwargs": {"enable_thinking": true},
        "reasoning": {"effort": "medium"}
    }
    """).thinkingBudget,
            8192);
        XCTAssertEqual(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }
    """).thinkingBudget,
            0);
        let disagreeingRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": true,
        "chat_template_kwargs": {"enable_thinking": false}
    }
    """));
        do {
            try disagreeingRequest.validate();
            XCTFail("disagreeing flags must fail loudly");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.conflictingThinkingEnableFlags) = validationError else {
                XCTFail("expected ConflictingThinkingEnableFlags, got \(validationError)");
                return;
            }
        }
    }

    func testShouldPreferAnExplicitDisableOverALevelName() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "none"},
        "reasoning_effort": "high"
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 0);
    }

    func testShouldRejectDisagreeingEffortLevelNames() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "low"},
        "reasoning_effort": "high"
    }
    """));
        do {
            try chatCompletionRequest.validate();
            XCTFail("disagreeing levels must fail loudly");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.conflictingReasoningEfforts) = validationError else {
                XCTFail("expected ConflictingReasoningEfforts, got \(validationError)");
                return;
            }
        }
    }

    func testShouldRejectDisabledThinkingWithAPositiveBudget() throws {
        let chatCompletionRequest: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": false,
        "thinking_budget": 5000
    }
    """));
        do {
            try chatCompletionRequest.validate();
            XCTFail("disable plus budget must fail loudly");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.thinkingDisabledWhileBudgetRequested) = validationError else {
                XCTFail("expected ThinkingDisabledWhileBudgetRequested, got \(validationError)");
                return;
            }
        }
    }

    func testShouldAbsorbUnknownReasoningAndTemplateKwargSubfields() {
        for unknownSubfield in [
            "{\"effort\": \"low\", \"bogus\": 1}",
            "{\"max_tokens\": 1, \"bogus\": 1}",
            // Copilot Desktop sends the OpenAI `summary` spelling; unknown fields
            // inside provider-shaped thinking objects must never reject the thread.
            "{\"effort\": \"medium\", \"summary\": \"auto\"}",
        ] {
            let requestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": \(unknownSubfield)
    }
    """;
            do {
                _ = try OpenAiChatCompletionRequest.decoded(
                    wireValue: try RestContractTestFixture.wireValue(requestJson));
            } catch {
                XCTFail("an unknown reasoning subfield must be absorbed: \(requestJson)");
            }
        }
        let kwargsRequestJson: String = """
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "chat_template_kwargs": {"enable_thinking": true, "temperature": 0.1}
    }
    """;
        do {
            _ = try OpenAiChatCompletionRequest.decoded(
                wireValue: try RestContractTestFixture.wireValue(kwargsRequestJson));
        } catch {
            XCTFail("unknown chat_template_kwargs entries must be absorbed");
        }
    }

    func testShouldKeepResolvedThinkingControlsUnchangedBesideAbsorbedFields() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "medium", "summary": "auto", "vendor_note": {"nested": true}}
    }
    """);
        XCTAssertEqual(requestParts.thinkingBudget, 8192);
        XCTAssertFalse(requestParts.reasoningExcluded);
    }

    private static func parseChatRequestParts(_ requestJson: String) throws -> OpenAiChatCompletionRequestParts {
        return try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson)).intoParts();
    }
}
