import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_chat_completion_request/thinking_budget_spellings.rs.
@Suite(.tags(.hermeticJourney))
final class ChatCompletionRequestThinkingBudgetSpellingsTests {

    @Test
    func should_resolve_the_coding_agent_thinking_budget_spelling_alone() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_token_budget": 96
    }
    """);
        #expect(requestParts.thinkingBudget == 96);
    }

    @Test
    func should_resolve_the_documented_thinking_budget_alias_alone() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget_tokens": 64
    }
    """);
        #expect(requestParts.thinkingBudget == 64);
    }

    @Test
    func should_accept_agreeing_thinking_budget_spellings() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 32,
        "thinking_token_budget": 32,
        "thinking_budget_tokens": 32
    }
    """);
        #expect(requestParts.thinkingBudget == 32);
    }

    @Test
    func should_reject_disagreeing_thinking_budget_spellings() throws {
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
            Issue.record("disagreeing budget spellings must fail loudly");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.thinkingControls(
                        .conflictingNumericThinkingBudgets(
                            thinkingBudget: 32,
                            thinkingTokenBudget: 96,
                            thinkingBudgetTokens: nil,
                            reasoningMaxTokens: nil)));
        }
    }

    @Test
    func should_map_the_reasoning_effort_levels_to_thinking_budgets() throws {
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
            #expect(
                requestParts.thinkingBudget == effortLevel.expectedBudget,
                "reasoning_effort \(effortLevel.reasoningEffort) must enforce its token budget");
        }
    }

    @Test
    func should_prefer_the_explicit_thinking_budget_over_reasoning_effort() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 300,
        "reasoning_effort": "high"
    }
    """);
        #expect(requestParts.thinkingBudget == 300);
    }

    @Test
    func should_reject_an_unknown_reasoning_effort_label() throws {
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
            Issue.record("an unknown reasoning effort must fail loudly instead of being dropped");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError
                    == OpenAiChatCompletionValidationError.thinkingControls(
                        .unknownReasoningEffort(reasoningEffort: "turbo")));
        }
    }

    @Test
    func should_disable_thinking_for_off_reasoning_effort() throws {
        for reasoningEffort in ["off", "none"] {
            let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning_effort": "\(reasoningEffort)"
    }
    """);
            #expect(
                requestParts.thinkingBudget == 0,
                "reasoning_effort \(reasoningEffort) must close the thinking channel");
        }
    }

    @Test
    func should_resolve_the_reasoning_object_effort_and_max_tokens() throws {
        #expect(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "low"}
    }
    """).thinkingBudget
                == 2048);
        #expect(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"max_tokens": 5000}
    }
    """).thinkingBudget
                == 5000);
        // reasoning.max_tokens must agree with the top-level numeric spellings.
        #expect(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "thinking_budget": 5000,
        "reasoning": {"max_tokens": 5000}
    }
    """).thinkingBudget
                == 5000);
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
            Issue.record("a numeric disagreement must fail loudly");
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.conflictingNumericThinkingBudgets) = validationError else {
                Issue.record("expected ConflictingNumericThinkingBudgets, got \(validationError)");
                return;
            }
        }
    }

    @Test
    func should_disable_thinking_from_the_reasoning_object_enabled_flag() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"enabled": false}
    }
    """);
        #expect(requestParts.thinkingBudget == 0);
    }

    @Test
    func should_flag_reasoning_exclusion_without_changing_the_budget() throws {
        let excludedParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"max_tokens": 5000, "exclude": true}
    }
    """);
        #expect(excludedParts.thinkingBudget == 5000);
        #expect(excludedParts.reasoningExcluded);
        let plainParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}]
    }
    """);
        #expect(plainParts.reasoningExcluded == false);
    }

    @Test
    func should_accept_the_flat_enable_thinking_flag_and_chat_template_kwargs() throws {
        #expect(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": false
    }
    """).thinkingBudget
                == 0);
        #expect(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "enable_thinking": true,
        "chat_template_kwargs": {"enable_thinking": true},
        "reasoning": {"effort": "medium"}
    }
    """).thinkingBudget
                == 8192);
        #expect(
            try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "chat_template_kwargs": {"enable_thinking": false}
    }
    """).thinkingBudget
                == 0);
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
            Issue.record("disagreeing flags must fail loudly");
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.conflictingThinkingEnableFlags) = validationError else {
                Issue.record("expected ConflictingThinkingEnableFlags, got \(validationError)");
                return;
            }
        }
    }

    @Test
    func should_prefer_an_explicit_disable_over_a_level_name() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "none"},
        "reasoning_effort": "high"
    }
    """);
        #expect(requestParts.thinkingBudget == 0);
    }

    @Test
    func should_reject_disagreeing_effort_level_names() throws {
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
            Issue.record("disagreeing levels must fail loudly");
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.conflictingReasoningEfforts) = validationError else {
                Issue.record("expected ConflictingReasoningEfforts, got \(validationError)");
                return;
            }
        }
    }

    @Test
    func should_reject_disabled_thinking_with_a_positive_budget() throws {
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
            Issue.record("disable plus budget must fail loudly");
        } catch let validationError as OpenAiChatCompletionValidationError {
            guard case .thinkingControls(.thinkingDisabledWhileBudgetRequested) = validationError else {
                Issue.record("expected ThinkingDisabledWhileBudgetRequested, got \(validationError)");
                return;
            }
        }
    }

    @Test
    func should_absorb_unknown_reasoning_and_template_kwarg_subfields() {
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
                Issue.record("an unknown reasoning subfield must be absorbed: \(requestJson)");
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
            Issue.record("unknown chat_template_kwargs entries must be absorbed");
        }
    }

    @Test
    func should_keep_resolved_thinking_controls_unchanged_beside_absorbed_fields() throws {
        let requestParts: OpenAiChatCompletionRequestParts = try Self.parseChatRequestParts("""
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [{"role": "user", "content": "write a function"}],
        "reasoning": {"effort": "medium", "summary": "auto", "vendor_note": {"nested": true}}
    }
    """);
        #expect(requestParts.thinkingBudget == 8192);
        #expect(requestParts.reasoningExcluded == false);
    }

    private static func parseChatRequestParts(_ requestJson: String) throws -> OpenAiChatCompletionRequestParts {
        return try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson)).intoParts();
    }
}
