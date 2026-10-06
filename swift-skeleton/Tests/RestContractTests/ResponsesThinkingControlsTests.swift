import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_responses_thinking_controls.rs.
@Suite(.tags(.hermeticJourney))
final class ResponsesThinkingControlsTests {

    @Test
    func should_resolve_the_reasoning_object_max_tokens_into_the_thinking_budget() throws {
        let requestParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"max_tokens": 5000}
        }
        """);
        #expect(requestParts.thinkingBudget == 5000);
    }

    @Test
    func should_resolve_the_reasoning_object_effort_levels() throws {
        let effortLevels: Array<(effort: String, expectedBudget: UInt32)> = [
            ("minimal", 1024),
            ("low", 2048),
            ("medium", 8192),
            ("high", 16384),
            ("xhigh", 16384),
            ("max", 16384),
        ];
        for effortLevel in effortLevels {
            let requestParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"effort": "\(effortLevel.effort)"}
        }
        """);
            #expect(
                requestParts.thinkingBudget == effortLevel.expectedBudget,
                "reasoning.effort \(effortLevel.effort) must enforce its token budget");
        }
    }

    @Test
    func should_disable_thinking_from_every_disable_spelling() throws {
        let disableSpellings: Array<String> = [
            "\"reasoning\": {\"effort\": \"none\"}",
            "\"reasoning\": {\"enabled\": false}",
            "\"enable_thinking\": false",
            "\"chat_template_kwargs\": {\"enable_thinking\": false}",
            "\"reasoning_effort\": \"off\"",
        ];
        for disableSpelling in disableSpellings {
            let requestParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            \(disableSpelling)
        }
        """);
            #expect(
                requestParts.thinkingBudget == 0,
                "disable spelling \(disableSpelling) must close the thinking channel");
        }
    }

    @Test
    func should_flag_reasoning_exclusion_without_changing_the_budget() throws {
        let excludedParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"max_tokens": 5000, "exclude": true}
        }
        """);
        #expect(excludedParts.thinkingBudget == 5000);
        #expect(excludedParts.reasoningExcluded);
        let plainParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation."
        }
        """);
        #expect(plainParts.reasoningExcluded == false);
    }

    @Test
    func should_prefer_the_top_level_numeric_budget_over_reasoning_max_tokens_conflict_free() throws {
        let requestParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "thinking_budget": 5000,
            "reasoning": {"max_tokens": 5000, "exclude": true}
        }
        """);
        #expect(requestParts.thinkingBudget == 5000);
        #expect(requestParts.reasoningExcluded);
    }

    @Test
    func should_reject_disagreeing_numeric_and_level_spellings() throws {
        // numeric vs reasoning.max_tokens
        guard case .thinkingControls(.conflictingNumericThinkingBudgets) = try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "thinking_budget": 96,
            "reasoning": {"max_tokens": 5000}
        }
        """) else {
            Issue.record("expected ConflictingNumericThinkingBudgets");
            return;
        }
        // reasoning_effort vs reasoning.effort
        guard case .thinkingControls(.conflictingReasoningEfforts) = try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "reasoning_effort": "high",
            "reasoning": {"effort": "low"}
        }
        """) else {
            Issue.record("expected ConflictingReasoningEfforts");
            return;
        }
        // enable flags
        guard case .thinkingControls(.conflictingThinkingEnableFlags) = try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "enable_thinking": true,
            "reasoning": {"enabled": false}
        }
        """) else {
            Issue.record("expected ConflictingThinkingEnableFlags");
            return;
        }
        // disable with positive budget
        guard case .thinkingControls(.thinkingDisabledWhileBudgetRequested) = try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "enable_thinking": false,
            "thinking_budget": 5000
        }
        """) else {
            Issue.record("expected ThinkingDisabledWhileBudgetRequested");
            return;
        }
    }

    @Test
    func should_absorb_unknown_subfields_of_the_reasoning_object() {
        for unknownSubfield in [
            "{\"effort\": \"low\", \"bogus\": 1}",
            "{\"bogus\": true}",
            // Copilot Desktop sends the OpenAI `summary` spelling; unknown fields
            // inside provider-shaped thinking objects must never reject the thread.
            "{\"effort\": \"medium\", \"summary\": \"auto\"}",
        ] {
            let requestJson: String = """
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "reasoning": \(unknownSubfield)
        }
        """;
            do {
                _ = try OpenAiResponsesRequest.decoded(
                    wireValue: try RestContractTestFixture.wireValue(requestJson));
            } catch {
                Issue.record("an unknown reasoning subfield must be absorbed: \(unknownSubfield)");
            }
        }
        let kwargsRequestJson: String = """
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "chat_template_kwargs": {"enable_thinking": true, "bogus": 1}
        }
        """;
        do {
            _ = try OpenAiResponsesRequest.decoded(
                wireValue: try RestContractTestFixture.wireValue(kwargsRequestJson));
        } catch {
            Issue.record("unknown chat_template_kwargs entries must be absorbed");
        }
    }

    @Test
    func should_reject_an_unknown_reasoning_effort_label() throws {
        #expect(
            try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "reasoning_effort": "turbo"
        }
        """)
                == OpenAiResponsesValidationError.thinkingControls(
                    .unknownReasoningEffort(reasoningEffort: "turbo")));
    }

    private static func parseResponsesRequestParts(_ requestJson: String) throws -> OpenAiResponsesRequestParts {
        return try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson)).intoParts();
    }

    private static func validateResponsesRequest(_ requestJson: String) throws -> OpenAiResponsesValidationError {
        let responsesRequest: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        do {
            _ = try responsesRequest.intoParts();
            Issue.record("conflicting or invalid spellings must fail loudly");
        } catch let validationError as OpenAiResponsesValidationError {
            return validationError;
        }
        struct UnreachableValidationFailure: Error {}
        throw UnreachableValidationFailure();
    }
}
