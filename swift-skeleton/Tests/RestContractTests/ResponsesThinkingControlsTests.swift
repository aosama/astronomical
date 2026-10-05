import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_responses_thinking_controls.rs.
final class ResponsesThinkingControlsTests: XCTestCase {

    func testShouldResolveTheReasoningObjectMaxTokensIntoTheThinkingBudget() throws {
        let requestParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"max_tokens": 5000}
        }
        """);
        XCTAssertEqual(requestParts.thinkingBudget, 5000);
    }

    func testShouldResolveTheReasoningObjectEffortLevels() throws {
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
            XCTAssertEqual(
                requestParts.thinkingBudget,
                effortLevel.expectedBudget,
                "reasoning.effort \(effortLevel.effort) must enforce its token budget");
        }
    }

    func testShouldDisableThinkingFromEveryDisableSpelling() throws {
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
            XCTAssertEqual(
                requestParts.thinkingBudget,
                0,
                "disable spelling \(disableSpelling) must close the thinking channel");
        }
    }

    func testShouldFlagReasoningExclusionWithoutChangingTheBudget() throws {
        let excludedParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "reasoning": {"max_tokens": 5000, "exclude": true}
        }
        """);
        XCTAssertEqual(excludedParts.thinkingBudget, 5000);
        XCTAssertTrue(excludedParts.reasoningExcluded);
        let plainParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation."
        }
        """);
        XCTAssertFalse(plainParts.reasoningExcluded);
    }

    func testShouldPreferTheTopLevelNumericBudgetOverReasoningMaxTokensConflictFree() throws {
        let requestParts: OpenAiResponsesRequestParts = try Self.parseResponsesRequestParts("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Summarize this conversation.",
            "thinking_budget": 5000,
            "reasoning": {"max_tokens": 5000, "exclude": true}
        }
        """);
        XCTAssertEqual(requestParts.thinkingBudget, 5000);
        XCTAssertTrue(requestParts.reasoningExcluded);
    }

    func testShouldRejectDisagreeingNumericAndLevelSpellings() throws {
        // numeric vs reasoning.max_tokens
        guard case .thinkingControls(.conflictingNumericThinkingBudgets) = try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "thinking_budget": 96,
            "reasoning": {"max_tokens": 5000}
        }
        """) else {
            XCTFail("expected ConflictingNumericThinkingBudgets");
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
            XCTFail("expected ConflictingReasoningEfforts");
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
            XCTFail("expected ConflictingThinkingEnableFlags");
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
            XCTFail("expected ThinkingDisabledWhileBudgetRequested");
            return;
        }
    }

    func testShouldAbsorbUnknownSubfieldsOfTheReasoningObject() {
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
                XCTFail("an unknown reasoning subfield must be absorbed: \(unknownSubfield)");
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
            XCTFail("unknown chat_template_kwargs entries must be absorbed");
        }
    }

    func testShouldRejectAnUnknownReasoningEffortLabel() throws {
        XCTAssertEqual(
            try Self.validateResponsesRequest("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "hello",
            "reasoning_effort": "turbo"
        }
        """),
            .thinkingControls(.unknownReasoningEffort(reasoningEffort: "turbo")));
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
            XCTFail("conflicting or invalid spellings must fail loudly");
        } catch let validationError as OpenAiResponsesValidationError {
            return validationError;
        }
        struct UnreachableValidationFailure: Error {}
        throw UnreachableValidationFailure();
    }
}
