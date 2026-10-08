import Foundation;

import Testing;

import MLX;
import MLXGuidedGeneration;
import MLXLMCommon;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * Hermetic journeys for the enforced structured-generation adapter: the
 * upstream grammar engine masks exactly the allowed tokens, commits advance
 * the matcher to termination, choice constraints lower to literal
 * alternations with escaped GBNF literals, and the chat processor compiles
 * an enforced constraint into the prepared request while the unsupported
 * guided-regex kind fails closed with a bounded reason.
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class Qwen35GuidedConstraintTests {

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    /// Inline synthetic JSON vocabulary with literal UTF-8 pieces
    /// (VocabType .raw), so mask expectations are exact.
    private static let jsonVocabulary: Array<String> = [
        "{", "}", "\"", "a", ":", ",", " ", "<|im_end|>",
    ];
    private static let jsonEndOfSequenceTokenId: Int32 = 7;
    /// Whole-word vocabulary for the choice journeys: one token per literal.
    private static let choiceVocabulary: Array<String> = ["What", "Two", "<|im_end|>"];

    private static func jsonGrammarTokenizer() throws -> GrammarTokenizer {
        return try GrammarTokenizer(
            vocab: jsonVocabulary, vocabType: .raw, eosTokenId: jsonEndOfSequenceTokenId);
    }

    private static func jsonConstraint() throws -> Qwen35GuidedConstraint {
        return Qwen35GuidedConstraint(grammarConstraint: try GrammarConstraint(
            tokenizer: try Self.jsonGrammarTokenizer(),
            jsonSchema: "{\"type\":\"object\"}"));
    }

    @Test
    func should_mask_logits_to_the_grammar_allowed_tokens_for_a_json_object_constraint() throws {
        let compiledConstraint: Qwen35GuidedConstraint = try Self.jsonConstraint();
        let logitRow: MLXArray = MLXArray(Array(repeating: Float(0.5), count: 8));

        let maskedRow: MLXArray = try compiledConstraint.maskLogits(logitRow);
        let maskedValues: Array<Float> = maskedRow.asArray(Float.self);

        #expect(maskedValues.count == 8);
        // Canonical JSON opens with the brace only; every other piece — the
        // close brace, quote, letter, colon, comma, whitespace, and end
        // marker — is masked at the grammar's start state.
        #expect(maskedValues[0] == 0.5);
        for disallowedTokenId: Int in [1, 2, 3, 4, 5, 6, 7] {
            #expect(maskedValues[disallowedTokenId].isInfinite
                && maskedValues[disallowedTokenId] < 0, """
                token \(disallowedTokenId) should be masked
                """);
        }
    }

    @Test
    func should_allow_the_end_marker_only_after_a_complete_object() throws {
        let compiledConstraint: Qwen35GuidedConstraint = try Self.jsonConstraint();

        #expect(compiledConstraint.isTerminated() == false);
        try compiledConstraint.commitToken(0);
        // After the opening brace the close brace, a quoted property name,
        // and separating whitespace are the legal continuations; everything
        // else stays masked.
        let openObjectRow: MLXArray = try compiledConstraint.maskLogits(
            MLXArray(Array(repeating: Float(0.5), count: 8)));
        let openObjectValues: Array<Float> = openObjectRow.asArray(Float.self);
        #expect(openObjectValues[1] == 0.5);
        #expect(openObjectValues[2] == 0.5);
        #expect(openObjectValues[6] == 0.5);
        for disallowedTokenId: Int in [0, 3, 4, 5, 7] {
            #expect(openObjectValues[disallowedTokenId].isInfinite
                && openObjectValues[disallowedTokenId] < 0, """
                token \(disallowedTokenId) should stay masked inside the object
                """);
        }
        // The empty object {} is a complete JSON value: committing the close
        // brace leaves the end-of-sequence marker as the only legal token,
        // and committing it drives the matcher to its stop state.
        try compiledConstraint.commitToken(1);
        let completeObjectRow: MLXArray = try compiledConstraint.maskLogits(
            MLXArray(Array(repeating: Float(0.5), count: 8)));
        let completeObjectValues: Array<Float> = completeObjectRow.asArray(Float.self);
        #expect(completeObjectValues[7] == 0.5);
        let stillLegalCount: Int = completeObjectValues
            .filter { (maskedValue: Float) -> Bool in maskedValue == 0.5 }.count;
        #expect(stillLegalCount == 1);
        try compiledConstraint.commitToken(7);
        #expect(compiledConstraint.isTerminated());
    }

    @Test
    func should_allow_only_the_choice_literal_tokens_for_a_choice_constraint() throws {
        let grammarTokenizer: GrammarTokenizer = try GrammarTokenizer(
            vocab: Self.choiceVocabulary, vocabType: .raw, eosTokenId: 2);
        let compiledConstraint: Qwen35GuidedConstraint = Qwen35GuidedConstraint(
            grammarConstraint: try GrammarConstraint(
                tokenizer: grammarTokenizer,
                grammar: Qwen35GuidedConstraint.choiceGrammar(["What", "Two"]),
                rootRule: nil));
        let logitRow: MLXArray = MLXArray(Array(repeating: Float(0.5), count: 3));

        let maskedRow: MLXArray = try compiledConstraint.maskLogits(logitRow);
        let maskedValues: Array<Float> = maskedRow.asArray(Float.self);

        // Both whole-word literals are the grammar's first-token options and
        // the end marker is not.
        #expect(maskedValues[0] == 0.5);
        #expect(maskedValues[1] == 0.5);
        #expect(maskedValues[2].isInfinite && maskedValues[2] < 0);

        // The whole-word literal completes in one commit; the end marker
        // then becomes the only legal token, and committing it stops the
        // grammar.
        try compiledConstraint.commitToken(0);
        let completedRow: MLXArray = try compiledConstraint.maskLogits(
            MLXArray(Array(repeating: Float(0.5), count: 3)));
        let completedValues: Array<Float> = completedRow.asArray(Float.self);
        #expect(completedValues[2] == 0.5);
        let stillLegalCount: Int = completedValues
            .filter { (maskedValue: Float) -> Bool in maskedValue == 0.5 }.count;
        #expect(stillLegalCount == 1);
        try compiledConstraint.commitToken(2);
        #expect(compiledConstraint.isTerminated());
    }

    @Test
    func should_build_an_escaped_gbnf_alternation_for_choices() {
        #expect(Qwen35GuidedConstraint.choiceGrammar(["Romeo", "Juliet"])
            == "root ::= \"Romeo\" | \"Juliet\"");
        #expect(Qwen35GuidedConstraint.choiceGrammar(["say \"hi\""])
            == "root ::= \"say \\\"hi\\\"\"");
        #expect(Qwen35GuidedConstraint.choiceGrammar(["line\nbreak", "tab\there"])
            == "root ::= \"line\\nbreak\" | \"tab\\there\"");
    }

    @Test(.timeLimit(.minutes(1)))
    func should_compile_the_enforced_constraint_into_the_prepared_request() throws {
        let processor: Qwen35ChatProcessor = try Self.makeProcessor();
        var constrainedCommand: ChatGenerationCommand = Self.chatCommand(requestId: 151);
        constrainedCommand = ChatGenerationCommand(
            requestId: constrainedCommand.requestId, model: constrainedCommand.model,
            messages: constrainedCommand.messages, tools: constrainedCommand.tools,
            toolChoice: constrainedCommand.toolChoice, settings: constrainedCommand.settings,
            structuredGeneration: .choice(choices: ["What", "Two"]));

        let preparedGeneration: any ActiveChatGeneration = try processor.prepareChatGeneration(
            constrainedCommand);
        let preparedRequest: Qwen35PreparedInferenceRequest = try #require(
            preparedGeneration.inferenceRequest as? Qwen35PreparedInferenceRequest);

        #expect(preparedRequest.guidedConstraint != nil);
        // The template opens the thinking channel, so the constraint starts
        // dormant and the reasoning-end boundary token is resolved.
        #expect(preparedRequest.startsInsideThinking);
        #expect(preparedRequest.naturalReasoningEndTokenIds.isEmpty == false);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reject_the_unsupported_regex_constraint_with_the_bounded_reason() throws {
        let processor: Qwen35ChatProcessor = try Self.makeProcessor();
        var regexCommand: ChatGenerationCommand = Self.chatCommand(requestId: 152);
        regexCommand = ChatGenerationCommand(
            requestId: regexCommand.requestId, model: regexCommand.model,
            messages: regexCommand.messages, tools: regexCommand.tools,
            toolChoice: regexCommand.toolChoice, settings: regexCommand.settings,
            structuredGeneration: .regex(pattern: "Romeo|Juliet"));

        #expect(throws: ChatPreparationRejection.self) {
            _ = try processor.prepareChatGeneration(regexCommand);
        }
    }

    // MARK: Journey plumbing

    private static func makeProcessor() throws -> Qwen35ChatProcessor {
        let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildInMemoryRuntime(
            modelDirectory: Self.synthesizedModelDirectory().path,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                TinyDenseModelFixture.autoregressiveConfiguration()));
        guard let processor = runtime.processor as? Qwen35ChatProcessor else {
            Issue.record("expected the dense chat processor");
            throw WorkerHarness.harnessFailure("unexpected processor type");
        }
        return processor;
    }

    private static func synthesizedModelDirectory() throws -> URL {
        let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-guided-journey-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true);
        try TinyDenseModelFixture.writeModelDirectory(modelDirectoryUrl: modelDirectoryUrl);
        return modelDirectoryUrl;
    }

    private static func chatCommand(requestId: UInt64) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5",
            messages: [.user(content: "What is the play about?", images: Array())],
            tools: Array(),
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: 16, temperatureThousandths: nil, topPThousandths: nil,
                seed: nil, thinkingBudget: nil),
            structuredGeneration: nil);
    }
}
