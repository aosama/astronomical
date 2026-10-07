import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Marker-defect permutations: closed envelopes with a usable function name
/// salvage into tool calls (missing `<`, missing `>`, missing closes),
/// unclosed envelopes salvage at generation end, and nameless shapes stream
/// as text instead of aborting generation.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/marker_permutations.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserMarkerPermutationTests {

    private typealias Support = Qwen35OutputParserJourneySupport;

    private enum ClosedEnvelopeExpectation {
        case toolCall(functionName: String, argumentsJson: String);
        case visibleText;
    }

    private struct MarkerDefectCase {
        let markerDefect: String;
        let qwenEnvelope: String;
        let expectation: ClosedEnvelopeExpectation;
    }

    private struct UnclosedEnvelopeCase {
        let markerDefect: String;
        let qwenFragment: String;
        let expectation: ClosedEnvelopeExpectation;
    }

    @Test(.timeLimit(.minutes(1)))
    func should_honor_the_fail_open_contract_for_every_closed_qwen_marker_defect() throws {
        for closedEnvelopeCase in closedEnvelopeCases() {
            let outputParser = try Support.literaryOutputParser();
            let parseOutcome = try? outputParser.pushFragment(closedEnvelopeCase.qwenEnvelope);
            assertClosedEnvelopeOutcome(
                closedEnvelopeCase.markerDefect, closedEnvelopeCase.qwenEnvelope, parseOutcome,
                closedEnvelopeCase.expectation);
            #expect(
                outputParser.finish().isEmpty,
                "defect \(closedEnvelopeCase.markerDefect) left pending parser output");
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_forward_unclosed_qwen_envelopes_when_generation_finishes() throws {
        for unclosedEnvelopeCase in unclosedEnvelopeCases() {
            let outputParser = try Support.literaryOutputParser();
            let pushOutcome = try outputParser.pushFragment(unclosedEnvelopeCase.qwenFragment);
            #expect(
                qwen35ToolCalls(pushOutcome).isEmpty,
                "unclosed defect \(unclosedEnvelopeCase.markerDefect) emitted a tool call during push");
            let finishEvents = outputParser.finish();
            assertUnclosedFinishOutcome(
                unclosedEnvelopeCase.markerDefect, unclosedEnvelopeCase.qwenFragment, finishEvents,
                unclosedEnvelopeCase.expectation);
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_recover_tool_calls_that_omit_the_envelope_open() throws {
        // The model sometimes writes Qwen function tags without the surrounding
        // envelope. The call must still reach the harness, and any slop before
        // the function open streams as visible text.
        let declaredOpen = try Support.literaryOutputParser();
        let declaredEvents = try declaredOpen.pushFragment(
            "<function=\(Support.declaredCharacterFunction)><parameter=name>Romeo</parameter></function>\(Support.toolCallEnd)");
        #expect(declaredEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.declaredCharacterFunction,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);

        let undeclaredOpen = try Support.literaryOutputParser();
        let undeclaredEvents = try undeclaredOpen.pushFragment(
            "tool_call><function=\(Support.undeclaredFunctionName)><parameter=name>Romeo</parameter></function>\(Support.toolCallEnd)");
        #expect(undeclaredEvents == [
            .textDelta("tool_call>"),
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.undeclaredFunctionName,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_salvage_an_oversized_qwen_tool_call_fragment_without_aborting_generation() throws {
        let outputParser = try Support.literaryOutputParser();
        #expect(
            try outputParser.pushFragment(
                "\(Support.toolCallStart)<function=find_character><parameter=name>Romeo"
            ).isEmpty);
        let oversizedFragment = String(repeating: "R", count: 20 * 1024);
        let salvagedEvents = try outputParser.pushFragment(oversizedFragment);
        #expect(salvagedEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.declaredCharacterFunction,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);
        #expect(outputParser.finish().isEmpty);
    }

    private func closedEnvelopeCases() -> Array<MarkerDefectCase> {
        func closedCase(
            _ markerDefect: String, _ qwenFunctionBody: String,
            _ expectation: ClosedEnvelopeExpectation
        ) -> MarkerDefectCase {
            return MarkerDefectCase(
                markerDefect: markerDefect,
                qwenEnvelope: "\(Support.toolCallStart)\(qwenFunctionBody)\(Support.toolCallEnd)",
                expectation: expectation);
        }
        func toolCall(_ functionName: String, _ argumentsJson: String) -> ClosedEnvelopeExpectation {
            return .toolCall(functionName: functionName, argumentsJson: argumentsJson);
        }
        return [
            closedCase(
                "canonical declared call",
                "<function=find_character><parameter=name>Romeo</parameter></function>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "missing < on function open",
                "function=find_character><parameter=name>Romeo</parameter></function>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "missing < on parameter open",
                "<function=find_character>parameter=name>Romeo</parameter></function>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "missing < on function and parameter opens",
                "function=find_character>parameter=name>Romeo</parameter></function>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "missing > on function open",
                "<function=find_character\n<parameter=name>Romeo</parameter></function>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "missing function close",
                "<function=find_character><parameter=name>Romeo</parameter>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "missing parameter close",
                "<function=find_character><parameter=name>Romeo</function>",
                toolCall(Support.declaredCharacterFunction, Support.romeoArgumentsJson)),
            closedCase(
                "undeclared with missing < on parameter open",
                "<function=inspect_verse>\nparameter=name>Romeo</parameter></function>",
                toolCall(Support.undeclaredFunctionName, Support.romeoArgumentsJson)),
            closedCase(
                "undeclared name only",
                "<function=inspect_verse></function>",
                toolCall(Support.undeclaredFunctionName, Support.emptyArgumentsJson)),
            closedCase(
                "undeclared with _key slop then a well-formed value pair",
                "<function=read>\n<_key>argument-key</parameter><parameter=value>romeo-and-juliet.md</parameter></function>",
                toolCall("read", "{\"value\":\"romeo-and-juliet.md\"}")),
            closedCase(
                "empty function name",
                "<function=></function>",
                .visibleText),
            closedCase(
                "arguments without a function marker",
                "<parameter=name>Romeo</parameter>",
                .visibleText),
        ];
    }

    private func unclosedEnvelopeCases() -> Array<UnclosedEnvelopeCase> {
        return [
            UnclosedEnvelopeCase(
                markerDefect: "missing tool_call close",
                qwenFragment:
                    "\(Support.toolCallStart)<function=find_character><parameter=name>Romeo</parameter></function>",
                expectation: .toolCall(
                    functionName: Support.declaredCharacterFunction,
                    argumentsJson: Support.romeoArgumentsJson)),
            UnclosedEnvelopeCase(
                markerDefect: "missing < on tool_call close",
                qwenFragment:
                    "\(Support.toolCallStart)<function=find_character><parameter=name>Romeo</parameter></function>/tool_call>",
                expectation: .toolCall(
                    functionName: Support.declaredCharacterFunction,
                    argumentsJson: Support.romeoArgumentsJson)),
            UnclosedEnvelopeCase(
                markerDefect: "truncated tool_call close",
                qwenFragment:
                    "\(Support.toolCallStart)<function=inspect_verse><parameter=name>Romeo</parameter></function></tool_c",
                expectation: .toolCall(
                    functionName: Support.undeclaredFunctionName,
                    argumentsJson: Support.romeoArgumentsJson)),
            UnclosedEnvelopeCase(
                markerDefect: "unclosed undeclared call",
                qwenFragment: "\(Support.toolCallStart)<function=inspect_verse>",
                expectation: .toolCall(
                    functionName: Support.undeclaredFunctionName,
                    argumentsJson: Support.emptyArgumentsJson)),
            UnclosedEnvelopeCase(
                markerDefect: "nameless unclosed arguments",
                qwenFragment: "\(Support.toolCallStart)<parameter=name>Romeo</parameter>",
                expectation: .visibleText),
        ];
    }

    private func assertUnclosedFinishOutcome(
        _ markerDefect: String, _ qwenFragment: String,
        _ finishEvents: Array<Qwen35OutputEvent>, _ expectation: ClosedEnvelopeExpectation
    ) {
        if case .visibleText = expectation {
            #expect(
                qwen35ToolCalls(finishEvents).isEmpty,
                "defect=\(markerDefect); fragment=\(qwenFragment) emitted a tool call");
            #expect(
                !qwen35TextDeltas(finishEvents).filter { !$0.isEmpty }.isEmpty,
                "defect=\(markerDefect); fragment=\(qwenFragment) dropped nameless tool-call text");
            return;
        }
        assertClosedEnvelopeOutcome(markerDefect, qwenFragment, finishEvents, expectation);
    }

    private func assertClosedEnvelopeOutcome(
        _ markerDefect: String, _ qwenEnvelope: String,
        _ outputEvents: Array<Qwen35OutputEvent>?, _ expectation: ClosedEnvelopeExpectation
    ) {
        let failureContext = "defect=\(markerDefect); envelope=\(qwenEnvelope)";
        guard let outputEvents else {
            Issue.record("\(failureContext) aborted generation");
            return;
        }
        switch expectation {
        case let .toolCall(functionName, argumentsJson):
            #expect(outputEvents == [
                .toolCall(Qwen35ToolCall(
                    index: 0, functionName: functionName, argumentsJson: argumentsJson)),
            ], "\(failureContext)");
        case .visibleText:
            #expect(
                qwen35ToolCalls(outputEvents).isEmpty,
                "\(failureContext) emitted a tool call");
        }
    }
}
