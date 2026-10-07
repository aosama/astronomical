import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Fail-open coverage for non-canonical tool-call dialects and stream-end
/// flushes: Claude-style invoke envelopes, bare Qwen functions, partial
/// marker prefixes, oversized text frames, and nameless invoke bodies reach
/// the harness as tool calls or visible text; generation never aborts.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/foreign_dialect.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserForeignDialectTests {

    private typealias Support = Qwen35OutputParserJourneySupport;

    @Test(.timeLimit(.minutes(1)))
    func should_parse_a_claude_style_invoke_envelope_as_a_tool_call() throws {
        let outputParser = try Support.literaryOutputParser();
        let outputEvents = try outputParser.pushFragment(
            Support.invokeEnvelope(
                functionName: Support.declaredCharacterFunction, parameterName: "name",
                parameterValue: "Romeo"));

        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.declaredCharacterFunction,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);
        #expect(outputParser.finish().isEmpty);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_parse_a_bare_qwen_function_and_resume_visible_text() throws {
        let outputParser = try Support.literaryOutputParser();
        let outputEvents = try outputParser.pushFragment(
            Support.bareQwenFunction(
                functionName: Support.declaredSceneFunction, parameterName: "scene",
                parameterValue: "balcony", trailingText: "Juliet waits below."));

        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.declaredSceneFunction,
                argumentsJson: Support.balconyArgumentsJson)),
            .textDelta("\nJuliet waits below."),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_salvage_an_unclosed_invoke_envelope_when_generation_finishes() throws {
        let outputParser = try Support.literaryOutputParser();
        let unclosedInvoke = Support.unclosedInvokeEnvelope(
            functionName: Support.undeclaredFunctionName, parameterName: "name",
            parameterValue: "Romeo");
        #expect(try outputParser.pushFragment(unclosedInvoke).isEmpty);
        let finishEvents = outputParser.finish();
        #expect(finishEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.undeclaredFunctionName,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_flush_partial_marker_prefixes_as_text_when_generation_ends() throws {
        for partialMarker in ["<", "<t", "<inv", "<tool", "<func"] {
            let outputParser = try Support.literaryOutputParser();
            let fragment = "Romeo seeks the friar. \(partialMarker)";
            let pushEvents = try outputParser.pushFragment(fragment);
            let finishEvents = outputParser.finish();
            let allEvents = pushEvents + finishEvents;
            var streamedText = "";
            for outputEvent in allEvents {
                guard case let .textDelta(text) = outputEvent else {
                    Issue.record(
                        "partial marker \(partialMarker) emitted a non-text event: \(outputEvent)");
                    continue;
                }
                streamedText += text;
            }
            #expect(
                streamedText == fragment,
                "partial marker \(partialMarker) must flush verbatim instead of aborting");
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_flush_an_oversized_text_frame_instead_of_aborting_generation() throws {
        let outputParser = try Support.literaryOutputParser();
        let oversizedVerse = String(repeating: "Romeo ", count: 4 * 1024) + "<";
        #expect(oversizedVerse.utf8.count > 16 * 1024);
        let outputEvents = try outputParser.pushFragment(oversizedVerse);
        #expect(outputEvents == [.textDelta(oversizedVerse)]);
        #expect(outputParser.finish().isEmpty);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_keep_streaming_text_after_the_pending_cap_is_crossed() throws {
        let outputParser = try Support.literaryOutputParser();
        let verseBlock = String(repeating: "Juliet ", count: 8 * 1024);
        var streamedBytes = 0;
        for _ in 0..<20 {
            let outputEvents = try outputParser.pushFragment(verseBlock);
            for outputEvent in outputEvents {
                guard case let .textDelta(text) = outputEvent else {
                    Issue.record("unexpected event while flushing: \(outputEvent)");
                    continue;
                }
                streamedBytes += text.utf8.count;
            }
        }
        let finishEvents = outputParser.finish();
        for outputEvent in finishEvents {
            guard case let .textDelta(text) = outputEvent else {
                Issue.record("unexpected finish event: \(outputEvent)");
                continue;
            }
            streamedBytes += text.utf8.count;
        }
        #expect(
            streamedBytes == verseBlock.utf8.count * 20,
            "no generated text may be dropped while flushing");
    }

    @Test(.timeLimit(.minutes(1)))
    func should_forward_a_nameless_invoke_body_as_visible_text_not_an_abort() throws {
        let outputParser = try Support.literaryOutputParser();
        let namelessInvoke =
            "\(Support.toolCallStart)<invoke name=\"\">\n<parameter name=\"name\">\nRomeo\n</parameter>\n</function>\n\(Support.toolCallEnd)";
        let outputEvents = try outputParser.pushFragment(namelessInvoke);
        #expect(qwen35ToolCalls(outputEvents).isEmpty);
        #expect(outputParser.finish().isEmpty);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_parse_an_invoke_block_that_closes_with_the_invoke_end_tag() throws {
        let outputParser = try Support.literaryOutputParser();
        let outputEvents = try outputParser.pushFragment(
            Support.invokeClosedWithInvokeEnd(
                functionName: Support.declaredCharacterFunction, parameterName: "name",
                parameterValue: "Romeo"));
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.declaredCharacterFunction,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_promote_an_invoke_inside_prompt_opened_reasoning_to_a_tool_call() throws {
        let parser = try Qwen35OutputParser(
            declaredTools: Support.literaryDeclaredTools(), startsInsideThinking: true);
        let fragment =
            "Romeo waits on the balcony.\n\(Support.invokeClosedWithInvokeEnd(functionName: Support.declaredCharacterFunction, parameterName: "name", parameterValue: "Romeo"))";
        let outputEvents = try parser.pushFragment(fragment);
        #expect(outputEvents == [
            .reasoningDelta("Romeo waits on the balcony.\n"),
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: Support.declaredCharacterFunction,
                argumentsJson: Support.romeoArgumentsJson)),
        ]);
    }
}
