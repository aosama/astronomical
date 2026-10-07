import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Reasoning-channel journeys: think markers split the channels, prompt-opened
/// reasoning continues, late thinking blocks suppress, generation may stop
/// inside reasoning, and split control markers buffer across fragments.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/reasoning.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserReasoningTests {

    private let markers = Qwen35OutputParserJourneySupport.self;

    @Test(.timeLimit(.minutes(1)))
    func should_emit_reasoning_and_text_without_qwen3_5_marker_syntax() throws {
        let outputParser = try Qwen35OutputParser(declaredTools: []);

        #expect(
            try outputParser.pushFragment("\(markers.thinkStart)inspect the source")
                == [.reasoningDelta("inspect the source")]);
        #expect(
            try outputParser.pushFragment("\(markers.thinkEnd)then edit it.")
                == [.textDelta("then edit it.")]);
        #expect(outputParser.finish().isEmpty);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_continue_reasoning_already_opened_by_the_generation_prompt() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [], startsInsideThinking: true);

        #expect(
            try outputParser.pushFragment("Inspect first\(markers.thinkEnd)Then answer.")
                == [.reasoningDelta("Inspect first"), .textDelta("Then answer.")]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_suppress_late_thinking_blocks_after_visible_text_has_started() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [], startsInsideThinking: true);

        #expect(
            try outputParser.pushFragment(
                "Initial plan\(markers.thinkEnd)Visible answer.\(markers.thinkStart)late private thought\(markers.thinkEnd)Done."
            )
                == [
                    .reasoningDelta("Initial plan"),
                    .textDelta("Visible answer."),
                    .textDelta("Done."),
                ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_finish_with_streamed_reasoning_when_generation_stops_before_the_thinking_end_marker()
        throws
    {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [], startsInsideThinking: true);

        #expect(
            try outputParser.pushFragment("Thinking through the short answer")
                == [.reasoningDelta("Thinking through the short answer")]);
        #expect(outputParser.finish().isEmpty);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_buffer_control_markers_split_across_token_fragments() throws {
        let outputParser = try Qwen35OutputParser(declaredTools: []);

        #expect(try outputParser.pushFragment("<thi").isEmpty);
        #expect(
            try outputParser.pushFragment("nk>plan</thi") == [.reasoningDelta("plan")]);
        #expect(try outputParser.pushFragment("nk>answer") == [.textDelta("answer")]);
    }
}
