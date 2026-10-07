import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Tool-call markers quoted inside model prose must not hijack the parser: a
/// coding model discussing tool syntax streams the prose as reasoning and
/// still delivers the real call that follows the think close.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/prose_marker_hijack.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserProseMarkerHijackTests {

    @Test(.timeLimit(.minutes(1)))
    func should_resync_when_quoted_prose_markers_hijack_the_tool_call_state() throws {
        let markers = Qwen35OutputParserJourneySupport.self;
        let parser = try Qwen35OutputParser(
            declaredTools: markers.literaryDeclaredTools(), startsInsideThinking: true);

        let reasoningProse =
            "The diffstat shows the consolidation. The normalizer rewrites markers like `<invoke name=...>` into the Qwen grammar before parsing. Let me verify the wiring.";
        var events = try parser.pushFragment(reasoningProse);

        let remainder =
            "\(markers.thinkEnd)\nNow let me isolate the shipped changes.\n\(markers.toolCallStart)\n<function=find_character>\n<parameter=name>Romeo</parameter>\n\(markers.toolCallEnd)\n";
        events += try parser.pushFragment(remainder);
        events += parser.finish();

        let toolCalls = qwen35ToolCalls(events);
        #expect(toolCalls.count == 1);
        #expect(toolCalls.first?.functionName == markers.declaredCharacterFunction);
        #expect(toolCalls.first?.argumentsJson == markers.romeoArgumentsJson);

        for textDelta in qwen35TextDeltas(events) {
            #expect(!textDelta.contains(markers.toolCallStart));
            #expect(!textDelta.contains(markers.thinkEnd));
        }
        let reasoningDeltas = events.compactMap { (event: Qwen35OutputEvent) -> String? in
            if case let .reasoningDelta(text) = event {
                return text;
            }
            return nil;
        };
        #expect(reasoningDeltas.joined().contains("<invoke name=...>"));
    }
}
