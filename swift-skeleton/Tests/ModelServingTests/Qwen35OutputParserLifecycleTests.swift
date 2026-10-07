import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Hermetic journeys for the tool-call lifecycle: a schema-validated call
/// emits after its closing marker, valid batches exceed sixteen calls, and a
/// large generated parameter crosses many fragment bounds while buffered.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/tool_call_lifecycle.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserLifecycleTests {

    @Test(.timeLimit(.minutes(1)))
    func should_emit_a_schema_validated_tool_call_after_its_closing_marker() throws {
        let declaredTools = [
            ChatToolDefinition(
                name: "glob",
                description: "List matching files.",
                parametersJson:
                    "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},\"required\":[\"pattern\"]}"),
        ];
        let outputParser = try Qwen35OutputParser(declaredTools: declaredTools);

        let toolCallXml =
            "\(Qwen35OutputParserJourneySupport.toolCallStart)\n<function=glob>\n<parameter=pattern>\nsrc/**/*.rs\n</parameter>\n</function>\n\(Qwen35OutputParserJourneySupport.toolCallEnd)";
        let outputEvents = try outputParser.pushFragment(toolCallXml);

        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "glob", argumentsJson: "{\"pattern\":\"src/**/*.rs\"}")),
        ]);
        #expect(outputParser.finish().isEmpty);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_more_than_sixteen_tool_calls_when_the_model_emits_a_valid_batch() throws {
        let declaredTools = [
            ChatToolDefinition(
                name: "glob",
                description: "List matching files.",
                parametersJson:
                    "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},\"required\":[\"pattern\"]}"),
        ];
        let outputParser = try Qwen35OutputParser(declaredTools: declaredTools);
        let toolCallXml = (0..<17).map { (toolCallNumber: Int) -> String in
            return
                "\(Qwen35OutputParserJourneySupport.toolCallStart)<function=glob><parameter=pattern>src/\(toolCallNumber).rs</parameter></function>\(Qwen35OutputParserJourneySupport.toolCallEnd)";
        }.joined();

        let outputEvents = try outputParser.pushFragment(toolCallXml);

        #expect(outputEvents.count == 17);
        #expect(outputEvents.last == .toolCall(Qwen35ToolCall(
            index: 16, functionName: "glob", argumentsJson: "{\"pattern\":\"src/16.rs\"}")));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_a_large_generated_tool_parameter_when_the_output_frame_fits() throws {
        let declaredTools = [
            ChatToolDefinition(
                name: "edit",
                description: "Apply a generated patch.",
                parametersJson:
                    "{\"type\":\"object\",\"properties\":{\"patch\":{\"type\":\"string\"}},\"required\":[\"patch\"]}"),
        ];
        let outputParser = try Qwen35OutputParser(declaredTools: declaredTools);
        let largeGeneratedPatch = String(repeating: "x", count: 160 * 1024);

        _ = try outputParser.pushFragment(
            "\(Qwen35OutputParserJourneySupport.toolCallStart)<function=edit><parameter=patch>");
        for chunkStart in stride(from: 0, to: largeGeneratedPatch.utf8.count, by: 4 * 1024) {
            let chunkEnd = min(chunkStart + 4 * 1024, largeGeneratedPatch.utf8.count);
            let generatedPatchFragment = String(
                decoding: Array(largeGeneratedPatch.utf8)[chunkStart..<chunkEnd], as: UTF8.self);
            _ = try outputParser.pushFragment(generatedPatchFragment);
        }
        let outputEvents = try outputParser.pushFragment(
            "</parameter></function>\(Qwen35OutputParserJourneySupport.toolCallEnd)");

        let toolCalls = qwen35ToolCalls(outputEvents);
        #expect(toolCalls.count == 1);
        #expect(toolCalls.first?.functionName == "edit");
        let patchJsonFragment = "\"patch\":\"\(largeGeneratedPatch)\"";
        #expect(toolCalls.first?.argumentsJson == "{\(patchJsonFragment)}");
    }
}
