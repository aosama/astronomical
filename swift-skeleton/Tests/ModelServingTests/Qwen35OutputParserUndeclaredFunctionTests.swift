import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// A well-formed undeclared function reaches the harness as a tool call: the
/// harness owns unknown-tool rejection, and reasoning streamed before it still
/// flows. No model-visible correction is invented for undeclared names.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/undeclared_function.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserUndeclaredFunctionTests {

    @Test(.timeLimit(.minutes(1)))
    func should_forward_an_undeclared_function_as_a_tool_call_for_the_harness() throws {
        let declaredTools = [
            ChatToolDefinition(
                name: "open-brain_openbrain_recall",
                description: nil,
                parametersJson: "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}"),
            ChatToolDefinition(
                name: "bash",
                description: nil,
                parametersJson: "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}}}"),
        ];
        let outputParser = try Qwen35OutputParser(declaredTools: declaredTools);

        let markers = Qwen35OutputParserJourneySupport.self;
        let toolCallXml =
            "\(markers.thinkStart)I should search memory.\(markers.thinkEnd)\(markers.toolCallStart)\n<function=open_brain>\n<parameter=query>repo history\n</parameter>\n</function>\n\(markers.toolCallEnd)";
        let outputEvents = try outputParser.pushFragment(toolCallXml);

        #expect(outputEvents.contains(.reasoningDelta("I should search memory.")));
        let toolCalls = qwen35ToolCalls(outputEvents);
        #expect(toolCalls == [
            Qwen35ToolCall(index: 0, functionName: "open_brain", argumentsJson: "{\"query\":\"repo history\"}"),
        ]);
    }
}
