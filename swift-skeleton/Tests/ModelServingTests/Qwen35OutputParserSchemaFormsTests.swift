import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Declared schema-form journeys: nullable type lists, nullable `anyOf`
/// unions with constraint keywords, multi-branch unions, unresolvable unions
/// that degrade to dynamic parsing, schema rejection messages that name the
/// offending tool and property, and deep declared schemas accepted without
/// recursive validation.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/schema_forms.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserSchemaFormsTests {

    private let toolCallStart = Qwen35OutputParserJourneySupport.toolCallStart;
    private let toolCallEnd = Qwen35OutputParserJourneySupport.toolCallEnd;

    private func pushSingleCall(
        _ outputParser: Qwen35OutputParser, functionName: String, parameter: String
    ) throws -> Array<Qwen35OutputEvent> {
        return try outputParser.pushFragment(
            "\(toolCallStart)<function=\(functionName)><parameter=\(parameter)</parameter></function>\(toolCallEnd)");
    }

    @Test(.timeLimit(.minutes(1)))
    func should_emit_null_for_a_nullable_string_tool_parameter() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "read", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":[\"string\",\"null\"]}},\"required\":[\"path\"]}"),
            ]);
        let outputEvents = try pushSingleCall(outputParser, functionName: "read", parameter: "path>null");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(index: 0, functionName: "read", argumentsJson: "{\"path\":null}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_emit_null_for_an_any_of_nullable_string_tool_parameter() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "recall", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"project_id\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}}}"),
            ]);
        let outputEvents = try pushSingleCall(
            outputParser, functionName: "recall", parameter: "project_id>null");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "recall", argumentsJson: "{\"project_id\":null}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_an_opencode_constrained_nullable_integer_tool_parameter() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "recall", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"recency_days\":{\"anyOf\":[{\"exclusiveMinimum\":0,\"maximum\":9007199254740991,\"type\":\"integer\"},{\"type\":\"null\"}]}}}"),
            ]);
        let outputEvents = try pushSingleCall(
            outputParser, functionName: "recall", parameter: "recency_days>7");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "recall", argumentsJson: "{\"recency_days\":7}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_an_open_brain_nullable_string_with_minlength_maxlength_constraints()
        throws
    {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "open-brain_recall", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"project_id\":{\"anyOf\":[{\"type\":\"string\",\"minLength\":1,\"maxLength\":200},{\"type\":\"null\"}]}}}"),
            ]);
        let outputEvents = try pushSingleCall(
            outputParser, functionName: "open-brain_recall", parameter: "project_id>ob1-staging");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "open-brain_recall",
                argumentsJson: "{\"project_id\":\"ob1-staging\"}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_copilot_string_or_array_tool_parameters() throws {
        for (parameterText, expectedArgumentsJson) in [
            ("src", "{\"paths\":\"src\"}"),
            ("[\"src\",\"tests\"]", "{\"paths\":[\"src\",\"tests\"]}"),
        ] {
            let outputParser = try Qwen35OutputParser(
                declaredTools: [
                    ChatToolDefinition(
                        name: "grep", description: nil,
                        parametersJson:
                            "{\"type\":\"object\",\"properties\":{\"paths\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"array\",\"items\":{\"type\":\"string\"}}]}}}"),
                ]);
            let outputEvents = try pushSingleCall(
                outputParser, functionName: "grep", parameter: "paths>\(parameterText)");
            #expect(outputEvents == [
                .toolCall(Qwen35ToolCall(
                    index: 0, functionName: "grep", argumentsJson: expectedArgumentsJson)),
            ]);
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_copilot_opaque_canvas_action_parameters() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "invoke_canvas_action", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"input\":{\"description\":\"Action input matching the action input schema\"}},\"required\":[\"input\"]}"),
            ]);
        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=invoke_canvas_action><parameter=input>{\"action\":\"zoom\"}</parameter></function>\(toolCallEnd)");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "invoke_canvas_action",
                argumentsJson: "{\"input\":{\"action\":\"zoom\"}}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_a_multi_branch_string_any_of_tool_parameter() throws {
        // A client tool inventory may declare an enum-like parameter as more
        // than two anyOf branches; only the branch types affect argument
        // parsing, so the count must not reject the complete chat thread.
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "xcode_build", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"includeBuildLog\":{\"anyOf\":[{\"type\":\"string\",\"const\":\"onFailure\"},{\"type\":\"string\",\"const\":\"always\"},{\"type\":\"string\",\"const\":\"never\"}]}}}"),
            ]);
        let outputEvents = try pushSingleCall(
            outputParser, functionName: "xcode_build", parameter: "includeBuildLog>always");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "xcode_build",
                argumentsJson: "{\"includeBuildLog\":\"always\"}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_a_multi_branch_any_of_with_a_null_branch() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "recall", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"limit\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"},{\"type\":\"null\"}]}}}"),
            ]);
        let outputEvents = try pushSingleCall(
            outputParser, functionName: "recall", parameter: "limit>7");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(index: 0, functionName: "recall", argumentsJson: "{\"limit\":7}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_name_the_offending_tool_and_property_when_a_declared_schema_is_rejected() throws {
        var rejectionMessage = "";
        do {
            _ = try Qwen35OutputParser(
                declaredTools: [
                    ChatToolDefinition(
                        name: "broken_tool", description: nil,
                        parametersJson: "{\"type\":\"object\",\"properties\":{\"flag\":{\"type\":42}}}"),
                ]);
            Issue.record("an unsupported property type declaration must be rejected");
        } catch let parserError as Qwen35OutputParserError {
            rejectionMessage = parserError.description;
        }
        #expect(rejectionMessage.contains("broken_tool") && rejectionMessage.contains("flag"));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_tolerate_unresolvable_any_of_in_a_declared_tool_property() throws {
        // Real harness inventories declare properties whose anyOf no resolver
        // can reduce to a single coercion type: branches without a type member,
        // single-branch unions, type-list shorthands, and non-array anyOf
        // values. These degrade to dynamic JSON parsing instead of rejecting
        // the complete chat thread (#771).
        let unresolvableShapes: Array<(String, String, String)> = [
            ("{\"anyOf\":[{\"enum\":[\"agent\",\"team\",\"workspace\"]}]}", "workspace", "{\"agent_scope\":\"workspace\"}"),
            ("{\"anyOf\":[\"string\",\"null\"]}", "agent", "{\"agent_scope\":\"agent\"}"),
            ("{\"anyOf\":{\"type\":\"string\"}}", "team", "{\"agent_scope\":\"team\"}"),
        ];
        for (propertySchema, parameterValue, expectedArgumentsJson) in unresolvableShapes {
            let outputParser = try Qwen35OutputParser(
                declaredTools: [
                    ChatToolDefinition(
                        name: "save_workflow", description: nil,
                        parametersJson:
                            "{\"type\":\"object\",\"properties\":{\"agent_scope\":\(propertySchema)}}"),
                ]);
            let outputEvents = try pushSingleCall(
                outputParser, functionName: "save_workflow",
                parameter: "agent_scope>\(parameterValue)");
            #expect(outputEvents == [
                .toolCall(Qwen35ToolCall(
                    index: 0, functionName: "save_workflow",
                    argumentsJson: expectedArgumentsJson)),
            ]);
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_accept_a_declared_tool_schema_deeper_than_the_previous_parser_limit() throws {
        var nestedSchema = "{\"type\":\"string\"}";
        for _ in 0..<10 {
            nestedSchema =
                "{\"type\":\"object\",\"properties\":{\"child\":\(nestedSchema)},\"required\":[\"child\"]}";
        }
        _ = try Qwen35OutputParser(
            declaredTools: [
                ChatToolDefinition(
                    name: "deep", description: nil,
                    parametersJson:
                        "{\"type\":\"object\",\"properties\":{\"root\":\(nestedSchema)},\"required\":[\"root\"]}"),
            ]);
    }
}
