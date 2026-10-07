import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Permissive argument journeys: enum violations, wrong item types, nested
/// schema defects, extra properties, and failed coercions pass through to the
/// client (last value wins for duplicates) instead of killing the generation.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/permissive_arguments.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserPermissiveArgumentsTests {

    private let toolCallStart = Qwen35OutputParserJourneySupport.toolCallStart;
    private let toolCallEnd = Qwen35OutputParserJourneySupport.toolCallEnd;

    private func declaredTool(
        _ name: String, _ parametersJson: String
    ) -> Array<ChatToolDefinition> {
        return [ChatToolDefinition(name: name, description: nil, parametersJson: parametersJson)];
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pass_through_a_string_tool_parameter_outside_its_declared_enum() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: declaredTool(
                "search",
                "{\"type\":\"object\",\"properties\":{\"mode\":{\"type\":\"string\",\"enum\":[\"files\",\"content\"]}},\"required\":[\"mode\"]}"));

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=search><parameter=mode>network</parameter></function>\(toolCallEnd)");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "search", argumentsJson: "{\"mode\":\"network\"}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pass_through_an_array_tool_parameter_with_a_wrong_item_type() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: declaredTool(
                "inspect",
                "{\"type\":\"object\",\"properties\":{\"paths\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"paths\"]}"));

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=inspect><parameter=paths>[\"src/lib.rs\",7]</parameter></function>\(toolCallEnd)");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "inspect",
                argumentsJson: "{\"paths\":[\"src/lib.rs\",7]}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pass_through_a_nested_object_tool_parameter_missing_a_required_property() throws {
        let editTool = declaredTool(
            "edit",
            "{\"type\":\"object\",\"properties\":{\"change\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}},\"required\":[\"change\"]}");
        let outputParser = try Qwen35OutputParser(declaredTools: editTool);

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=edit><parameter=change>{}</parameter></function>\(toolCallEnd)");
        #expect(outputEvents == [
            .toolCall(Qwen35ToolCall(
                index: 0, functionName: "edit", argumentsJson: "{\"change\":{}}")),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pass_through_a_nested_object_tool_parameter_with_an_undeclared_property() throws {
        let editTool = declaredTool(
            "edit",
            "{\"type\":\"object\",\"properties\":{\"change\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}},\"required\":[\"change\"]}");
        let outputParser = try Qwen35OutputParser(declaredTools: editTool);

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=edit><parameter=change>{\"path\":\"src/lib.rs\",\"mode\":\"append\"}</parameter></function>\(toolCallEnd)");
        let argumentsJson = try #require(qwen35ToolCalls(outputEvents).first?.argumentsJson);
        let arguments = try #require(qwen35JsonArguments(argumentsJson));
        guard case let .object(changeObject)? = arguments.objectValue(forKey: "change") else {
            Issue.record("expected a change object, got \(arguments)");
            return;
        }
        #expect(changeObject["path"] == .string("src/lib.rs"));
        #expect(changeObject["mode"] == .string("append"));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pass_through_a_nested_object_tool_parameter_with_a_wrong_property_type() throws {
        let editTool = declaredTool(
            "edit",
            "{\"type\":\"object\",\"properties\":{\"change\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}},\"required\":[\"change\"]}");
        let outputParser = try Qwen35OutputParser(declaredTools: editTool);

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=edit><parameter=change>{\"path\":7}</parameter></function>\(toolCallEnd)");
        let argumentsJson = try #require(qwen35ToolCalls(outputEvents).first?.argumentsJson);
        let arguments = try #require(qwen35JsonArguments(argumentsJson));
        guard case let .object(changeObject)? = arguments.objectValue(forKey: "change") else {
            Issue.record("expected a change object, got \(arguments)");
            return;
        }
        #expect(changeObject["path"] == .number(7));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pass_through_tool_arguments_with_extra_object_properties_without_rejecting()
        throws
    {
        let outputParser = try Qwen35OutputParser(
            declaredTools: declaredTool(
                "open-brain_recall",
                "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"scope\":{\"type\":\"object\",\"properties\":{\"visibility\":{\"type\":\"string\"},\"project_only\":{\"type\":\"boolean\"},\"include_unconfirmed\":{\"type\":\"boolean\"},\"include_stale\":{\"type\":\"boolean\"}}}},\"required\":[\"query\"]}"));

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=open-brain_recall><parameter=query>astronomical recent work session context</parameter><parameter=scope>{\"recency_days\":7,\"include_unconfirmed\":false}</parameter></function>\(toolCallEnd)");
        let argumentsJson = try #require(qwen35ToolCalls(outputEvents).first?.argumentsJson);
        let arguments = try #require(qwen35JsonArguments(argumentsJson));
        guard case let .object(scopeObject)? = arguments.objectValue(forKey: "scope") else {
            Issue.record("expected a scope object, got \(arguments)");
            return;
        }
        #expect(
            arguments.objectValue(forKey: "query")
                == .string("astronomical recent work session context"));
        #expect(scopeObject["recency_days"] == .number(7));
        #expect(scopeObject["include_unconfirmed"] == .boolean(false));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_overwrite_duplicate_tool_parameters_instead_of_rejecting() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: declaredTool(
                "edit",
                "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}"));

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=edit><parameter=path>first.rs</parameter><parameter=path>second.rs</parameter></function>\(toolCallEnd)");
        let argumentsJson = try #require(qwen35ToolCalls(outputEvents).first?.argumentsJson);
        let arguments = try #require(qwen35JsonArguments(argumentsJson));
        #expect(arguments.objectValue(forKey: "path") == .string("second.rs"));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fall_back_to_string_when_type_parsing_fails_instead_of_rejecting() throws {
        let outputParser = try Qwen35OutputParser(
            declaredTools: declaredTool(
                "search",
                "{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"integer\"},\"enabled\":{\"type\":\"boolean\"},\"ratio\":{\"type\":\"number\"}}}"));

        let outputEvents = try outputParser.pushFragment(
            "\(toolCallStart)<function=search><parameter=count>not-a-number</parameter><parameter=enabled>maybe</parameter><parameter=ratio>undefined</parameter></function>\(toolCallEnd)");
        let argumentsJson = try #require(qwen35ToolCalls(outputEvents).first?.argumentsJson);
        let arguments = try #require(qwen35JsonArguments(argumentsJson));
        #expect(arguments.objectValue(forKey: "count") == .string("not-a-number"));
        #expect(arguments.objectValue(forKey: "enabled") == .string("maybe"));
        #expect(arguments.objectValue(forKey: "ratio") == .string("undefined"));
    }
}

extension Qwen35JourneyJsonValue {

    fileprivate func objectValue(forKey key: String) -> Qwen35JourneyJsonValue? {
        guard case let .object(entries) = self else {
            return nil;
        }
        return entries[key];
    }
}
