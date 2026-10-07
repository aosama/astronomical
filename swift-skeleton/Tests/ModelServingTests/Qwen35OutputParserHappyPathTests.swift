import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Well-formed Qwen3.5 tool-call permutations: canonical, pretty-printed,
/// whitespace-jammed, undeclared, and typed-argument layouts all reach the
/// harness as tool calls, in source order.
/// Mirrors crates/model-serving/tests/qwen3_5_hermetic/output_parser/happy_path_permutations.rs.
@Suite(.tags(.hermeticJourney))
struct Qwen35OutputParserHappyPathTests {

    struct WellFormedCallCase {
        let layoutDescription: String;
        let qwenFunctionBody: String;
        let functionName: String;
        let argumentsJson: String;
    }

    @Test(.timeLimit(.minutes(1)))
    func should_emit_a_tool_call_for_every_well_formed_qwen_layout() throws {
        for wellFormedCall in wellFormedSingleCallCases() {
            let outputParser = try Qwen35OutputParserJourneySupport.literaryOutputParser();
            let toolCallXml =
                "\(Qwen35OutputParserJourneySupport.toolCallStart)\(wellFormedCall.qwenFunctionBody)\(Qwen35OutputParserJourneySupport.toolCallEnd)";
            let outputEvents = try outputParser.pushFragment(toolCallXml);
            #expect(outputEvents == [
                .toolCall(Qwen35ToolCall(
                    index: 0, functionName: wellFormedCall.functionName,
                    argumentsJson: wellFormedCall.argumentsJson)),
            ], "\(wellFormedCall.layoutDescription)");
            #expect(
                outputParser.finish().isEmpty,
                "\(wellFormedCall.layoutDescription)");
        }
    }

    @Test(.timeLimit(.minutes(1)))
    func should_emit_sequential_well_formed_qwen_tool_calls_in_source_order() throws {
        let outputParser = try Qwen35OutputParserJourneySupport.literaryOutputParser();
        let outputEvents = try outputParser.pushFragment(
            "They are central.\(Qwen35OutputParserJourneySupport.toolCallStart)<function=find_character><parameter=name>Romeo</parameter></function>\(Qwen35OutputParserJourneySupport.toolCallEnd)\(Qwen35OutputParserJourneySupport.toolCallStart)<function=summarize_scene><parameter=scene>balcony</parameter></function>\(Qwen35OutputParserJourneySupport.toolCallEnd)Then continue."
        );

        #expect(outputEvents == [
            .textDelta("They are central."),
            .toolCall(Qwen35ToolCall(
                index: 0,
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.romeoArgumentsJson)),
            .toolCall(Qwen35ToolCall(
                index: 1,
                functionName: Qwen35OutputParserJourneySupport.declaredSceneFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.balconyArgumentsJson)),
            .textDelta("Then continue."),
        ]);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_keep_qwen_argument_order_independent_for_required_and_extra_fields() throws {
        let extraAfterRequired = try Qwen35OutputParserJourneySupport.literaryOutputParser();
        let extraBeforeRequired = try Qwen35OutputParserJourneySupport.literaryOutputParser();
        let extraAfterRequiredEvents = try extraAfterRequired.pushFragment(
            "\(Qwen35OutputParserJourneySupport.toolCallStart)<function=find_character><parameter=name>Romeo</parameter><parameter=description>Locate the character</parameter></function>\(Qwen35OutputParserJourneySupport.toolCallEnd)");
        let extraBeforeRequiredEvents = try extraBeforeRequired.pushFragment(
            "\(Qwen35OutputParserJourneySupport.toolCallStart)<function=find_character><parameter=description>Locate the character</parameter><parameter=name>Romeo</parameter></function>\(Qwen35OutputParserJourneySupport.toolCallEnd)");

        let afterArgumentsJson = try #require(qwen35ToolCalls(extraAfterRequiredEvents).first?
            .argumentsJson);
        let beforeArgumentsJson = try #require(qwen35ToolCalls(extraBeforeRequiredEvents).first?
            .argumentsJson);
        let afterArguments = try #require(qwen35JsonArguments(afterArgumentsJson));
        let beforeArguments = try #require(qwen35JsonArguments(beforeArgumentsJson));
        #expect(afterArguments == beforeArguments);
        guard case let .object(afterEntries) = afterArguments else {
            Issue.record("expected parsed tool arguments");
            return;
        }
        #expect(afterEntries["name"] == .string(Qwen35OutputParserJourneySupport.characterName));
    }

    private func wellFormedSingleCallCases() -> Array<WellFormedCallCase> {
        let undeclared = Qwen35OutputParserJourneySupport.undeclaredFunctionName;
        return [
            WellFormedCallCase(
                layoutDescription: "jammed declared character call",
                qwenFunctionBody: "<function=find_character><parameter=name>Romeo</parameter></function>",
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.romeoArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "pretty-printed declared character call",
                qwenFunctionBody:
                    "\n<function=find_character>\n<parameter=name>\nRomeo\n</parameter>\n</function>\n",
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.romeoArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "spaces between Qwen tags",
                qwenFunctionBody:
                    "<function=find_character> <parameter=name>Romeo</parameter> </function>",
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.romeoArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "tabs between Qwen tags",
                qwenFunctionBody:
                    "<function=find_character>\t<parameter=name>Romeo</parameter></function>",
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.romeoArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "declared scene call",
                qwenFunctionBody:
                    "<function=summarize_scene><parameter=scene>balcony</parameter></function>",
                functionName: Qwen35OutputParserJourneySupport.declaredSceneFunction,
                argumentsJson: Qwen35OutputParserJourneySupport.balconyArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "empty required string argument",
                qwenFunctionBody: "<function=find_character><parameter=name></parameter></function>",
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: "{\"name\":\"\"}"),
            WellFormedCallCase(
                layoutDescription: "multi-word character name",
                qwenFunctionBody:
                    "<function=find_character><parameter=name>Romeo Montague</parameter></function>",
                functionName: Qwen35OutputParserJourneySupport.declaredCharacterFunction,
                argumentsJson: "{\"name\":\"Romeo Montague\"}"),
            WellFormedCallCase(
                layoutDescription: "path-shaped extra argument on an undeclared function",
                qwenFunctionBody:
                    "<function=read><parameter=path>romeo-and-juliet.md</parameter></function>",
                functionName: "read",
                argumentsJson: "{\"path\":\"romeo-and-juliet.md\"}"),
            WellFormedCallCase(
                layoutDescription: "hyphenated undeclared name with no arguments",
                qwenFunctionBody: "<function=repo-discovery-guide></function>",
                functionName: "repo-discovery-guide",
                argumentsJson: Qwen35OutputParserJourneySupport.emptyArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "undeclared function with a literary argument",
                qwenFunctionBody:
                    "<function=\(undeclared)><parameter=name>Romeo</parameter></function>",
                functionName: undeclared,
                argumentsJson: Qwen35OutputParserJourneySupport.romeoArgumentsJson),
            WellFormedCallCase(
                layoutDescription: "undeclared function with two arguments",
                qwenFunctionBody:
                    "<function=\(undeclared)><parameter=name>Romeo</parameter><parameter=scene>balcony</parameter></function>",
                functionName: undeclared,
                argumentsJson: "{\"name\":\"Romeo\",\"scene\":\"balcony\"}"),
            WellFormedCallCase(
                layoutDescription: "JSON array extra argument stays structured",
                qwenFunctionBody:
                    "<function=\(undeclared)><parameter=quotes>[\"O Romeo\",\"O Juliet\"]</parameter></function>",
                functionName: undeclared,
                argumentsJson: "{\"quotes\":[\"O Romeo\",\"O Juliet\"]}"),
            WellFormedCallCase(
                layoutDescription: "numeric extra argument stays a number",
                qwenFunctionBody: "<function=\(undeclared)><parameter=act>2</parameter></function>",
                functionName: undeclared,
                argumentsJson: "{\"act\":2}"),
            WellFormedCallCase(
                layoutDescription: "boolean extra argument stays a boolean",
                qwenFunctionBody:
                    "<function=\(undeclared)><parameter=tragic>true</parameter></function>",
                functionName: undeclared,
                argumentsJson: "{\"tragic\":true}"),
            WellFormedCallCase(
                layoutDescription: "scene name with punctuation",
                qwenFunctionBody:
                    "<function=summarize_scene><parameter=scene>Capulet's orchard</parameter></function>",
                functionName: Qwen35OutputParserJourneySupport.declaredSceneFunction,
                argumentsJson: "{\"scene\":\"Capulet's orchard\"}"),
        ];
    }
}

/// Minimal parsed-JSON view for order-independent argument comparisons.
enum Qwen35JourneyJsonValue: Equatable {
    case null;
    case boolean(Bool);
    case number(Double);
    case string(String);
    case array(Array<Qwen35JourneyJsonValue>);
    case object(Dictionary<String, Qwen35JourneyJsonValue>);
}

/// Parses a serialized tool-arguments document for structural comparison.
func qwen35JsonArguments(_ argumentsJson: String) -> Qwen35JourneyJsonValue? {
    guard
        let wireValue = try? JsonWireParser.parseDocument(documentBytes: Data(argumentsJson.utf8))
    else {
        return nil;
    }
    return qwen35JourneyValue(wireValue);
}

private func qwen35JourneyValue(_ wireValue: JsonWireValue) -> Qwen35JourneyJsonValue {
    switch wireValue {
    case .null:
        return .null;
    case let .boolean(booleanValue):
        return .boolean(booleanValue);
    case let .unsignedInteger(unsignedValue):
        return .number(Double(unsignedValue));
    case let .signedInteger(signedValue):
        return .number(Double(signedValue));
    case let .double(doubleValue):
        return .number(doubleValue);
    case let .float32(floatValue):
        return .number(Double(floatValue));
    case let .string(stringValue):
        return .string(stringValue);
    case let .array(entryValues):
        return .array(entryValues.map(qwen35JourneyValue));
    case let .object(wireObject):
        var objectEntries: Dictionary<String, Qwen35JourneyJsonValue> = [:];
        for entry in wireObject.entries {
            objectEntries[entry.key] = qwen35JourneyValue(entry.value);
        }
        return .object(objectEntries);
    }
}
