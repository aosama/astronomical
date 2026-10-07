import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

/// Shared Romeo-and-Juliet literary tool fixtures and Qwen3.5 marker
/// constants for the output-parser journeys, mirroring the Rust
/// `output_parser/support.rs` fixture set.
enum Qwen35OutputParserJourneySupport {

    static let declaredCharacterFunction = "find_character";
    static let declaredSceneFunction = "summarize_scene";
    static let undeclaredFunctionName = "inspect_verse";
    static let characterName = "Romeo";
    static let romeoArgumentsJson = "{\"name\":\"Romeo\"}";
    static let balconyArgumentsJson = "{\"scene\":\"balcony\"}";
    static let emptyArgumentsJson = "{}";

    static let thinkStart = "<think>";
    static let thinkEnd = "</think>";
    static let toolCallStart = "<tool_call>";
    static let toolCallEnd = "</tool_call>";

    /// Assembles marker literals from fragments so this source never spells a
    /// live foreign invoke block outside its assemble-site.
    static func invokeEnvelope(
        functionName: String, parameterName: String, parameterValue: String
    ) -> String {
        return "<invoke name=\"\(functionName)\">\n<parameter name=\"\(parameterName)\">\n\(parameterValue)\n</parameter>\n</function>\n</tool_call>";
    }

    /// The unclosed invoke shape that salvages at generation end.
    static func unclosedInvokeEnvelope(functionName: String, parameterName: String, parameterValue: String) -> String {
        return "<invoke name=\"\(functionName)\">\n<parameter name=\"\(parameterName)\">\n\(parameterValue)\n</parameter>";
    }

    /// Invoke body closed with `</invoke>` instead of the envelope close.
    static func invokeClosedWithInvokeEnd(
        functionName: String, parameterName: String, parameterValue: String
    ) -> String {
        return "<invoke name=\"\(functionName)\">\n<parameter name=\"\(parameterName)\">\n\(parameterValue)\n</parameter>\n</invoke>";
    }

    /// Bare Qwen function without the envelope, with trailing visible text.
    static func bareQwenFunction(
        functionName: String, parameterName: String, parameterValue: String, trailingText: String
    ) -> String {
        return "<function=\(functionName)>\n<parameter name=\"\(parameterName)\">\n\(parameterValue)\n</parameter>\n</function>\n\(trailingText)";
    }

    static func literaryDeclaredTools() -> Array<ChatToolDefinition> {
        return [
            ChatToolDefinition(
                name: declaredCharacterFunction,
                description: "Locate a character in Romeo and Juliet.",
                parametersJson:
                    "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}"),
            ChatToolDefinition(
                name: declaredSceneFunction,
                description: "Summarize a scene in Romeo and Juliet.",
                parametersJson:
                    "{\"type\":\"object\",\"properties\":{\"scene\":{\"type\":\"string\"}},\"required\":[\"scene\"]}"),
        ];
    }

    static func literaryOutputParser() throws -> Qwen35OutputParser {
        return try Qwen35OutputParser(declaredTools: literaryDeclaredTools());
    }
}

/// Extracts the tool-call events from a mixed event stream.
func qwen35ToolCalls(
    _ outputEvents: Array<Qwen35OutputEvent>
) -> Array<Qwen35ToolCall> {
    return outputEvents.compactMap { (outputEvent: Qwen35OutputEvent) -> Qwen35ToolCall? in
        if case let .toolCall(toolCall) = outputEvent {
            return toolCall;
        }
        return nil;
    };
}

/// Extracts the text-delta events from a mixed event stream.
func qwen35TextDeltas(_ outputEvents: Array<Qwen35OutputEvent>) -> Array<String> {
    return outputEvents.compactMap { (outputEvent: Qwen35OutputEvent) -> String? in
        if case let .textDelta(text) = outputEvent {
            return text;
        }
        return nil;
    };
}
