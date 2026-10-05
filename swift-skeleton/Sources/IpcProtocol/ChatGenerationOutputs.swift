import Foundation;

/// One ordered output emitted while structured chat generation is active.
/// Wire shape is internally tagged with `kind` in snake_case.
public enum ChatGenerationOutput: Equatable {
    /// Assistant-visible text with model control syntax removed.
    case text(text: String);
    /// Model reasoning kept separate from assistant-visible text.
    case reasoning(text: String);
    /// One complete validated function call.
    case toolCall(toolCallIndex: UInt16, functionName: String, argumentsJson: String);

    private static let expectedVariantNames: Array<String> = ["text", "reasoning", "tool_call"];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case let .text(text):
            var wireObject = ChatGenerationOutput.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("text"));
            wireObject.appendEntry(key: "text", value: .string(text));
            return .object(wireObject);
        case let .reasoning(text):
            var wireObject = ChatGenerationOutput.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("reasoning"));
            wireObject.appendEntry(key: "text", value: .string(text));
            return .object(wireObject);
        case let .toolCall(toolCallIndex, functionName, argumentsJson):
            var wireObject = ChatGenerationOutput.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("tool_call"));
            wireObject.appendEntry(key: "tool_call_index", value: .unsignedInteger(UInt64(toolCallIndex)));
            wireObject.appendEntry(key: "function_name", value: .string(functionName));
            wireObject.appendEntry(key: "arguments_json", value: .string(argumentsJson));
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatGenerationOutput {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: ChatGenerationOutput.expectedVariantNames) {
        case "text":
            let parsedOutput = ChatGenerationOutput.text(text: try wireObject.decodeString(fieldName: "text"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["text"]);
            return parsedOutput;
        case "reasoning":
            let parsedOutput = ChatGenerationOutput.reasoning(text: try wireObject.decodeString(fieldName: "text"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["text"]);
            return parsedOutput;
        default:
            let parsedOutput = ChatGenerationOutput.toolCall(
                toolCallIndex: try wireObject.decodeUInt16(fieldName: "tool_call_index"),
                functionName: try wireObject.decodeString(fieldName: "function_name"),
                argumentsJson: try wireObject.decodeString(fieldName: "arguments_json"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["tool_call_index", "function_name", "arguments_json"]);
            return parsedOutput;
        }
    }

    private static func emptyWireObject() -> JsonWireObject {
        return JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
    }
}
