// ResponsesInput+FunctionCalls.swift — RestContract
//
// Function-call input items of crates/rest-contract/src/openai_responses_input.rs:
// replayed function calls and their outputs.

import Foundation;
import IpcProtocol;

/// One replayed function call issued by a prior assistant turn.
public struct OpenAiResponseFunctionCallInput: Equatable {
    private let callIdentifier: String;
    private let functionName: String;
    private let argumentsJsonText: String;

    fileprivate init(callIdentifier: String, functionName: String, argumentsJsonText: String) {
        self.callIdentifier = callIdentifier;
        self.functionName = functionName;
        self.argumentsJsonText = argumentsJsonText;
    }

    /// The call identifier correlating this call with its later output item.
    internal var callId: String {
        return self.callIdentifier;
    }

    /// The called function name.
    internal var name: String {
        return self.functionName;
    }

    /// The JSON-encoded function arguments.
    internal var argumentsJson: String {
        return self.argumentsJsonText;
    }

    /// Mirrors the serde derive with `deny_unknown_fields`; the `type` key is
    /// required to spell `function_call`.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseFunctionCallInput {
        let callObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try callObject.rejectUnknownFields(
            allowedFieldNames: ["type", "id", "call_id", "name", "arguments", "status"]);
        var decodedFunctionCallType: Bool = false;
        var decodedItemIdentifier: String? = nil;
        var decodedItemStatus: OpenAiResponseItemStatus? = nil;
        var decodedCallIdentifier: String? = nil;
        var decodedFunctionName: String? = nil;
        var decodedArgumentsJsonText: String? = nil;
        for entry in callObject.entries {
            switch entry.key {
            case "type":
                if decodedFunctionCallType {
                    throw JsonWireProblem.duplicateField(fieldName: "type");
                }
                decodedFunctionCallType = true;
                _ = try OpenAiResponseFunctionCallType.decoded(wireValue: entry.value);
            case "id":
                if decodedItemIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "id");
                }
                if entry.value.isNull == false {
                    decodedItemIdentifier = try JsonWireValue.extractString(entry.value);
                }
            case "status":
                if decodedItemStatus != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "status");
                }
                if entry.value.isNull == false {
                    decodedItemStatus = try OpenAiResponseItemStatus.decoded(wireValue: entry.value);
                }
            case "call_id":
                if decodedCallIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "call_id");
                }
                decodedCallIdentifier = try JsonWireValue.extractString(entry.value);
            case "name":
                if decodedFunctionName != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "name");
                }
                decodedFunctionName = try JsonWireValue.extractString(entry.value);
            case "arguments":
                if decodedArgumentsJsonText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "arguments");
                }
                decodedArgumentsJsonText = try JsonWireValue.extractString(entry.value);
            default:
                throw JsonWireProblem.unknownField(
                    fieldName: entry.key,
                    expectedFields: ["type", "id", "call_id", "name", "arguments", "status"]);
            }
        }
        guard let resolvedCallIdentifier: String = decodedCallIdentifier else {
            throw JsonWireProblem.missingField(fieldName: "call_id");
        }
        guard let resolvedFunctionName: String = decodedFunctionName else {
            throw JsonWireProblem.missingField(fieldName: "name");
        }
        guard let resolvedArgumentsJsonText: String = decodedArgumentsJsonText else {
            throw JsonWireProblem.missingField(fieldName: "arguments");
        }
        return OpenAiResponseFunctionCallInput(
            callIdentifier: resolvedCallIdentifier,
            functionName: resolvedFunctionName,
            argumentsJsonText: resolvedArgumentsJsonText);
    }
}

/// Recognized `type` spelling of a Responses function-call input item.
internal enum OpenAiResponseFunctionCallType: Equatable {
    case functionCall;

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseFunctionCallType {
        _ = try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["function_call"]);
        return .functionCall;
    }
}

/// One replayed function output returned for a prior function call.
public struct OpenAiResponseFunctionCallOutputInput: Equatable {
    private let callIdentifier: String;
    private let outputText: String;

    fileprivate init(callIdentifier: String, outputText: String) {
        self.callIdentifier = callIdentifier;
        self.outputText = outputText;
    }

    /// The call identifier this output answers.
    internal var callId: String {
        return self.callIdentifier;
    }

    /// The function output payload.
    internal var output: String {
        return self.outputText;
    }

    /// Mirrors the serde derive with `deny_unknown_fields`; the `type` key is
    /// required to spell `function_call_output`.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseFunctionCallOutputInput {
        let outputObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try outputObject.rejectUnknownFields(
            allowedFieldNames: ["type", "id", "call_id", "output", "status"]);
        var decodedFunctionCallOutputType: Bool = false;
        var decodedItemIdentifier: String? = nil;
        var decodedItemStatus: OpenAiResponseItemStatus? = nil;
        var decodedCallIdentifier: String? = nil;
        var decodedOutputText: String? = nil;
        for entry in outputObject.entries {
            switch entry.key {
            case "type":
                if decodedFunctionCallOutputType {
                    throw JsonWireProblem.duplicateField(fieldName: "type");
                }
                decodedFunctionCallOutputType = true;
                _ = try OpenAiResponseFunctionCallOutputType.decoded(wireValue: entry.value);
            case "id":
                if decodedItemIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "id");
                }
                if entry.value.isNull == false {
                    decodedItemIdentifier = try JsonWireValue.extractString(entry.value);
                }
            case "status":
                if decodedItemStatus != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "status");
                }
                if entry.value.isNull == false {
                    decodedItemStatus = try OpenAiResponseItemStatus.decoded(wireValue: entry.value);
                }
            case "call_id":
                if decodedCallIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "call_id");
                }
                decodedCallIdentifier = try JsonWireValue.extractString(entry.value);
            case "output":
                if decodedOutputText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "output");
                }
                decodedOutputText = try JsonWireValue.extractString(entry.value);
            default:
                throw JsonWireProblem.unknownField(
                    fieldName: entry.key,
                    expectedFields: ["type", "id", "call_id", "output", "status"]);
            }
        }
        guard let resolvedCallIdentifier: String = decodedCallIdentifier else {
            throw JsonWireProblem.missingField(fieldName: "call_id");
        }
        guard let resolvedOutputText: String = decodedOutputText else {
            throw JsonWireProblem.missingField(fieldName: "output");
        }
        return OpenAiResponseFunctionCallOutputInput(
            callIdentifier: resolvedCallIdentifier,
            outputText: resolvedOutputText);
    }
}

/// Recognized `type` spelling of a Responses function-call-output input item.
internal enum OpenAiResponseFunctionCallOutputType: Equatable {
    case functionCallOutput;

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseFunctionCallOutputType {
        _ = try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["function_call_output"]);
        return .functionCallOutput;
    }
}

/// Lifecycle status accepted on replayed input items.
///
/// The Rust contract declares this enum twice with identical variants (a
/// private input-side copy and a public response-side copy); one Swift module
/// cannot hold two same-named top-level types, so both sides share this one.
public enum OpenAiResponseItemStatus: Equatable {
    case inProgress;
    case completed;
    case incomplete;

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseItemStatus {
        switch try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["in_progress", "completed", "incomplete"]) {
        case "in_progress": return .inProgress;
        case "completed": return .completed;
        default: return .incomplete;
        }
    }

    /// Serializes with the serde `rename_all = "snake_case"` spelling.
    public func wireValue() -> JsonWireValue {
        switch self {
        case .inProgress: return .string("in_progress");
        case .completed: return .string("completed");
        case .incomplete: return .string("incomplete");
        }
    }
}
