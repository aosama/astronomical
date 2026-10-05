import Foundation;

/// A bounded request-scoped failure that leaves the worker process responsive.
/// Wire shape is serde's externally tagged enum: unit variants serialize as
/// plain strings and struct variants as single-entry objects.
public enum ChatGenerationFailureReason: Equatable {
    /// The worker independently rejected malformed structured chat input.
    case invalidRequest(reason: String);
    /// A fatal model-execution failure reported before the worker exits.
    /// The reason is bounded and safe for the local API; native details stay in logs.
    case fatalExecution(reason: String);
    /// Prompt plus requested output exceeds the model-native context window.
    case contextLengthExceeded(actualTotalContextTokens: UInt32, maximumContextTokens: UInt32);
    /// A different generation request already owns the worker's bounded capacity.
    case engineBusy;
    /// Generated tokens could not be decoded or parsed into the declared output contract.
    case malformedModelOutput;

    private static let expectedVariantNames: Array<String> = [
        "invalid_request", "fatal_execution", "context_length_exceeded", "engine_busy", "malformed_model_output",
    ];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case let .invalidRequest(reason):
            return .object(ChatGenerationFailureReason.singleEntryWireObject(variantName: "invalid_request", payloadWireValue: ChatGenerationFailureReason.reasonObjectWireValue(reason)));
        case let .fatalExecution(reason):
            return .object(ChatGenerationFailureReason.singleEntryWireObject(variantName: "fatal_execution", payloadWireValue: ChatGenerationFailureReason.reasonObjectWireValue(reason)));
        case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens):
            var payloadObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
            payloadObject.appendEntry(key: "actual_total_context_tokens", value: .unsignedInteger(UInt64(actualTotalContextTokens)));
            payloadObject.appendEntry(key: "maximum_context_tokens", value: .unsignedInteger(UInt64(maximumContextTokens)));
            return .object(ChatGenerationFailureReason.singleEntryWireObject(variantName: "context_length_exceeded", payloadWireValue: .object(payloadObject)));
        case .engineBusy:
            return .string("engine_busy");
        case .malformedModelOutput:
            return .string("malformed_model_output");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatGenerationFailureReason {
        switch wireValue {
        case let .string(variantName):
            return try ChatGenerationFailureReason.unitVariant(variantName: variantName);
        case let .object(variantObject):
            return try ChatGenerationFailureReason.structVariant(variantObject: variantObject);
        default:
            throw JsonWireProblem.invalidType(expectedTypeName: "enum ChatGenerationFailureReason", found: wireValue.foundDescription);
        }
    }

    private static func unitVariant(variantName: String) throws -> ChatGenerationFailureReason {
        switch variantName {
        case "engine_busy": return .engineBusy;
        case "malformed_model_output": return .malformedModelOutput;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(ChatGenerationFailureReason.expectedVariantNames))");
        }
    }

    private static func structVariant(variantObject: JsonWireObject) throws -> ChatGenerationFailureReason {
        guard variantObject.entries.count == 1, let singleEntry = variantObject.entries.first else {
            throw JsonWireProblem.malformedDocument(problem: "expected map with a single entry");
        }
        let variantName = singleEntry.key;
        switch variantName {
        case "invalid_request":
            return .invalidRequest(reason: try ChatGenerationFailureReason.decodeReasonPayload(singleEntry.value));
        case "fatal_execution":
            return .fatalExecution(reason: try ChatGenerationFailureReason.decodeReasonPayload(singleEntry.value));
        case "context_length_exceeded":
            let payloadObject = try JsonWireValue.extractObject(singleEntry.value);
            let parsedReason = ChatGenerationFailureReason.contextLengthExceeded(
                actualTotalContextTokens: try payloadObject.decodeUInt32(fieldName: "actual_total_context_tokens"),
                maximumContextTokens: try payloadObject.decodeUInt32(fieldName: "maximum_context_tokens"));
            try payloadObject.rejectUnknownFields(allowedFieldNames: ["actual_total_context_tokens", "maximum_context_tokens"]);
            return parsedReason;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(ChatGenerationFailureReason.expectedVariantNames))");
        }
    }

    private static func decodeReasonPayload(_ payloadWireValue: JsonWireValue) throws -> String {
        let payloadObject = try JsonWireValue.extractObject(payloadWireValue);
        let reasonText = try payloadObject.decodeString(fieldName: "reason");
        try payloadObject.rejectUnknownFields(allowedFieldNames: ["reason"]);
        return reasonText;
    }

    private static func reasonObjectWireValue(_ reasonText: String) -> JsonWireValue {
        var payloadObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        payloadObject.appendEntry(key: "reason", value: .string(reasonText));
        return .object(payloadObject);
    }

    private static func singleEntryWireObject(variantName: String, payloadWireValue: JsonWireValue) -> JsonWireObject {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: variantName, value: payloadWireValue);
        return wireObject;
    }
}

/// A bounded reason why structured chat generation stopped.
public enum ChatGenerationCompletionReason: Equatable {
    /// The model emitted its configured end-of-sequence token.
    case endOfSequence;
    /// The request produced exactly its allowed output-token count.
    case maximumOutputTokens;
    /// The model emitted at least one complete validated function call.
    case toolCalls;
    /// The supervisor cancelled the active request.
    case cancelled;

    private static let expectedVariantNames: Array<String> = [
        "end_of_sequence", "maximum_output_tokens", "tool_calls", "cancelled",
    ];

    internal var wireName: String {
        switch self {
        case .endOfSequence: return "end_of_sequence";
        case .maximumOutputTokens: return "maximum_output_tokens";
        case .toolCalls: return "tool_calls";
        case .cancelled: return "cancelled";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatGenerationCompletionReason {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "end_of_sequence": return .endOfSequence;
        case "maximum_output_tokens": return .maximumOutputTokens;
        case "tool_calls": return .toolCalls;
        case "cancelled": return .cancelled;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(ChatGenerationCompletionReason.expectedVariantNames))");
        }
    }
}
