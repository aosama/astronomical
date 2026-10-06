import Foundation;

/// One validated text-embedding request sent to the local inference worker.
/// Embedding inference is one encoder forward pass with pooling, so the IPC
/// command carries text inputs only and the worker returns one vector per input.
public struct EmbeddingsCommand: Equatable {
    public let requestId: RequestId;
    public let model: String;
    public let inputs: Array<String>;
    public let encodingFormat: EmbeddingEncodingFormat;
    /// Requested vector width when the caller truncates below the native width.
    public let dimensions: UInt32?;

    public init(
        requestId: RequestId,
        model: String,
        inputs: Array<String>,
        encodingFormat: EmbeddingEncodingFormat,
        dimensions: UInt32?
    ) {
        self.requestId = requestId;
        self.model = model;
        self.inputs = inputs;
        self.encodingFormat = encodingFormat;
        self.dimensions = dimensions;
    }

    /// Bounded request-side input count that keeps one command inside one IPC frame.
    public static let maximumEmbeddingInputCount: Int = 256;

    internal static let wireFieldNames: Array<String> = [
        "request_id", "model", "inputs", "encoding_format", "dimensions",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "request_id", value: self.requestId.wireValue());
        wireObject.appendEntry(key: "model", value: .string(self.model));
        wireObject.appendEntry(key: "inputs", value: JsonWireValue.mappedArray(self.inputs, mappedWireValue: { (inputText: String) -> JsonWireValue in .string(inputText) }));
        wireObject.appendEntry(key: "encoding_format", value: self.encodingFormat.wireValue());
        wireObject.appendEntry(key: "dimensions", value: EmbeddingsCommand.optionalUInt32WireValue(self.dimensions));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> EmbeddingsCommand {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCommand = EmbeddingsCommand(
            requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
            model: try wireObject.decodeString(fieldName: "model"),
            inputs: try wireObject.decodeArray(fieldName: "inputs", mappedElement: { (elementWireValue: JsonWireValue) throws -> String in
                try JsonWireValue.extractString(elementWireValue)
            }),
            encodingFormat: try EmbeddingEncodingFormat.fromWireValue(try wireObject.requireObjectValue(fieldName: "encoding_format")),
            dimensions: try wireObject.decodeOptionalUInt32(fieldName: "dimensions"));
        try wireObject.rejectUnknownFields(allowedFieldNames: EmbeddingsCommand.wireFieldNames);
        return parsedCommand;
    }

    private static func optionalUInt32WireValue(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }
}

/// Encoding applied to each returned vector.
public enum EmbeddingEncodingFormat: Equatable, Sendable {
    case float;
    case base64;

    private static let expectedVariantNames: Array<String> = ["float", "base64"];

    internal var wireName: String {
        switch self {
        case .float: return "float";
        case .base64: return "base64";
        }
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.wireName);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> EmbeddingEncodingFormat {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "float": return .float;
        case "base64": return .base64;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(EmbeddingEncodingFormat.expectedVariantNames))");
        }
    }
}

/// Failure delivered after an embeddings request was admitted to the worker.
/// Wire shape is serde's externally tagged enum: unit variants serialize as
/// plain strings and struct variants as single-entry objects.
public enum EmbeddingsFailureReason: Equatable, Sendable {
    /// The worker independently rejected malformed structured embeddings input.
    case invalidRequest(reason: String);
    /// A fatal model-execution failure reported before the worker exits.
    case fatalExecution(reason: String);
    /// Prompt tokens exceed the loaded encoder context.
    case contextLengthExceeded(actualTotalContextTokens: UInt32, maximumContextTokens: UInt32);
    /// A different embeddings request already owns the worker's bounded capacity.
    case engineBusy;
    /// Generated vectors could not be pooled or normalized into the declared contract.
    case malformedModelOutput;

    private static let expectedVariantNames: Array<String> = [
        "invalid_request", "fatal_execution", "context_length_exceeded", "engine_busy", "malformed_model_output",
    ];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case let .invalidRequest(reason):
            return .object(EmbeddingsFailureReason.singleEntryWireObject(variantName: "invalid_request", payloadWireValue: EmbeddingsFailureReason.reasonObjectWireValue(reason)));
        case let .fatalExecution(reason):
            return .object(EmbeddingsFailureReason.singleEntryWireObject(variantName: "fatal_execution", payloadWireValue: EmbeddingsFailureReason.reasonObjectWireValue(reason)));
        case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens):
            var payloadObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
            payloadObject.appendEntry(key: "actual_total_context_tokens", value: .unsignedInteger(UInt64(actualTotalContextTokens)));
            payloadObject.appendEntry(key: "maximum_context_tokens", value: .unsignedInteger(UInt64(maximumContextTokens)));
            return .object(EmbeddingsFailureReason.singleEntryWireObject(variantName: "context_length_exceeded", payloadWireValue: .object(payloadObject)));
        case .engineBusy:
            return .string("engine_busy");
        case .malformedModelOutput:
            return .string("malformed_model_output");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> EmbeddingsFailureReason {
        switch wireValue {
        case let .string(variantName):
            return try EmbeddingsFailureReason.unitVariant(variantName: variantName);
        case let .object(variantObject):
            return try EmbeddingsFailureReason.structVariant(variantObject: variantObject);
        default:
            throw JsonWireProblem.invalidType(expectedTypeName: "enum EmbeddingsFailureReason", found: wireValue.foundDescription);
        }
    }

    private static func unitVariant(variantName: String) throws -> EmbeddingsFailureReason {
        switch variantName {
        case "engine_busy": return .engineBusy;
        case "malformed_model_output": return .malformedModelOutput;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(EmbeddingsFailureReason.expectedVariantNames))");
        }
    }

    private static func structVariant(variantObject: JsonWireObject) throws -> EmbeddingsFailureReason {
        guard variantObject.entries.count == 1, let singleEntry = variantObject.entries.first else {
            throw JsonWireProblem.malformedDocument(problem: "expected map with a single entry");
        }
        let variantName = singleEntry.key;
        switch variantName {
        case "invalid_request":
            return .invalidRequest(reason: try EmbeddingsFailureReason.decodeReasonPayload(singleEntry.value));
        case "fatal_execution":
            return .fatalExecution(reason: try EmbeddingsFailureReason.decodeReasonPayload(singleEntry.value));
        case "context_length_exceeded":
            let payloadObject = try JsonWireValue.extractObject(singleEntry.value);
            let parsedReason = EmbeddingsFailureReason.contextLengthExceeded(
                actualTotalContextTokens: try payloadObject.decodeUInt32(fieldName: "actual_total_context_tokens"),
                maximumContextTokens: try payloadObject.decodeUInt32(fieldName: "maximum_context_tokens"));
            try payloadObject.rejectUnknownFields(allowedFieldNames: ["actual_total_context_tokens", "maximum_context_tokens"]);
            return parsedReason;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(EmbeddingsFailureReason.expectedVariantNames))");
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
