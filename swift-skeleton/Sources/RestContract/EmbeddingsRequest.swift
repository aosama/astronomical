import Foundation;
import IpcProtocol;

/// Strict request validation for the OpenAI-compatible embeddings boundary.
/// Port of crates/rest-contract/src/openai_embeddings_request.rs.

/// One request stays well inside the 32 MiB IPC frame budget after JSON
/// expansion, so a single embeddings command can never overflow the wire.
private let MAXIMUM_EMBEDDING_INPUT_COUNT: Int = 256;
private let MAXIMUM_EMBEDDING_INPUT_BYTES: Int = 8_192;
private let MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES: Int = 1_000_000;

/// One unknown top-level field absorbed by serde's flatten, kept in
/// BTreeMap (byte-ordered) sequence so the first rejection matches Rust.
private struct UnknownRequestField: Equatable {
    fileprivate let fieldName: String;
    fileprivate let fieldValue: JsonWireValue;
}

/// One strict request to the local OpenAI-compatible embeddings endpoint.
public struct OpenAiEmbeddingsRequest: Equatable {
    private let modelText: String;
    private let inputValue: OpenAiEmbeddingInput;
    private let encodingFormatName: String?;
    private let dimensionsValue: UInt32?;
    private let unknownFields: Array<UnknownRequestField>;

    fileprivate init(
        model: String, input: OpenAiEmbeddingInput, encodingFormat: String?,
        dimensions: UInt32?, unknownFields: Array<UnknownRequestField>) {
        self.modelText = model;
        self.inputValue = input;
        self.encodingFormatName = encodingFormat;
        self.dimensionsValue = dimensions;
        self.unknownFields = unknownFields;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiEmbeddingsRequest {
        let requestObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let knownFieldNames: Array<String> = ["model", "input", "encoding_format", "dimensions"];
        var unknownFields: Array<UnknownRequestField> = Array();
        for propertyName: String in requestObject.keyNames {
            if knownFieldNames.contains(propertyName) == false {
                unknownFields.append(UnknownRequestField(
                    fieldName: propertyName, fieldValue: requestObject.value(forKey: propertyName)!));
            }
        }
        // Rust collects flattened unknown fields in a BTreeMap, which orders
        // keys by UTF-8 bytes; reproduce that ordering for the first-key rule.
        unknownFields.sort { (leftEntry: UnknownRequestField, rightEntry: UnknownRequestField) -> Bool in
            return Array(leftEntry.fieldName.utf8).lexicographicallyPrecedes(Array(rightEntry.fieldName.utf8));
        };
        return OpenAiEmbeddingsRequest(
            model: try requestObject.decodeString(fieldName: "model"),
            input: try OpenAiEmbeddingInput.decoded(
                wireValue: try requestObject.requireObjectValue(fieldName: "input")),
            encodingFormat: try requestObject.decodeOptionalStringAllowingAbsent(fieldName: "encoding_format"),
            dimensions: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "dimensions"),
            unknownFields: unknownFields);
    }

    /// Validates and consumes this public request into protocol-neutral parts.
    public func intoParts() throws -> OpenAiEmbeddingsRequestParts {
        if let firstUnknownField: UnknownRequestField = self.unknownFields.first {
            throw OpenAiEmbeddingsValidationError.unknownField(fieldName: firstUnknownField.fieldName);
        }
        if self.modelText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw OpenAiEmbeddingsValidationError.emptyModel;
        }
        let inputs: Array<String>;
        switch self.inputValue {
        case .single(let text):
            inputs = [text];
        case .multiple(let texts):
            inputs = texts;
        }
        if inputs.isEmpty {
            throw OpenAiEmbeddingsValidationError.emptyInput;
        }
        if inputs.count > MAXIMUM_EMBEDDING_INPUT_COUNT {
            throw OpenAiEmbeddingsValidationError.inputCountExceeded(
                actualInputCount: inputs.count, maximumInputCount: MAXIMUM_EMBEDDING_INPUT_COUNT);
        }
        var totalBytes: Int = 0;
        for inputText: String in inputs {
            let inputBytes: Int = inputText.utf8.count;
            totalBytes = totalBytes + inputBytes;
            if inputBytes > MAXIMUM_EMBEDDING_INPUT_BYTES {
                throw OpenAiEmbeddingsValidationError.inputTextTooLarge(
                    actualInputBytes: inputBytes, maximumInputBytes: MAXIMUM_EMBEDDING_INPUT_BYTES);
            }
        }
        if totalBytes > MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES {
            throw OpenAiEmbeddingsValidationError.totalInputBytesExceeded(
                actualTotalBytes: totalBytes, maximumTotalBytes: MAXIMUM_EMBEDDING_TOTAL_INPUT_BYTES);
        }
        let encodingFormat: OpenAiEmbeddingEncodingFormat;
        switch self.encodingFormatName {
        case nil, "float":
            encodingFormat = .float;
        case "base64":
            encodingFormat = .base64;
        case .some(let unsupportedEncodingFormat):
            throw OpenAiEmbeddingsValidationError.unsupportedEncodingFormat(
                encodingFormat: unsupportedEncodingFormat);
        }
        if self.dimensionsValue == 0 {
            throw OpenAiEmbeddingsValidationError.invalidDimensions;
        }
        return OpenAiEmbeddingsRequestParts(
            model: self.modelText,
            inputs: inputs,
            encodingFormat: encodingFormat,
            dimensions: self.dimensionsValue);
    }
}

/// One or many text inputs submitted in request order. Untagged on the wire.
public enum OpenAiEmbeddingInput: Equatable {
    case single(String);
    case multiple(Array<String>);

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiEmbeddingInput {
        if case let .string(singleText) = wireValue {
            return .single(singleText);
        }
        if case let .array(textValues) = wireValue {
            return .multiple(try textValues.map({ (textWireValue: JsonWireValue) throws -> String in
                return try JsonWireValue.extractString(textWireValue);
            }));
        }
        throw JsonWireProblem.malformedDocument(
            problem: "data did not match any variant of untagged enum OpenAiEmbeddingInput");
    }
}

/// Validated embeddings request ready for supervisor translation.
public struct OpenAiEmbeddingsRequestParts: Equatable {
    public let model: String;
    public let inputs: Array<String>;
    public let encodingFormat: OpenAiEmbeddingEncodingFormat;
    public let dimensions: UInt32?;

    public init(
        model: String, inputs: Array<String>, encodingFormat: OpenAiEmbeddingEncodingFormat,
        dimensions: UInt32?) {
        self.model = model;
        self.inputs = inputs;
        self.encodingFormat = encodingFormat;
        self.dimensions = dimensions;
    }
}

/// Encoding applied to each returned vector.
public enum OpenAiEmbeddingEncodingFormat: Equatable {
    case float;
    case base64;

    /// Canonical OpenAI wire value.
    public func asStr() -> String {
        switch self {
        case .float: return "float";
        case .base64: return "base64";
        }
    }
}

/// Rejection reasons validated before queue admission.
public enum OpenAiEmbeddingsValidationError: Error, Equatable {
    case emptyModel;
    case emptyInput;
    case inputCountExceeded(actualInputCount: Int, maximumInputCount: Int);
    case inputTextTooLarge(actualInputBytes: Int, maximumInputBytes: Int);
    case totalInputBytesExceeded(actualTotalBytes: Int, maximumTotalBytes: Int);
    case unsupportedEncodingFormat(encodingFormat: String);
    case invalidDimensions;
    case unknownField(fieldName: String);

    public var errorDescription: String? {
        switch self {
        case .emptyModel:
            return "model must not be empty";
        case .emptyInput:
            return "input must contain at least one text string";
        case .inputCountExceeded(let actualInputCount, let maximumInputCount):
            return "embedding input count is \(actualInputCount), outside the 1..=\(maximumInputCount) range";
        case .inputTextTooLarge(let actualInputBytes, let maximumInputBytes):
            return "one embedding input has \(actualInputBytes) bytes, exceeding the \(maximumInputBytes)-byte limit";
        case .totalInputBytesExceeded(let actualTotalBytes, let maximumTotalBytes):
            return "aggregate embedding input has \(actualTotalBytes) bytes, exceeding the \(maximumTotalBytes)-byte limit";
        case .unsupportedEncodingFormat(let encodingFormat):
            return "encoding_format '\(encodingFormat)' is unsupported; use float or base64";
        case .invalidDimensions:
            return "dimensions must be a positive vector width";
        case .unknownField(let fieldName):
            return "request field '\(fieldName)' is unknown";
        }
    }
}
