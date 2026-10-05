import Foundation;
import IpcProtocol;

/// OpenAI `response_format` for chat and Responses.
///
/// Extra-body `structured_outputs` masks illegal logits. OpenAI `response_format`
/// still uses a bounded prompt instruction plus best-effort extraction, disclosed
/// through the HTTP Warning header. Port of
/// crates/rest-contract/src/openai_response_format.rs (wire types and validation).
public enum ResponseFormatConstants {
    /// Maximum serialized JSON Schema accepted on `response_format`.
    public static let MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES: Int = 65_536;
    /// Advertised when extra-body structured_outputs can mask illegal logits.
    public static let STRUCTURED_OUTPUT_ENFORCEMENT_LOGITS_MASK: String = "logits_mask";
    /// RFC 7234 Warning when json_object / json_schema cannot be grammar-enforced.
    public static let UNENFORCED_RESPONSE_FORMAT_WARNING: String =
        "199 astronomical \"response_format not enforced; grammar-constrained decoding unavailable, output is best-effort\"";
    /// RFC 7234 Warning when `strict` json_schema cannot be grammar-enforced.
    public static let UNENFORCED_STRICT_RESPONSE_FORMAT_WARNING: String =
        "199 astronomical \"response_format strict json_schema not enforced; grammar-constrained decoding unavailable, output is best-effort and NOT schema-enforced\"";
}

/// Wire `response_format` object from Chat Completions.
public struct OpenAiResponseFormat: Equatable {
    private let formatTypeName: String;
    private let jsonSchemaSpec: OpenAiJsonSchemaSpec?;

    fileprivate init(formatType: String, jsonSchema: OpenAiJsonSchemaSpec?) {
        self.formatTypeName = formatType;
        self.jsonSchemaSpec = jsonSchema;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseFormat {
        let formatObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try formatObject.rejectUnknownFields(allowedFieldNames: ["type", "json_schema"]);
        return OpenAiResponseFormat(
            formatType: try formatObject.decodeString(fieldName: "type"),
            jsonSchema: try formatObject.decodeOptionalRawValueAllowingAbsent(fieldName: "json_schema")
                .map({ (schemaWireValue: JsonWireValue) throws -> OpenAiJsonSchemaSpec in
                    return try OpenAiJsonSchemaSpec.decoded(wireValue: schemaWireValue);
                }));
    }

    public var formatType: String {
        return self.formatTypeName;
    }

    public var jsonSchema: OpenAiJsonSchemaSpec? {
        return self.jsonSchemaSpec;
    }

    /// Validates Chat Completions `response_format`. `text` becomes `nil`.
    public func intoStructuredOutput() throws -> OpenAiStructuredOutput? {
        return try OpenAiResponseFormat.structuredOutputFromTypeAndSchema(
            formatType: self.formatTypeName,
            schemaName: self.jsonSchemaSpec?.name,
            schemaDescription: self.jsonSchemaSpec?.schemaDescription,
            schema: self.jsonSchemaSpec?.schema,
            strict: self.jsonSchemaSpec?.strict ?? false);
    }

    /// Shared validation core mirroring the Rust free function
    /// `structured_output_from_type_and_schema`.
    static func structuredOutputFromTypeAndSchema(
        formatType: String, schemaName: String?, schemaDescription: String?,
        schema: JsonWireValue?, strict: Bool) throws -> OpenAiStructuredOutput? {
        switch formatType {
        case "text":
            return nil;
        case "json_object":
            return .jsonObject;
        case "json_schema":
            let name: String = schemaName ?? "";
            if name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw OpenAiStructuredOutputValidationError.jsonSchemaNameEmpty;
            }
            guard let schema: JsonWireValue = schema else {
                throw OpenAiStructuredOutputValidationError.jsonSchemaMustBeObject;
            }
            guard case .object = schema else {
                throw OpenAiStructuredOutputValidationError.jsonSchemaMustBeObject;
            }
            let schemaNestingDepth: Int = ChatToolValidation.jsonNestingDepth(schema);
            if schemaNestingDepth > ChatCompletionLimits.MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH {
                throw OpenAiStructuredOutputValidationError.jsonSchemaNestingTooDeep(
                    actualSchemaNestingDepth: schemaNestingDepth,
                    maximumSchemaNestingDepth: ChatCompletionLimits.MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH);
            }
            let serializedSchemaByteCount: Int;
            do {
                serializedSchemaByteCount = try ChatToolValidation.canonicalSerializedByteCount(schema);
            } catch {
                serializedSchemaByteCount = 0;
            }
            if serializedSchemaByteCount > ResponseFormatConstants.MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES {
                throw OpenAiStructuredOutputValidationError.jsonSchemaTooLarge(
                    actualSchemaBytes: serializedSchemaByteCount,
                    maximumSchemaBytes: ResponseFormatConstants.MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES);
            }
            var visibleDescription: String? = nil;
            if let schemaDescription: String = schemaDescription, schemaDescription.isEmpty == false {
                visibleDescription = schemaDescription;
            }
            return .jsonSchema(
                name: name,
                description: visibleDescription,
                schema: ChatToolValidation.canonicalWireValue(schema),
                strict: strict);
        default:
            throw OpenAiStructuredOutputValidationError.unsupportedType(formatType: formatType);
        }
    }
}

/// Nested `json_schema` object on Chat Completions `response_format`.
/// Extra keys are ignored so OpenAI clients that send unused optional fields still validate.
public struct OpenAiJsonSchemaSpec: Equatable {
    private let nameText: String?;
    private let schemaDescriptionText: String?;
    private let schemaValue: JsonWireValue?;
    private let strictFlag: Bool?;

    fileprivate init(name: String?, schemaDescription: String?, schema: JsonWireValue?, strict: Bool?) {
        self.nameText = name;
        self.schemaDescriptionText = schemaDescription;
        self.schemaValue = schema;
        self.strictFlag = strict;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiJsonSchemaSpec {
        let specObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        return OpenAiJsonSchemaSpec(
            name: try specObject.decodeOptionalStringAllowingAbsent(fieldName: "name"),
            schemaDescription: try specObject.decodeOptionalStringAllowingAbsent(fieldName: "description"),
            schema: try specObject.decodeOptionalRawValueAllowingAbsent(fieldName: "schema"),
            strict: try specObject.decodeOptionalBoolAllowingAbsent(fieldName: "strict"));
    }

    public var name: String? {
        return self.nameText;
    }

    public var schemaDescription: String? {
        return self.schemaDescriptionText;
    }

    public var schema: JsonWireValue? {
        return self.schemaValue;
    }

    public var strict: Bool? {
        return self.strictFlag;
    }
}

/// Validated structured-output request after public contract checks.
public enum OpenAiStructuredOutput: Equatable {
    /// Any JSON value.
    case jsonObject;
    /// JSON that should match the caller-supplied schema.
    case jsonSchema(name: String, description: String?, schema: JsonWireValue, strict: Bool);

    /// Prompt instruction used while sample-time grammar masking is unavailable.
    public func jsonOutputInstruction() -> String {
        switch self {
        case .jsonObject:
            return "Output a single JSON value and nothing else after any reasoning: no markdown fences and no prose.";
        case .jsonSchema(let name, let description, let schema, _):
            let serializedSchema: String;
            do {
                serializedSchema = try ChatToolValidation.canonicalSerializedText(schema);
            } catch {
                serializedSchema = "{}";
            }
            var descriptionClause: String = "";
            if let description: String = description {
                descriptionClause = " (\(description))";
            }
            return "Output a single JSON object named \(name)\(descriptionClause) matching this schema and nothing else after any reasoning: no markdown fences and no prose. Schema: \(serializedSchema)";
        }
    }

    /// HTTP Warning value when this request cannot be grammar-enforced.
    public func unenforcedWarningHeader() -> String {
        switch self {
        case .jsonSchema(_, _, _, true):
            return ResponseFormatConstants.UNENFORCED_STRICT_RESPONSE_FORMAT_WARNING;
        case .jsonObject, .jsonSchema(_, _, _, false):
            return ResponseFormatConstants.UNENFORCED_RESPONSE_FORMAT_WARNING;
        }
    }

    /// Parses Responses `text.format` into the same structured-output enum.
    public static func structured_output_from_responses_text_format(
        _ textConfiguration: JsonWireValue?) throws -> OpenAiStructuredOutput? {
        guard let textConfiguration: JsonWireValue = textConfiguration else {
            return nil;
        }
        guard let formatObject: JsonWireValue = ResponseFormatValueAccess.field(textConfiguration, key: "format") else {
            return nil;
        }
        let formatType: String = ResponseFormatValueAccess.stringValue(ResponseFormatValueAccess.field(formatObject, key: "type")) ?? "text";
        let nestedSchema: JsonWireValue? = ResponseFormatValueAccess.field(formatObject, key: "json_schema");
        let schemaName: String? = ResponseFormatValueAccess.stringValue(ResponseFormatValueAccess.field(formatObject, key: "name"))
            ?? ResponseFormatValueAccess.stringValue(ResponseFormatValueAccess.field(nestedSchema, key: "name"));
        let schemaDescription: String? = ResponseFormatValueAccess.stringValue(ResponseFormatValueAccess.field(formatObject, key: "description"))
            ?? ResponseFormatValueAccess.stringValue(ResponseFormatValueAccess.field(nestedSchema, key: "description"));
        let schema: JsonWireValue? = ResponseFormatValueAccess.field(formatObject, key: "schema")
            ?? ResponseFormatValueAccess.field(nestedSchema, key: "schema");
        let strict: Bool = ResponseFormatValueAccess.boolValue(ResponseFormatValueAccess.field(formatObject, key: "strict"))
            ?? ResponseFormatValueAccess.boolValue(ResponseFormatValueAccess.field(nestedSchema, key: "strict"))
            ?? false;
        return try OpenAiResponseFormat.structuredOutputFromTypeAndSchema(
            formatType: formatType,
            schemaName: schemaName,
            schemaDescription: schemaDescription,
            schema: schema,
            strict: strict);
    }

    /// Picks one structured-output request when both Chat and Responses fields are set.
    public static func merge_structured_output_requests(
        responseFormat: OpenAiStructuredOutput?, textFormat: OpenAiStructuredOutput?) throws -> OpenAiStructuredOutput? {
        switch (responseFormat, textFormat) {
        case (nil, nil):
            return nil;
        case (.some(let structuredOutput), nil), (nil, .some(let structuredOutput)):
            return structuredOutput;
        case (.some(let left), .some(let right)) where left == right:
            return left;
        case (.some, .some):
            throw OpenAiStructuredOutputValidationError.conflictingStructuredOutputFields;
        }
    }
}

/// Rejection for a malformed or unsupported structured-output request.
public enum OpenAiStructuredOutputValidationError: Error, Equatable {
    /// `response_format.type` is not text, json_object, or json_schema.
    case unsupportedType(formatType: String);
    /// json_schema requests must name the schema.
    case jsonSchemaNameEmpty;
    /// json_schema requests must include a schema object.
    case jsonSchemaMustBeObject;
    /// Schema nesting matches the tool-schema depth cap.
    case jsonSchemaNestingTooDeep(actualSchemaNestingDepth: Int, maximumSchemaNestingDepth: Int);
    /// Schema payload is too large to inject into a prompt.
    case jsonSchemaTooLarge(actualSchemaBytes: Int, maximumSchemaBytes: Int);
    /// Chat `response_format` and Responses `text.format` disagreed.
    case conflictingStructuredOutputFields;

    public var errorDescription: String? {
        switch self {
        case .unsupportedType(let formatType):
            return "response_format type '\(formatType)' is unsupported";
        case .jsonSchemaNameEmpty:
            return "response_format json_schema name must not be empty";
        case .jsonSchemaMustBeObject:
            return "response_format json_schema.schema must be an object";
        case .jsonSchemaNestingTooDeep(let actualSchemaNestingDepth, let maximumSchemaNestingDepth):
            return "response_format schema nesting depth is \(actualSchemaNestingDepth), exceeding \(maximumSchemaNestingDepth)";
        case .jsonSchemaTooLarge(let actualSchemaBytes, let maximumSchemaBytes):
            return "response_format schema is \(actualSchemaBytes) bytes, exceeding the \(maximumSchemaBytes) byte limit";
        case .conflictingStructuredOutputFields:
            return "response_format and text.format must describe the same structured output";
        }
    }
}

/// `serde_json::Value`-style field access: `get` only resolves on objects and
/// returns nil for every other shape, matching `Value::get` semantics.
public enum ResponseFormatValueAccess {

    public static func field(_ wireValue: JsonWireValue?, key fieldName: String) -> JsonWireValue? {
        guard let wireValue: JsonWireValue = wireValue else {
            return nil;
        }
        guard case let .object(objectValue) = wireValue else {
            return nil;
        }
        return objectValue.value(forKey: fieldName);
    }

    static func stringValue(_ wireValue: JsonWireValue?) -> String? {
        guard let wireValue: JsonWireValue = wireValue else {
            return nil;
        }
        guard case let .string(textValue) = wireValue else {
            return nil;
        }
        return textValue;
    }

    static func boolValue(_ wireValue: JsonWireValue?) -> Bool? {
        guard let wireValue: JsonWireValue = wireValue else {
            return nil;
        }
        guard case let .boolean(booleanValue) = wireValue else {
            return nil;
        }
        return booleanValue;
    }
}
