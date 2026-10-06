// ResponsesTools.swift — RestContract
//
// Port of crates/rest-contract/src/openai_responses_tools.rs.
//
// Responses-native tool declarations and tool-selection input.

import Foundation;
import IpcProtocol;

/// One Responses-native tool declaration accepted by the local model.
public struct OpenAiResponseToolDefinition: Equatable {
    private let toolTypeName: String;
    private let nameText: String?;
    private let descriptionText: String?;
    private let parametersSchema: JsonWireValue?;
    private let strictSchemaFlag: Bool?;
    /// Unknown fields absorbed by serde's flatten, kept byte-sorted by key so
    /// the emptiness check and any rejection match the Rust BTreeMap ordering.
    private let additionalFields: Array<OpenAiUnknownRequestField>;

    fileprivate init(
        toolTypeName: String, nameText: String?, descriptionText: String?,
        parametersSchema: JsonWireValue?, strictSchemaFlag: Bool?,
        additionalFields: Array<OpenAiUnknownRequestField>) {
        self.toolTypeName = toolTypeName;
        self.nameText = nameText;
        self.descriptionText = descriptionText;
        self.parametersSchema = parametersSchema;
        self.strictSchemaFlag = strictSchemaFlag;
        self.additionalFields = additionalFields;
    }

    /// Mirrors the serde derive for a struct with `#[serde(flatten)]`: known
    /// fields decode by their exact wire names, a missing or null `Option`
    /// decodes to `None`, duplicate known fields are rejected the way serde's
    /// derived visitor does, and every other key lands in the flattened map.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseToolDefinition {
        let toolObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var decodedToolTypeName: String? = nil;
        var decodedNameText: String? = nil;
        var decodedDescriptionText: String? = nil;
        var decodedParametersSchema: JsonWireValue? = nil;
        var decodedStrictSchemaFlag: Bool? = nil;
        var additionalFields: Array<OpenAiUnknownRequestField> = Array();
        for entry in toolObject.entries {
            switch entry.key {
            case "type":
                if decodedToolTypeName != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "type");
                }
                decodedToolTypeName = try JsonWireValue.extractString(entry.value);
            case "name":
                if decodedNameText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "name");
                }
                if entry.value.isNull == false {
                    decodedNameText = try JsonWireValue.extractString(entry.value);
                }
            case "description":
                if decodedDescriptionText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "description");
                }
                if entry.value.isNull == false {
                    decodedDescriptionText = try JsonWireValue.extractString(entry.value);
                }
            case "parameters":
                if decodedParametersSchema != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "parameters");
                }
                if entry.value.isNull == false {
                    decodedParametersSchema = entry.value;
                }
            case "strict":
                if decodedStrictSchemaFlag != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "strict");
                }
                if entry.value.isNull == false {
                    decodedStrictSchemaFlag = try JsonWireValue.extractBool(entry.value);
                }
            default:
                additionalFields.append(OpenAiUnknownRequestField(
                    fieldName: entry.key, fieldValue: entry.value));
            }
        }
        guard let resolvedToolTypeName: String = decodedToolTypeName else {
            throw JsonWireProblem.missingField(fieldName: "type");
        }
        // Rust collects flattened unknown fields in a BTreeMap, which orders
        // keys by UTF-8 bytes; reproduce that ordering for later inspection.
        additionalFields.sort(by: { (leftField: OpenAiUnknownRequestField, rightField: OpenAiUnknownRequestField) -> Bool in
            return Array(leftField.fieldName.utf8).lexicographicallyPrecedes(Array(rightField.fieldName.utf8));
        });
        return OpenAiResponseToolDefinition(
            toolTypeName: resolvedToolTypeName,
            nameText: decodedNameText,
            descriptionText: decodedDescriptionText,
            parametersSchema: decodedParametersSchema,
            strictSchemaFlag: decodedStrictSchemaFlag,
            additionalFields: additionalFields);
    }

    /// Validates this declaration and consumes it into protocol-neutral parts.
    internal func intoParts() throws -> OpenAiResponseToolDefinitionParts {
        if self.toolTypeName != "function" {
            throw OpenAiResponsesValidationError.unsupportedOption(optionName: "tools[].type");
        }
        if self.additionalFields.isEmpty == false {
            throw OpenAiResponsesValidationError.unsupportedOption(optionName: "tools[]");
        }
        let functionName: String = self.nameText ?? "";
        try ResponsesToolsSupport.validateFunctionName(functionName: functionName);
        if let strictSchemaFlag: Bool = self.strictSchemaFlag {
            if strictSchemaFlag {
                throw OpenAiResponsesValidationError.unsupportedOption(optionName: "tools[].strict=true");
            }
        }
        if let parametersSchema: JsonWireValue = self.parametersSchema {
            let schemaNestingDepth: Int = ChatToolValidation.jsonNestingDepth(parametersSchema);
            if schemaNestingDepth > ChatCompletionLimits.MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH {
                throw OpenAiResponsesValidationError.toolSchemaNestingTooDeep(
                    actualSchemaNestingDepth: schemaNestingDepth,
                    maximumSchemaNestingDepth: ChatCompletionLimits.MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH);
            }
        }
        let resolvedParameters: JsonWireValue = self.parametersSchema
            ?? JsonWireValue.object(JsonWireObject(entries: Array()));
        return OpenAiResponseToolDefinitionParts(
            name: functionName,
            description: self.descriptionText,
            parameters: resolvedParameters,
            strict: false);
    }
}

/// One validated local function declaration.
public struct OpenAiResponseToolDefinitionParts: Equatable {
    public let name: String;
    public let description: String?;
    public let parameters: JsonWireValue;
    public let strict: Bool;

    public init(name: String, description: String?, parameters: JsonWireValue, strict: Bool) {
        self.name = name;
        self.description = description;
        self.parameters = parameters;
        self.strict = strict;
    }

    /// serde_json stores parsed Values in a BTreeMap, so the Rust
    /// serialization of this schema emits keys in UTF-8 byte order; the
    /// worker consumes exactly those canonical bytes.
    public func canonicalParametersJson() throws -> String {
        return try ChatToolValidation.canonicalSerializedText(self.parameters);
    }
}

/// Responses tool-selection input, retained broadly for precise validation.
public enum OpenAiResponseToolChoice: Equatable {
    /// A bare selection mode string such as `auto` or `none`.
    case mode(String);
    /// Any object-shaped selection payload, retained for precise rejection.
    case selection(JsonWireValue);

    /// Mirrors the untagged serde derive: the `String` alternative is tried
    /// first and the `Value` alternative accepts anything, so every wire value
    /// decodes and precision is recovered during validation.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseToolChoice {
        if case let .string(modeText) = wireValue {
            return .mode(modeText);
        }
        return .selection(wireValue);
    }

    internal func intoParts() throws -> OpenAiResponseToolChoiceParts {
        switch self {
        case .mode(let modeText) where modeText == "auto":
            return .auto;
        case .mode(let modeText) where modeText == "none":
            return .none;
        case .mode, .selection:
            throw OpenAiResponsesValidationError.unsupportedOption(optionName: "tool_choice");
        }
    }
}

/// Tool-selection behavior the current Qwen3.5-MoE prompt can enforce.
public enum OpenAiResponseToolChoiceParts: Equatable {
    /// The model chooses whether to call a declared function.
    case auto;
    /// The model must not call a function.
    case none;

    public func kindName() -> String {
        switch self {
        case .auto: return "auto";
        case .none: return "none";
        }
    }
}

/// Module-private helpers mirroring the free functions at the bottom of
/// crates/rest-contract/src/openai_responses_tools.rs.
fileprivate enum ResponsesToolsSupport {

    /// Accepts only non-empty names of ASCII alphanumerics, underscores, and
    /// hyphens, mirroring the Rust byte-level check.
    fileprivate static func validateFunctionName(functionName: String) throws -> Void {
        let isPortableName: Bool = functionName.isEmpty == false
            && functionName.utf8.allSatisfy({ (characterByte: UInt8) -> Bool in
                return (characterByte >= 48 && characterByte <= 57)
                    || (characterByte >= 65 && characterByte <= 90)
                    || (characterByte >= 97 && characterByte <= 122)
                    || characterByte == UInt8(ascii: "_")
                    || characterByte == UInt8(ascii: "-");
            });
        if isPortableName {
            return;
        }
        throw OpenAiResponsesValidationError.invalidToolName(toolName: functionName);
    }
}
