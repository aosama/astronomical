import Foundation;
import IpcProtocol;

/// OpenAI tool type accepted by this endpoint.
public enum OpenAiToolType: Equatable {
    /// A JSON-schema function tool.
    case function;

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiToolType {
        let typeName: String = try JsonWireValue.extractString(wireValue);
        if typeName == "function" {
            return .function;
        }
        throw JsonWireProblem.malformedDocument(
            problem: "unknown variant `\(typeName)`, expected `function`");
    }
}

/// A function tool made available to the model.
public struct OpenAiToolDefinition: Equatable {
    private let toolType: OpenAiToolType;
    private let function: OpenAiFunctionDefinition;

    fileprivate init(toolType: OpenAiToolType, function: OpenAiFunctionDefinition) {
        self.toolType = toolType;
        self.function = function;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiToolDefinition {
        let toolObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try toolObject.rejectUnknownFields(allowedFieldNames: ["type", "function"]);
        return OpenAiToolDefinition(
            toolType: try OpenAiToolType.decoded(
                wireValue: try toolObject.requireObjectValue(fieldName: "type")),
            function: try OpenAiFunctionDefinition.decoded(
                wireValue: try toolObject.requireObjectValue(fieldName: "function")));
    }

    /// Returns the declared function name.
    public func name() -> String {
        return self.function.name;
    }

    /// Returns the serialized JSON Schema byte count for the bounded schema
    /// check, or zero when the function declares no parameters.
    internal func validate() throws -> Int {
        if self.toolType != OpenAiToolType.function {
            throw OpenAiChatCompletionValidationError.unsupportedToolType;
        }
        try ChatToolValidation.validateFunctionName(functionName: self.function.name);
        guard let parameters: JsonWireValue = self.function.parameters else {
            return 0;
        }
        let schemaNestingDepth: Int = ChatToolValidation.jsonNestingDepth(parameters);
        if schemaNestingDepth > ChatCompletionLimits.MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH {
            throw OpenAiChatCompletionValidationError.toolSchemaNestingTooDeep(
                actualSchemaNestingDepth: schemaNestingDepth,
                maximumSchemaNestingDepth: ChatCompletionLimits.MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH);
        }
        // serde_json re-serializes a parsed Value with BTreeMap-ordered keys;
        // the canonical sorted-key serialization reproduces those bytes so the
        // schema byte count matches the Rust contract exactly.
        do {
            return try ChatToolValidation.canonicalSerializedByteCount(parameters);
        } catch {
            throw OpenAiChatCompletionValidationError.toolSchemaSerializationFailed;
        }
    }

    internal func intoParts() throws -> OpenAiToolDefinitionParts {
        let serializedSchemaByteCount: Int = try self.validate();
        let parametersJson: String;
        if let parameters: JsonWireValue = self.function.parameters {
            parametersJson = try ChatToolValidation.canonicalSerializedText(
                parameters, expectedByteCount: serializedSchemaByteCount);
        } else {
            parametersJson = "{}";
        }
        return OpenAiToolDefinitionParts(
            name: self.function.name,
            description: self.function.description,
            parametersJson: parametersJson);
    }
}

/// One validated function declaration ready for protocol translation.
public struct OpenAiToolDefinitionParts: Equatable {
    /// The declared function name.
    public let name: String;
    /// Optional function description.
    public let description: String?;
    /// Canonical JSON Schema for the function parameters.
    public let parametersJson: String;

    public init(name: String, description: String?, parametersJson: String) {
        self.name = name;
        self.description = description;
        self.parametersJson = parametersJson;
    }
}

/// One function declaration in a tool definition.
public struct OpenAiFunctionDefinition: Equatable {
    private let nameText: String;
    private let descriptionText: String?;
    private let parametersValue: JsonWireValue?;

    fileprivate init(name: String, description: String?, parameters: JsonWireValue?) {
        self.nameText = name;
        self.descriptionText = description;
        self.parametersValue = parameters;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiFunctionDefinition {
        let functionObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try functionObject.rejectUnknownFields(allowedFieldNames: ["name", "description", "parameters"]);
        return OpenAiFunctionDefinition(
            name: try functionObject.decodeString(fieldName: "name"),
            description: try functionObject.decodeOptionalStringAllowingAbsent(fieldName: "description"),
            parameters: try functionObject.decodeOptionalRawValueAllowingAbsent(fieldName: "parameters"));
    }

    fileprivate var name: String {
        return self.nameText;
    }

    fileprivate var description: String? {
        return self.descriptionText;
    }

    fileprivate var parameters: JsonWireValue? {
        return self.parametersValue;
    }
}

/// A selected automatic tool mode or one named forced function. Untagged on
/// the wire: a bare mode string or a `{type, function:{name}}` object.
public enum OpenAiToolChoice: Equatable {
    /// `auto`, `none`, or `required`.
    case mode(String);
    /// A forced function choice.
    case function(toolType: OpenAiToolType, function: OpenAiFunctionChoice);

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiToolChoice {
        if case let .string(modeText) = wireValue {
            return .mode(modeText);
        }
        if case let .object(choiceObject) = wireValue {
            return .function(
                toolType: try OpenAiToolType.decoded(
                    wireValue: try choiceObject.requireObjectValue(fieldName: "type")),
                function: try OpenAiFunctionChoice.decoded(
                    wireValue: try choiceObject.requireObjectValue(fieldName: "function")));
        }
        throw JsonWireProblem.malformedDocument(
            problem: "data did not match any variant of untagged enum OpenAiToolChoice");
    }

    internal func intoMode() -> OpenAiToolChoiceMode {
        switch self {
        case .mode(let mode) where mode == "auto":
            return .auto;
        case .mode(let mode) where mode == "none":
            return .none;
        case .mode:
            return .required;
        case .function(_, let function):
            return .function(name: function.name());
        }
    }
}

/// The named function inside a forced tool choice.
public struct OpenAiFunctionChoice: Equatable {
    private let nameText: String;

    fileprivate init(name: String) {
        self.nameText = name;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiFunctionChoice {
        let choiceObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        return OpenAiFunctionChoice(name: try choiceObject.decodeString(fieldName: "name"));
    }

    internal func name() -> String {
        return self.nameText;
    }
}

/// The validated tool-selection mode that a worker can implement.
public enum OpenAiToolChoiceMode: Equatable {
    /// The model chooses whether to call a declared function.
    case auto;
    /// The model must not call a function.
    case none;
    /// The model must call a function.
    case required;
    /// The model must call one specific declared function.
    case function(name: String);
}

/// One assistant tool call retained in chat history.
public struct OpenAiAssistantToolCall: Equatable {
    private let identifier: String;
    private let toolType: OpenAiToolType;
    private let function: OpenAiAssistantToolFunction;

    fileprivate init(id: String, toolType: OpenAiToolType, function: OpenAiAssistantToolFunction) {
        self.identifier = id;
        self.toolType = toolType;
        self.function = function;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiAssistantToolCall {
        let toolCallObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        return OpenAiAssistantToolCall(
            id: try toolCallObject.decodeString(fieldName: "id"),
            toolType: try OpenAiToolType.decoded(
                wireValue: try toolCallObject.requireObjectValue(fieldName: "type")),
            function: try OpenAiAssistantToolFunction.decoded(
                wireValue: try toolCallObject.requireObjectValue(fieldName: "function")));
    }

    internal func validate() throws -> Void {
        try ChatToolValidation.validateNonEmptyString(fieldName: "assistant tool-call ID", stringValue: self.identifier);
        if self.toolType != OpenAiToolType.function {
            throw OpenAiChatCompletionValidationError.unsupportedToolType;
        }
        // History tool calls echo model output back. The output parser
        // deliberately fail-opens closed envelopes with unknown or sloppy
        // names to the harness, so clients replay arbitrary model-invented
        // names in subsequent requests; only a non-empty name is required to
        // round-trip. The strict portable grammar remains enforced on the
        // caller-declared `tools` definitions that drive the renderer.
        try ChatToolValidation.validateNonEmptyString(fieldName: "assistant tool-call name", stringValue: self.function.name);
    }

    internal func intoParts() -> OpenAiAssistantToolCallParts {
        return OpenAiAssistantToolCallParts(
            id: self.identifier,
            name: self.function.name,
            argumentsJson: self.function.arguments);
    }
}

/// One validated assistant function call ready for protocol translation.
public struct OpenAiAssistantToolCallParts: Equatable {
    /// Client-visible call ID used to correlate a later tool response.
    public let id: String;
    /// Declared function name.
    public let name: String;
    /// JSON-encoded function arguments.
    public let argumentsJson: String;

    public init(id: String, name: String, argumentsJson: String) {
        self.id = id;
        self.name = name;
        self.argumentsJson = argumentsJson;
    }
}

/// One JSON-encoded function invocation retained in assistant history.
public struct OpenAiAssistantToolFunction: Equatable {
    private let nameText: String;
    private let argumentsText: String;

    fileprivate init(name: String, arguments: String) {
        self.nameText = name;
        self.argumentsText = arguments;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiAssistantToolFunction {
        let functionObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        return OpenAiAssistantToolFunction(
            name: try functionObject.decodeString(fieldName: "name"),
            arguments: try functionObject.decodeString(fieldName: "arguments"));
    }

    fileprivate var name: String {
        return self.nameText;
    }

    fileprivate var arguments: String {
        return self.argumentsText;
    }
}

/// Stream-specific OpenAI options accepted by this endpoint.
public struct OpenAiStreamOptions: Equatable {
    private let includeUsageFlag: Bool;

    fileprivate init(includeUsage: Bool) {
        self.includeUsageFlag = includeUsage;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiStreamOptions {
        let optionsObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        return OpenAiStreamOptions(
            includeUsage: try optionsObject.decodeBoolAllowingAbsent(fieldName: "include_usage"));
    }

    /// Whether the terminal streamed chunk must carry usage information.
    internal var includeUsage: Bool {
        return self.includeUsageFlag;
    }
}

/// A single stop sequence or a bounded sequence list. Untagged on the wire.
public enum OpenAiStopSequences: Equatable {
    /// One stop sequence.
    case single(String);
    /// Multiple stop sequences.
    case multiple(Array<String>);

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiStopSequences {
        if case let .string(stopText) = wireValue {
            return .single(stopText);
        }
        if case let .array(stopValues) = wireValue {
            return .multiple(try stopValues.map({ (stopWireValue: JsonWireValue) throws -> String in
                return try JsonWireValue.extractString(stopWireValue);
            }));
        }
        throw JsonWireProblem.malformedDocument(
            problem: "data did not match any variant of untagged enum OpenAiStopSequences");
    }
}

/// Validation helpers shared by the chat tool wire types, mirroring the free
/// functions at the bottom of crates/rest-contract/src/openai_chat_types.rs.
internal enum ChatToolValidation {

    internal static func validateFunctionName(functionName: String) throws -> Void {
        try validateNonEmptyString(fieldName: "tool name", stringValue: functionName);
        let isPortableName: Bool = functionName.utf8.allSatisfy({ (character: UInt8) -> Bool in
            return (character >= 48 && character <= 57)
                || (character >= 65 && character <= 90)
                || (character >= 97 && character <= 122)
                || character == UInt8(ascii: "_") || character == UInt8(ascii: "-");
        });
        if isPortableName {
            return;
        }
        throw OpenAiChatCompletionValidationError.invalidToolName(toolName: functionName);
    }

    internal static func validateNonEmptyString(fieldName: String, stringValue: String) throws -> Void {
        if stringValue.isEmpty {
            throw OpenAiChatCompletionValidationError.emptyString(fieldName: fieldName);
        }
    }

    internal static func jsonNestingDepth(_ wireValue: JsonWireValue) -> Int {
        switch wireValue {
        case .array(let arrayValues):
            return 1 + (arrayValues.map({ (nestedValue: JsonWireValue) -> Int in jsonNestingDepth(nestedValue) }).max() ?? 0);
        case .object(let objectValue):
            return 1 + (objectValue.entries.map({ (entry: (key: String, value: JsonWireValue)) -> Int in jsonNestingDepth(entry.value) }).max() ?? 0);
        case .null, .boolean, .unsignedInteger, .signedInteger, .double, .float32, .string:
            return 0;
        }
    }

    /// serde_json stores parsed Values in a BTreeMap, so re-serializing emits
    /// keys sorted by UTF-8 byte order. Both helpers below reproduce those
    /// bytes from the insertion-ordered JsonWireObject tree.
    internal static func canonicalSerializedText(_ wireValue: JsonWireValue) throws -> String {
        return try canonicalWireValue(wireValue).serializedText;
    }

    internal static func canonicalSerializedText(_ wireValue: JsonWireValue, expectedByteCount: Int) throws -> String {
        let canonicalText: String = try canonicalWireValue(wireValue).serializedText;
        if canonicalText.utf8.count != expectedByteCount {
            throw OpenAiChatCompletionValidationError.toolSchemaSerializationFailed;
        }
        return canonicalText;
    }

    internal static func canonicalSerializedByteCount(_ wireValue: JsonWireValue) throws -> Int {
        return try canonicalWireValue(wireValue).serializedText.utf8.count;
    }

    internal static func canonicalWireValue(_ wireValue: JsonWireValue) -> JsonWireValue {
        switch wireValue {
        case .object(let objectValue):
            var sortedEntries: Array<(key: String, value: JsonWireValue)> = objectValue.entries;
            sortedEntries.sort { (leftEntry: (key: String, value: JsonWireValue), rightEntry: (key: String, value: JsonWireValue)) -> Bool in
                return lexicographicUtf8Order(leftEntry.key, rightEntry.key);
            };
            var canonicalObject: JsonWireObject = JsonWireObject(entries: Array());
            for sortedEntry: (key: String, value: JsonWireValue) in sortedEntries {
                canonicalObject.appendEntry(key: sortedEntry.key, value: canonicalWireValue(sortedEntry.value));
            }
            return .object(canonicalObject);
        case .array(let arrayValues):
            return .array(arrayValues.map({ (elementValue: JsonWireValue) -> JsonWireValue in
                return canonicalWireValue(elementValue);
            }));
        default:
            return wireValue;
        }
    }

    private static func lexicographicUtf8Order(_ leftText: String, _ rightText: String) -> Bool {
        return Array(leftText.utf8).lexicographicallyPrecedes(Array(rightText.utf8));
    }
}
