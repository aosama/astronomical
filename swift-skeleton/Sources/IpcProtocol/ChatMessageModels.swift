import Foundation;

/// One chat message crossing the supervisor-to-worker trust boundary.
/// Wire shape is internally tagged with `role` in snake_case.
public enum ChatMessage: Equatable {
    /// An initial model instruction.
    case system(content: String);
    /// A user message, possibly carrying decoded image inputs.
    case user(content: String, images: Array<ChatImageInput>);
    /// A previous assistant message and optional tool calls.
    case assistant(content: String?, reasoningContent: String?, toolCalls: Array<ChatAssistantToolCall>);
    /// A result from a prior tool call.
    case tool(toolCallId: String, content: String);

    private static let expectedVariantNames: Array<String> = ["system", "user", "assistant", "tool"];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case let .system(content):
            var wireObject = ChatMessage.emptyWireObject();
            wireObject.appendEntry(key: "role", value: .string("system"));
            wireObject.appendEntry(key: "content", value: .string(content));
            return .object(wireObject);
        case let .user(content, images):
            var wireObject = ChatMessage.emptyWireObject();
            wireObject.appendEntry(key: "role", value: .string("user"));
            wireObject.appendEntry(key: "content", value: .string(content));
            wireObject.appendEntry(key: "images", value: JsonWireValue.mappedArray(images, mappedWireValue: { (image: ChatImageInput) -> JsonWireValue in image.wireValue() }));
            return .object(wireObject);
        case let .assistant(content, reasoningContent, toolCalls):
            var wireObject = ChatMessage.emptyWireObject();
            wireObject.appendEntry(key: "role", value: .string("assistant"));
            wireObject.appendEntry(key: "content", value: ChatMessage.optionalStringWireValue(content));
            wireObject.appendEntry(key: "reasoning_content", value: ChatMessage.optionalStringWireValue(reasoningContent));
            wireObject.appendEntry(key: "tool_calls", value: JsonWireValue.mappedArray(toolCalls, mappedWireValue: { (toolCall: ChatAssistantToolCall) -> JsonWireValue in toolCall.wireValue() }));
            return .object(wireObject);
        case let .tool(toolCallId, content):
            var wireObject = ChatMessage.emptyWireObject();
            wireObject.appendEntry(key: "role", value: .string("tool"));
            wireObject.appendEntry(key: "tool_call_id", value: .string(toolCallId));
            wireObject.appendEntry(key: "content", value: .string(content));
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatMessage {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "role", expectedVariantNames: ChatMessage.expectedVariantNames) {
        case "system":
            let parsedMessage = ChatMessage.system(content: try wireObject.decodeString(fieldName: "content"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "role", allowedFieldNames: ["content"]);
            return parsedMessage;
        case "user":
            let parsedMessage = ChatMessage.user(
                content: try wireObject.decodeString(fieldName: "content"),
                images: try wireObject.decodeArrayAllowingAbsent(fieldName: "images", mappedElement: { (elementWireValue: JsonWireValue) throws -> ChatImageInput in
                    try ChatImageInput.fromWireValue(elementWireValue)
                }));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "role", allowedFieldNames: ["content", "images"]);
            return parsedMessage;
        case "assistant":
            let parsedMessage = ChatMessage.assistant(
                content: try wireObject.decodeOptionalString(fieldName: "content"),
                reasoningContent: try wireObject.decodeOptionalString(fieldName: "reasoning_content"),
                toolCalls: try wireObject.decodeArray(fieldName: "tool_calls", mappedElement: { (elementWireValue: JsonWireValue) throws -> ChatAssistantToolCall in
                    try ChatAssistantToolCall.fromWireValue(elementWireValue)
                }));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "role", allowedFieldNames: ["content", "reasoning_content", "tool_calls"]);
            return parsedMessage;
        default:
            let parsedMessage = ChatMessage.tool(
                toolCallId: try wireObject.decodeString(fieldName: "tool_call_id"),
                content: try wireObject.decodeString(fieldName: "content"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "role", allowedFieldNames: ["tool_call_id", "content"]);
            return parsedMessage;
        }
    }

    private static func emptyWireObject() -> JsonWireObject {
        return JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
    }

    private static func optionalStringWireValue(_ rawValue: String?) -> JsonWireValue {
        guard let unwrappedValue = rawValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }
}

/// One decoded image carried in a user chat message across the IPC boundary.
public struct ChatImageInput: Equatable {
    /// The MIME type parsed from the source data URI, e.g. `image/png`.
    public let mimeType: String;
    /// The raw decoded image file bytes (PNG/JPEG/WebP payload before pixel decoding).
    public let decodedBytes: Array<UInt8>;

    public init(mimeType: String, decodedBytes: Array<UInt8>) {
        self.mimeType = mimeType;
        self.decodedBytes = decodedBytes;
    }

    internal static let wireFieldNames: Array<String> = ["mime_type", "decoded_bytes"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "mime_type", value: .string(self.mimeType));
        wireObject.appendEntry(key: "decoded_bytes", value: .string(Base64Bytes.encode(imageFileBytes: self.decodedBytes)));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatImageInput {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedImage = ChatImageInput(
            mimeType: try wireObject.decodeString(fieldName: "mime_type"),
            decodedBytes: try Base64Bytes.decode(encodedText: try wireObject.decodeString(fieldName: "decoded_bytes")));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatImageInput.wireFieldNames);
        return parsedImage;
    }
}

/// One assistant function call retained in conversation history.
public struct ChatAssistantToolCall: Equatable {
    /// Client-visible call ID used to correlate the later tool response.
    public let id: String;
    /// The requested function and JSON argument document.
    public let function: ChatAssistantToolFunction;

    public init(id: String, function: ChatAssistantToolFunction) {
        self.id = id;
        self.function = function;
    }

    internal static let wireFieldNames: Array<String> = ["id", "function"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "id", value: .string(self.id));
        wireObject.appendEntry(key: "function", value: self.function.wireValue());
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatAssistantToolCall {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedToolCall = ChatAssistantToolCall(
            id: try wireObject.decodeString(fieldName: "id"),
            function: try ChatAssistantToolFunction.fromWireValue(try wireObject.requireObjectValue(fieldName: "function")));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatAssistantToolCall.wireFieldNames);
        return parsedToolCall;
    }
}

/// One named JSON function invocation retained in assistant history.
public struct ChatAssistantToolFunction: Equatable {
    /// The function name.
    public let name: String;
    /// Canonical JSON arguments serialized by the supervisor.
    public let argumentsJson: String;

    public init(name: String, argumentsJson: String) {
        self.name = name;
        self.argumentsJson = argumentsJson;
    }

    internal static let wireFieldNames: Array<String> = ["name", "arguments_json"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "name", value: .string(self.name));
        wireObject.appendEntry(key: "arguments_json", value: .string(self.argumentsJson));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatAssistantToolFunction {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedFunction = ChatAssistantToolFunction(
            name: try wireObject.decodeString(fieldName: "name"),
            argumentsJson: try wireObject.decodeString(fieldName: "arguments_json"));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatAssistantToolFunction.wireFieldNames);
        return parsedFunction;
    }
}
