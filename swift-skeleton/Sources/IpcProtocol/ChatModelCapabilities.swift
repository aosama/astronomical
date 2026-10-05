import Foundation;

/// Structured-chat capabilities reported by one ready worker model.
public struct ChatModelCapabilities: Equatable {
    /// Whether the worker emits reasoning separately from assistant-visible text.
    public let supportsReasoning: Bool;
    /// Whether the worker emits complete validated function calls.
    public let supportsToolCalls: Bool;
    /// Whether the model supports image input (vision).
    public let hasVision: Bool;
    /// Maximum prompt tokens a client may send when reserving one generation
    /// position. The engine enforces the selected prompt plus generation budget
    /// against the shared context window at admission time.
    public let maxInputTokens: UInt32;
    /// Independent per-request output-token ceiling. The prompt must leave
    /// enough context positions for the selected output budget.
    public let maxOutputTokens: UInt32;
    /// Total prompt plus generation position capacity of the loaded model
    /// (Qwen3.5-MoE `max_position_embeddings`).
    public let contextWindow: UInt32;

    public init(
        supportsReasoning: Bool,
        supportsToolCalls: Bool,
        hasVision: Bool,
        maxInputTokens: UInt32,
        maxOutputTokens: UInt32,
        contextWindow: UInt32
    ) {
        self.supportsReasoning = supportsReasoning;
        self.supportsToolCalls = supportsToolCalls;
        self.hasVision = hasVision;
        self.maxInputTokens = maxInputTokens;
        self.maxOutputTokens = maxOutputTokens;
        self.contextWindow = contextWindow;
    }

    internal static let wireFieldNames: Array<String> = [
        "supports_reasoning", "supports_tool_calls", "has_vision",
        "max_input_tokens", "max_output_tokens", "context_window",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "supports_reasoning", value: .boolean(self.supportsReasoning));
        wireObject.appendEntry(key: "supports_tool_calls", value: .boolean(self.supportsToolCalls));
        wireObject.appendEntry(key: "has_vision", value: .boolean(self.hasVision));
        wireObject.appendEntry(key: "max_input_tokens", value: .unsignedInteger(UInt64(self.maxInputTokens)));
        wireObject.appendEntry(key: "max_output_tokens", value: .unsignedInteger(UInt64(self.maxOutputTokens)));
        wireObject.appendEntry(key: "context_window", value: .unsignedInteger(UInt64(self.contextWindow)));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatModelCapabilities {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCapabilities = ChatModelCapabilities(
            supportsReasoning: try wireObject.decodeBool(fieldName: "supports_reasoning"),
            supportsToolCalls: try wireObject.decodeBool(fieldName: "supports_tool_calls"),
            hasVision: try wireObject.decodeBool(fieldName: "has_vision"),
            maxInputTokens: try wireObject.decodeUInt32(fieldName: "max_input_tokens"),
            maxOutputTokens: try wireObject.decodeUInt32(fieldName: "max_output_tokens"),
            contextWindow: try wireObject.decodeUInt32(fieldName: "context_window"));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatModelCapabilities.wireFieldNames);
        return parsedCapabilities;
    }
}
