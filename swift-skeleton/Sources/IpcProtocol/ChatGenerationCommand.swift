import Foundation;

/// One bounded structured chat-generation command for the local inference worker.
public struct ChatGenerationCommand: Equatable {
    public let requestId: RequestId;
    /// The exact worker-advertised model ID the request targets.
    public let model: String;
    /// Ordered system, user, assistant, and tool conversation history.
    public let messages: Array<ChatMessage>;
    /// Functions that the model may request but never execute itself.
    public let tools: Array<ChatToolDefinition>;
    /// The caller's function-selection mode.
    public let toolChoice: ChatToolChoice;
    /// Bounded sampling and output settings.
    public let settings: ChatGenerationSettings;
    /// Token-masked structured generation. Absent means ordinary sampling.
    public let structuredGeneration: StructuredGenerationConstraint?;

    public init(
        requestId: RequestId,
        model: String,
        messages: Array<ChatMessage>,
        tools: Array<ChatToolDefinition>,
        toolChoice: ChatToolChoice,
        settings: ChatGenerationSettings,
        structuredGeneration: StructuredGenerationConstraint?
    ) {
        self.requestId = requestId;
        self.model = model;
        self.messages = messages;
        self.tools = tools;
        self.toolChoice = toolChoice;
        self.settings = settings;
        self.structuredGeneration = structuredGeneration;
    }

    internal static let wireFieldNames: Array<String> = [
        "request_id", "model", "messages", "tools", "tool_choice", "settings",
        "structured_generation",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "request_id", value: self.requestId.wireValue());
        wireObject.appendEntry(key: "model", value: .string(self.model));
        wireObject.appendEntry(key: "messages", value: JsonWireValue.mappedArray(self.messages, mappedWireValue: { (message: ChatMessage) -> JsonWireValue in message.wireValue() }));
        wireObject.appendEntry(key: "tools", value: JsonWireValue.mappedArray(self.tools, mappedWireValue: { (tool: ChatToolDefinition) -> JsonWireValue in tool.wireValue() }));
        wireObject.appendEntry(key: "tool_choice", value: self.toolChoice.wireValue());
        wireObject.appendEntry(key: "settings", value: self.settings.wireValue());
        if let unwrappedConstraint = self.structuredGeneration {
            wireObject.appendEntry(key: "structured_generation", value: unwrappedConstraint.wireValue());
        }
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatGenerationCommand {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCommand = ChatGenerationCommand(
            requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")),
            model: try wireObject.decodeString(fieldName: "model"),
            messages: try wireObject.decodeArray(fieldName: "messages", mappedElement: { (elementWireValue: JsonWireValue) throws -> ChatMessage in
                try ChatMessage.fromWireValue(elementWireValue)
            }),
            tools: try wireObject.decodeArray(fieldName: "tools", mappedElement: { (elementWireValue: JsonWireValue) throws -> ChatToolDefinition in
                try ChatToolDefinition.fromWireValue(elementWireValue)
            }),
            toolChoice: try ChatToolChoice.fromWireValue(try wireObject.requireObjectValue(fieldName: "tool_choice")),
            settings: try ChatGenerationSettings.fromWireValue(try wireObject.requireObjectValue(fieldName: "settings")),
            structuredGeneration: try ChatGenerationCommand.decodeOptionalStructuredGeneration(wireObject: wireObject));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatGenerationCommand.wireFieldNames);
        return parsedCommand;
    }

    private static func decodeOptionalStructuredGeneration(wireObject: JsonWireObject) throws -> StructuredGenerationConstraint? {
        guard let constraintWireValue = wireObject.value(forKey: "structured_generation") else {
            return nil;
        }
        if constraintWireValue.isNull {
            return nil;
        }
        return try StructuredGenerationConstraint.fromWireValue(constraintWireValue);
    }
}

/// Bounded generation settings that influence model execution.
public struct ChatGenerationSettings: Equatable {
    /// The maximum number of generated tokens for this request.
    public let maxOutputTokens: UInt16;
    /// Optional sampling temperature in thousandths.
    public let temperatureThousandths: UInt16?;
    /// Optional nucleus-sampling threshold in thousandths.
    public let topPThousandths: UInt16?;
    /// Optional deterministic sampler seed.
    public let seed: UInt64?;
    /// Maximum tokens the model may spend inside the thinking block before
    /// being forced to close it. `nil` means no budget — the model thinks
    /// freely up to `maxOutputTokens`.
    public let thinkingBudget: UInt16?;

    public init(
        maxOutputTokens: UInt16,
        temperatureThousandths: UInt16?,
        topPThousandths: UInt16?,
        seed: UInt64?,
        thinkingBudget: UInt16?
    ) {
        self.maxOutputTokens = maxOutputTokens;
        self.temperatureThousandths = temperatureThousandths;
        self.topPThousandths = topPThousandths;
        self.seed = seed;
        self.thinkingBudget = thinkingBudget;
    }

    internal static let wireFieldNames: Array<String> = [
        "max_output_tokens", "temperature_thousandths", "top_p_thousandths", "seed", "thinking_budget",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "max_output_tokens", value: .unsignedInteger(UInt64(self.maxOutputTokens)));
        wireObject.appendEntry(key: "temperature_thousandths", value: ChatGenerationSettings.optionalUInt16WireValue(self.temperatureThousandths));
        wireObject.appendEntry(key: "top_p_thousandths", value: ChatGenerationSettings.optionalUInt16WireValue(self.topPThousandths));
        wireObject.appendEntry(key: "seed", value: ChatGenerationSettings.optionalUInt64WireValue(self.seed));
        wireObject.appendEntry(key: "thinking_budget", value: ChatGenerationSettings.optionalUInt16WireValue(self.thinkingBudget));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> ChatGenerationSettings {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedSettings = ChatGenerationSettings(
            maxOutputTokens: try wireObject.decodeUInt16(fieldName: "max_output_tokens"),
            temperatureThousandths: try wireObject.decodeOptionalUInt16(fieldName: "temperature_thousandths"),
            topPThousandths: try wireObject.decodeOptionalUInt16(fieldName: "top_p_thousandths"),
            seed: try wireObject.decodeOptionalUInt64(fieldName: "seed"),
            thinkingBudget: try wireObject.decodeOptionalUInt16AllowingAbsent(fieldName: "thinking_budget"));
        try wireObject.rejectUnknownFields(allowedFieldNames: ChatGenerationSettings.wireFieldNames);
        return parsedSettings;
    }

    private static func optionalUInt16WireValue(_ rawValue: UInt16?) -> JsonWireValue {
        guard let unwrappedValue = rawValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }

    private static func optionalUInt64WireValue(_ rawValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue = rawValue else {
            return .null;
        }
        return .unsignedInteger(unwrappedValue);
    }
}
