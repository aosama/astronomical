import Foundation;

/// A bounded semantic validation failure in one structured chat command; public
/// because ProtocolError carries it as an associated payload.
public enum ChatGenerationValidationError: Error, Equatable, CustomStringConvertible {
    case emptyModelId;
    case emptyMessages;
    case outputTokenCountOutOfRange(actualOutputTokens: UInt16, maximumOutputTokens: UInt16);
    case temperatureOutOfRange(actualTemperatureThousandths: UInt16, maximumTemperatureThousandths: UInt16);
    case topPOutOfRange(actualTopPThousandths: UInt16, maximumTopPThousandths: UInt16);
    case systemMessageMustBeFirst(messageIndex: Int);
    case unknownToolResultId(toolCallId: String);
    case duplicateToolResultId(toolCallId: String);
    case duplicateToolDefinitionName(functionName: String);
    case emptyToolDefinitionName;
    case toolChoiceNamesUnknownFunction(functionName: String);
    case unsupportedToolChoice(mode: String);
    case invalidToolSchema(functionName: String);
    case toolSchemaMustBeObject(functionName: String);
    case toolSchemaNestingTooDeep(functionName: String, actualSchemaNestingDepth: Int, maximumSchemaNestingDepth: Int);
    case duplicateAssistantToolCallId(toolCallId: String);
    case invalidAssistantToolCallArguments(toolCallId: String);
    case assistantToolCallArgumentsMustBeObject(toolCallId: String);
    case emptyAssistantToolCallId;
    case emptyAssistantToolCallFunctionName;
    case emptyToolResultId;
    case qwenThinkingChannelSeedTooLarge(actualSeedBytes: Int, maximumSeedBytes: Int);

    public var description: String {
        switch (self) {
        case .emptyModelId:
            return "model ID must not be empty";
        case .emptyMessages:
            return "messages must not be empty";
        case let .outputTokenCountOutOfRange(actualOutputTokens, maximumOutputTokens):
            return "output token count is \(actualOutputTokens), outside the 1..=\(maximumOutputTokens) token range";
        case let .temperatureOutOfRange(actualTemperatureThousandths, maximumTemperatureThousandths):
            return "temperature is \(actualTemperatureThousandths) thousandths, exceeding the \(maximumTemperatureThousandths)-thousandths limit";
        case let .topPOutOfRange(actualTopPThousandths, maximumTopPThousandths):
            return "top_p is \(actualTopPThousandths) thousandths, exceeding the \(maximumTopPThousandths)-thousandths limit";
        case let .systemMessageMustBeFirst(messageIndex):
            return "system message at position \(messageIndex) must be the first message";
        case let .unknownToolResultId(toolCallId):
            return "tool result refers to unknown assistant tool-call ID '\(toolCallId)'";
        case let .duplicateToolResultId(toolCallId):
            return "tool result for assistant tool-call ID '\(toolCallId)' appears more than once";
        case let .duplicateToolDefinitionName(functionName):
            return "tool function name '\(functionName)' appears more than once";
        case .emptyToolDefinitionName:
            return "tool function name must not be empty";
        case let .toolChoiceNamesUnknownFunction(functionName):
            return "tool choice names undeclared function '\(functionName)'";
        case let .unsupportedToolChoice(mode):
            return "tool choice mode '\(mode)' is unsupported by the current worker";
        case let .invalidToolSchema(functionName):
            return "tool schema for function '\(functionName)' is invalid JSON";
        case let .toolSchemaMustBeObject(functionName):
            return "tool schema for function '\(functionName)' must be a JSON object";
        case let .toolSchemaNestingTooDeep(functionName, actualSchemaNestingDepth, maximumSchemaNestingDepth):
            return "tool schema for function '\(functionName)' has nesting depth \(actualSchemaNestingDepth), exceeding \(maximumSchemaNestingDepth)";
        case let .duplicateAssistantToolCallId(toolCallId):
            return "assistant tool-call ID '\(toolCallId)' appears more than once";
        case let .invalidAssistantToolCallArguments(toolCallId):
            return "assistant tool-call ID '\(toolCallId)' has invalid JSON arguments";
        case let .assistantToolCallArgumentsMustBeObject(toolCallId):
            return "assistant tool-call ID '\(toolCallId)' arguments must be a JSON object";
        case .emptyAssistantToolCallId:
            return "assistant tool-call ID must not be empty";
        case .emptyAssistantToolCallFunctionName:
            return "assistant tool-call function name must not be empty";
        case .emptyToolResultId:
            return "tool result ID must not be empty";
        case let .qwenThinkingChannelSeedTooLarge(actualSeedBytes, maximumSeedBytes):
            return "Qwen thinking-channel seed has \(actualSeedBytes) bytes, exceeding \(maximumSeedBytes)";
        }
    }
}

private enum ChatGenerationValidation {

    private static let maximumChatToolSchemaNestingDepth: Int = 32;
    private static let maximumChatOutputTokens: UInt16 = UInt16.max;
    private static let maximumChatTemperatureThousandths: UInt16 = 2_000;
    private static let maximumChatTopPThousandths: UInt16 = 1_000;

    static func validateCommand(_ command: ChatGenerationCommand) throws {
        if command.model.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
            throw ChatGenerationValidationError.emptyModelId;
        }
        if command.messages.isEmpty {
            throw ChatGenerationValidationError.emptyMessages;
        }
        if command.settings.maxOutputTokens == 0 {
            throw ChatGenerationValidationError.outputTokenCountOutOfRange(
                actualOutputTokens: command.settings.maxOutputTokens,
                maximumOutputTokens: ChatGenerationValidation.maximumChatOutputTokens);
        }
        if let qwenThinkingChannelSeed = command.qwenThinkingChannelSeed,
            qwenThinkingChannelSeed.utf8.count > ChatGenerationCommand.maximumQwenThinkingChannelSeedBytes {
            throw ChatGenerationValidationError.qwenThinkingChannelSeedTooLarge(
                actualSeedBytes: qwenThinkingChannelSeed.utf8.count,
                maximumSeedBytes: ChatGenerationCommand.maximumQwenThinkingChannelSeedBytes);
        }
        if let temperatureThousandths = command.settings.temperatureThousandths,
            temperatureThousandths > ChatGenerationValidation.maximumChatTemperatureThousandths {
            throw ChatGenerationValidationError.temperatureOutOfRange(
                actualTemperatureThousandths: temperatureThousandths,
                maximumTemperatureThousandths: ChatGenerationValidation.maximumChatTemperatureThousandths);
        }
        if let topPThousandths = command.settings.topPThousandths,
            topPThousandths > ChatGenerationValidation.maximumChatTopPThousandths {
            throw ChatGenerationValidationError.topPOutOfRange(
                actualTopPThousandths: topPThousandths,
                maximumTopPThousandths: ChatGenerationValidation.maximumChatTopPThousandths);
        }
        let declaredToolNames: Set<String> = try ChatGenerationValidation.validateTools(command);
        try ChatGenerationValidation.validateMessages(command);
        switch command.toolChoice {
        case .auto, .none:
            break;
        case .required:
            throw ChatGenerationValidationError.unsupportedToolChoice(mode: "required");
        case let .function(functionName):
            if declaredToolNames.contains(functionName) == false {
                throw ChatGenerationValidationError.toolChoiceNamesUnknownFunction(functionName: functionName);
            }
            throw ChatGenerationValidationError.unsupportedToolChoice(mode: "function");
        }
    }

    private static func validateTools(_ command: ChatGenerationCommand) throws -> Set<String> {
        var declaredToolNames: Set<String> = Set<String>();
        for toolDefinition in command.tools {
            if toolDefinition.name.isEmpty {
                throw ChatGenerationValidationError.emptyToolDefinitionName;
            }
            try ChatGenerationValidation.validateToolSchema(toolDefinition);
            if declaredToolNames.contains(toolDefinition.name) {
                throw ChatGenerationValidationError.duplicateToolDefinitionName(functionName: toolDefinition.name);
            }
            declaredToolNames.insert(toolDefinition.name);
        }
        return declaredToolNames;
    }

    private static func validateMessages(_ command: ChatGenerationCommand) throws {
        var activeToolCallIds: Set<String> = Set<String>();
        var completedToolResultIds: Set<String> = Set<String>();
        for (messageIndex, chatMessage) in command.messages.enumerated() {
            if case .system = chatMessage, messageIndex != 0 {
                throw ChatGenerationValidationError.systemMessageMustBeFirst(messageIndex: messageIndex);
            }
            switch chatMessage {
            case let .assistant(_, _, toolCalls):
                for toolCall in toolCalls {
                    try ChatGenerationValidation.validateAssistantToolCall(toolCall);
                    let argumentWireValue: JsonWireValue;
                    do {
                        argumentWireValue = try JsonWireParser.parseDocument(
                            documentBytes: Data(toolCall.function.argumentsJson.utf8));
                    } catch {
                        throw ChatGenerationValidationError.invalidAssistantToolCallArguments(toolCallId: toolCall.id);
                    }
                    guard case .object = argumentWireValue else {
                        throw ChatGenerationValidationError.assistantToolCallArgumentsMustBeObject(toolCallId: toolCall.id);
                    }
                    if activeToolCallIds.contains(toolCall.id) {
                        throw ChatGenerationValidationError.duplicateAssistantToolCallId(toolCallId: toolCall.id);
                    }
                    activeToolCallIds.insert(toolCall.id);
                    completedToolResultIds.remove(toolCall.id);
                }
            case let .tool(toolCallId, _):
                if toolCallId.isEmpty {
                    throw ChatGenerationValidationError.emptyToolResultId;
                }
                if activeToolCallIds.remove(toolCallId) == nil {
                    if completedToolResultIds.contains(toolCallId) {
                        throw ChatGenerationValidationError.duplicateToolResultId(toolCallId: toolCallId);
                    }
                    throw ChatGenerationValidationError.unknownToolResultId(toolCallId: toolCallId);
                }
                if completedToolResultIds.contains(toolCallId) {
                    throw ChatGenerationValidationError.duplicateToolResultId(toolCallId: toolCallId);
                }
                completedToolResultIds.insert(toolCallId);
            case .system, .user:
                break;
            }
        }
    }

    private static func validateAssistantToolCall(_ toolCall: ChatAssistantToolCall) throws {
        if toolCall.id.isEmpty {
            throw ChatGenerationValidationError.emptyAssistantToolCallId;
        }
        if toolCall.function.name.isEmpty {
            throw ChatGenerationValidationError.emptyAssistantToolCallFunctionName;
        }
    }

    private static func validateToolSchema(_ toolDefinition: ChatToolDefinition) throws {
        let schemaWireValue: JsonWireValue;
        do {
            schemaWireValue = try JsonWireParser.parseDocument(
                documentBytes: Data(toolDefinition.parametersJson.utf8));
        } catch {
            throw ChatGenerationValidationError.invalidToolSchema(functionName: toolDefinition.name);
        }
        guard case .object = schemaWireValue else {
            throw ChatGenerationValidationError.toolSchemaMustBeObject(functionName: toolDefinition.name);
        }
        let schemaNestingDepth: Int = ChatGenerationValidation.jsonNestingDepth(schemaWireValue);
        if schemaNestingDepth > ChatGenerationValidation.maximumChatToolSchemaNestingDepth {
            throw ChatGenerationValidationError.toolSchemaNestingTooDeep(
                functionName: toolDefinition.name,
                actualSchemaNestingDepth: schemaNestingDepth,
                maximumSchemaNestingDepth: ChatGenerationValidation.maximumChatToolSchemaNestingDepth);
        }
    }

    private static func jsonNestingDepth(_ wireValue: JsonWireValue) -> Int {
        switch wireValue {
        case let .array(elementValues):
            let childDepths: Array<Int> = elementValues.map({ (elementValue: JsonWireValue) -> Int in ChatGenerationValidation.jsonNestingDepth(elementValue) });
            return 1 + (childDepths.max() ?? 0);
        case let .object(objectValue):
            let childDepths: Array<Int> = objectValue.entries.map({ (entry: (key: String, value: JsonWireValue)) -> Int in ChatGenerationValidation.jsonNestingDepth(entry.value) });
            return 1 + (childDepths.max() ?? 0);
        case .null, .boolean, .unsignedInteger, .signedInteger, .double, .float32, .string:
            return 0;
        }
    }
}

extension ChatGenerationCommand {

    /// Independently validates structured chat history after it crosses the worker boundary.
    public func validate() throws {
        return try ChatGenerationValidation.validateCommand(self);
    }
}
