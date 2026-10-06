import Foundation;

import IpcProtocol;
import RestContract;

/// A typed failure while translating public OpenAI chat data into the worker
/// protocol. Port of apps/supervisor/src/openai_chat_translation.rs's
/// OpenAiChatTranslationError; the texts mirror the thiserror messages the
/// REST surface has always shown.
public enum OpenAiChatTranslationError: Error, CustomStringConvertible {

    case publicValidation(OpenAiChatCompletionValidationError);
    case outputTokenCountTooLarge(actualOutputTokens: UInt32);
    case thinkingBudgetTooLarge;
    case samplingPrecisionUnsupported(parameterName: String, requestedValue: Float);
    case unsupportedToolChoice(mode: String);
    case ipcValidation(ChatGenerationValidationError);

    public var description: String {
        switch (self) {
        case let .publicValidation(validationError):
            return "OpenAI chat request validation failed: "
                + (validationError.errorDescription ?? String(describing: validationError));
        case let .outputTokenCountTooLarge(actualOutputTokens):
            return "OpenAI output token count \(actualOutputTokens) does not fit the IPC representation";
        case .thinkingBudgetTooLarge:
            return "thinking_budget exceeds the maximum representable token count";
        case let .samplingPrecisionUnsupported(parameterName, requestedValue):
            return "\(parameterName) value \(requestedValue) cannot be represented in thousandths";
        case let .unsupportedToolChoice(mode):
            return "tool choice mode '\(mode)' is unsupported";
        case let .ipcValidation(validationError):
            return "translated chat IPC command failed validation: \(validationError)";
        }
    }
}

/// Translates one validated public OpenAI Chat Completions request into
/// independent IPC data. Port of apps/supervisor/src/openai_chat_translation.rs.
enum OpenAiChatTranslation {

    private static let chronologicalSystemUpdateOpeningTag: String = "<system-update>\n";
    private static let chronologicalSystemUpdateClosingTag: String = "\n</system-update>";

    /// Translates one decoded public OpenAI Chat Completions request,
    /// rejecting it through the public validation error when it fails there.
    static func translateRequest(
        _ chatRequest: OpenAiChatCompletionRequest,
        requestId: RequestId
    ) throws -> ChatGenerationCommand {
        let requestParts: OpenAiChatCompletionRequestParts;
        do {
            requestParts = try chatRequest.intoParts();
        } catch let validationRejection as OpenAiChatCompletionValidationError {
            throw OpenAiChatTranslationError.publicValidation(validationRejection);
        }
        return try OpenAiChatTranslation.translateRequestParts(requestParts, requestId: requestId);
    }

    static func translateRequestParts(
        _ requestParts: OpenAiChatCompletionRequestParts,
        requestId: RequestId
    ) throws -> ChatGenerationCommand {
        guard let maximumOutputTokens: UInt16 = UInt16(exactly: requestParts.maximumOutputTokens) else {
            throw OpenAiChatTranslationError.outputTokenCountTooLarge(
                actualOutputTokens: requestParts.maximumOutputTokens);
        }
        var thinkingBudgetTokens: UInt16? = nil;
        if let requestedThinkingBudget: UInt32 = requestParts.thinkingBudget {
            guard let translatedThinkingBudget: UInt16 = UInt16(exactly: requestedThinkingBudget) else {
                throw OpenAiChatTranslationError.thinkingBudgetTooLarge;
            }
            thinkingBudgetTokens = translatedThinkingBudget;
        }
        var translatedMessages: Array<ChatMessage> = try translateMessages(requestParts.messages);
        if let structuredOutput: OpenAiStructuredOutput = requestParts.structuredOutput {
            // The first system message is the root instruction templates treat
            // as such, so the JSON rule is appended there; otherwise the rule
            // leads. The enforced token mask clamps the visible channel but
            // never tells the model what shape to plan, so the unenforced
            // prompt hint pairs with it.
            ChatSchemaConstraint.insertJsonOutputInstruction(
                &translatedMessages,
                jsonOutputInstruction: structuredOutput.jsonOutputInstruction());
        }
        let chatGenerationCommand: ChatGenerationCommand = ChatGenerationCommand(
            requestId: requestId,
            model: requestParts.model,
            messages: translatedMessages,
            tools: translateTools(requestParts.tools),
            toolChoice: try translateToolChoice(requestParts.toolChoice),
            settings: ChatGenerationSettings(
                maxOutputTokens: maximumOutputTokens,
                temperatureThousandths: try translateThousandths(
                    requestParts.temperature, parameterName: "temperature"),
                topPThousandths: try translateThousandths(requestParts.topP, parameterName: "top_p"),
                seed: requestParts.seed,
                thinkingBudget: thinkingBudgetTokens),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: requestParts.enforcedStructuredGeneration.map(
                ChatSchemaConstraint.constraintFromEnforcedGeneration));
        do {
            try chatGenerationCommand.validate();
        } catch let ipcRejection as ChatGenerationValidationError {
            throw OpenAiChatTranslationError.ipcValidation(ipcRejection);
        }
        return chatGenerationCommand;
    }

    private static func translateMessages(
        _ messageParts: Array<OpenAiChatMessageParts>
    ) throws -> Array<ChatMessage> {
        var translatedChatMessages: Array<ChatMessage> = Array();
        translatedChatMessages.reserveCapacity(messageParts.count);
        for messagePart: OpenAiChatMessageParts in messageParts {
            switch (messagePart) {
            case let .system(systemMessageContent):
                if translatedChatMessages.isEmpty {
                    translatedChatMessages.append(.system(content: systemMessageContent));
                } else {
                    // A later system turn is lowered to a chronological user
                    // update so prompt templates keep their root-instruction
                    // assumption while mid-conversation instructions survive.
                    appendChronologicalSystemUpdate(
                        &translatedChatMessages,
                        systemUpdateContent: systemMessageContent);
                }
            case let .user(userMessageContent, userMessageImages):
                translatedChatMessages.append(.user(
                    content: userMessageContent,
                    images: userMessageImages.map { (imageInput: ImageInput.OpenAiImageInput) -> ChatImageInput in
                        return ChatImageInput(
                            mimeType: imageInput.mimeType(),
                            decodedBytes: imageInput.decodedBytes());
                    }));
            case let .assistant(assistantMessageContent, assistantReasoningContent, assistantToolCallParts):
                translatedChatMessages.append(.assistant(
                    content: assistantMessageContent,
                    reasoningContent: assistantReasoningContent,
                    toolCalls: assistantToolCallParts.map { (toolCallPart: OpenAiAssistantToolCallParts) -> ChatAssistantToolCall in
                        return ChatAssistantToolCall(
                            id: toolCallPart.id,
                            function: ChatAssistantToolFunction(
                                name: toolCallPart.name,
                                argumentsJson: toolCallPart.argumentsJson));
                    }));
            case let .tool(toolCallId, toolResultContent):
                translatedChatMessages.append(.tool(toolCallId: toolCallId, content: toolResultContent));
            }
        }
        return translatedChatMessages;
    }

    private static func appendChronologicalSystemUpdate(
        _ translatedChatMessages: inout Array<ChatMessage>,
        systemUpdateContent: String
    ) -> Void {
        if case let .user(priorUserMessageContent, priorUserMessageImages) = translatedChatMessages.last {
            let appendedUserContent: String = priorUserMessageContent + "\n"
                + escapedSystemUpdateText(systemUpdateContent);
            translatedChatMessages[translatedChatMessages.count - 1] = .user(
                content: appendedUserContent,
                images: priorUserMessageImages);
            return;
        }
        translatedChatMessages.append(.user(
            content: escapedSystemUpdateText(systemUpdateContent),
            images: Array()));
    }

    private static func escapedSystemUpdateText(_ systemUpdateContent: String) -> String {
        var escapedUpdateText: String = chronologicalSystemUpdateOpeningTag;
        for systemUpdateCharacter: Character in systemUpdateContent {
            switch (systemUpdateCharacter) {
            case "&": escapedUpdateText += "&amp;";
            case "<": escapedUpdateText += "&lt;";
            case ">": escapedUpdateText += "&gt;";
            default: escapedUpdateText.append(systemUpdateCharacter);
            }
        }
        escapedUpdateText += chronologicalSystemUpdateClosingTag;
        return escapedUpdateText;
    }

    private static func translateTools(
        _ toolParts: Array<OpenAiToolDefinitionParts>
    ) -> Array<ChatToolDefinition> {
        return toolParts.map { (toolPart: OpenAiToolDefinitionParts) -> ChatToolDefinition in
            return ChatToolDefinition(
                name: toolPart.name,
                description: toolPart.description,
                parametersJson: toolPart.parametersJson);
        };
    }

    private static func translateToolChoice(
        _ toolChoiceMode: OpenAiToolChoiceMode
    ) throws -> ChatToolChoice {
        switch (toolChoiceMode) {
        case .auto:
            return .auto;
        case .none:
            return .none;
        case .required:
            throw OpenAiChatTranslationError.unsupportedToolChoice(mode: "required");
        case .function:
            throw OpenAiChatTranslationError.unsupportedToolChoice(mode: "function");
        }
    }

    private static func translateThousandths(
        _ samplingParameterValue: Float?,
        parameterName: String
    ) throws -> UInt16? {
        guard let samplingParameterValue = samplingParameterValue else {
            return nil;
        }
        let scaledSamplingParameter: Float = samplingParameterValue * 1_000.0;
        let roundedSamplingParameter: Float = scaledSamplingParameter.rounded();
        if abs(scaledSamplingParameter - roundedSamplingParameter) > 0.0001 {
            throw OpenAiChatTranslationError.samplingPrecisionUnsupported(
                parameterName: parameterName,
                requestedValue: samplingParameterValue);
        }
        guard roundedSamplingParameter.isFinite, roundedSamplingParameter >= 0,
              let truncatedThousandths: UInt32 = UInt32(exactly: roundedSamplingParameter),
              let thousandthsValue: UInt16 = UInt16(exactly: truncatedThousandths) else {
            throw OpenAiChatTranslationError.samplingPrecisionUnsupported(
                parameterName: parameterName,
                requestedValue: samplingParameterValue);
        }
        return thousandthsValue;
    }
}
