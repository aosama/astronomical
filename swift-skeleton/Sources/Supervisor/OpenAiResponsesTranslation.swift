import Foundation;

import IpcProtocol;
import RestContract;

/// A typed failure while translating public OpenAI Responses data into the
/// worker protocol. Port of apps/supervisor/src/openai_responses_translation.rs's
/// OpenAiResponsesTranslationError; the texts mirror the thiserror messages.
public enum OpenAiResponsesTranslationError: Error, CustomStringConvertible {

    case publicValidation(OpenAiResponsesValidationError);
    case outputTokenCountTooLarge(actualOutputTokens: UInt32);
    case samplingPrecisionUnsupported(parameterName: String, requestedValue: Float);
    case toolSchemaSerialization(String);
    case ipcValidation(ChatGenerationValidationError);
    case thinkingBudgetTooLarge;

    public var description: String {
        switch (self) {
        case let .publicValidation(validationError):
            return "OpenAI Responses request validation failed: "
                + (validationError.errorDescription ?? String(describing: validationError));
        case let .outputTokenCountTooLarge(actualOutputTokens):
            return "OpenAI Responses output token count \(actualOutputTokens) does not fit IPC";
        case let .samplingPrecisionUnsupported(parameterName, requestedValue):
            return "\(parameterName) value \(requestedValue) cannot be represented in thousandths";
        case let .toolSchemaSerialization(serializationError):
            return "Responses function-tool schema could not be serialized: \(serializationError)";
        case let .ipcValidation(validationError):
            return "translated Responses IPC command failed validation: \(validationError)";
        case .thinkingBudgetTooLarge:
            return "thinking_budget does not fit the worker representation";
        }
    }
}

/// Translates one validated Responses request into the existing worker
/// command. Port of apps/supervisor/src/openai_responses_translation.rs.
enum OpenAiResponsesTranslation {

    private static let chronologicalInstructionOpeningTag: String = "<system-update>\n";
    private static let chronologicalInstructionClosingTag: String = "\n</system-update>";

    /// Translates one decoded public Responses request, rejecting it through
    /// the public validation error when it fails there.
    static func translateRequest(
        _ responsesRequest: OpenAiResponsesRequest,
        requestId: RequestId
    ) throws -> ChatGenerationCommand {
        let requestParts: OpenAiResponsesRequestParts;
        do {
            requestParts = try responsesRequest.intoParts();
        } catch let validationRejection as OpenAiResponsesValidationError {
            throw OpenAiResponsesTranslationError.publicValidation(validationRejection);
        }
        return try OpenAiResponsesTranslation.translateRequestParts(requestParts, requestId: requestId);
    }

    static func translateRequestParts(
        _ requestParts: OpenAiResponsesRequestParts,
        requestId: RequestId
    ) throws -> ChatGenerationCommand {
        var translatedMessages: Array<ChatMessage> = Array();
        if let instructionsText: String = requestParts.instructions {
            translatedMessages.append(.system(content: instructionsText));
        }
        switch (requestParts.input) {
        case let .text(inputText):
            translatedMessages.append(.user(content: inputText, images: Array()));
        case let .items(responseInputItems):
            OpenAiResponsesTranslation.translateResponseInputItems(
                responseInputItems, &translatedMessages);
        }
        guard let maximumOutputTokens: UInt16 = UInt16(exactly: requestParts.maximumOutputTokens) else {
            throw OpenAiResponsesTranslationError.outputTokenCountTooLarge(
                actualOutputTokens: requestParts.maximumOutputTokens);
        }
        var thinkingBudgetTokens: UInt16? = nil;
        if let requestedThinkingBudget: UInt32 = requestParts.thinkingBudget {
            guard let translatedThinkingBudget: UInt16 = UInt16(exactly: requestedThinkingBudget) else {
                throw OpenAiResponsesTranslationError.thinkingBudgetTooLarge;
            }
            thinkingBudgetTokens = translatedThinkingBudget;
        }
        if let structuredOutput: OpenAiStructuredOutput = requestParts.structuredOutput {
            ChatSchemaConstraint.insertJsonOutputInstruction(
                &translatedMessages,
                jsonOutputInstruction: structuredOutput.jsonOutputInstruction());
        }
        let chatGenerationCommand: ChatGenerationCommand = ChatGenerationCommand(
            requestId: requestId,
            model: requestParts.model,
            messages: translatedMessages,
            tools: try OpenAiResponsesTranslation.translateTools(requestParts.tools),
            toolChoice: OpenAiResponsesTranslation.translateToolChoice(requestParts.toolChoice),
            settings: ChatGenerationSettings(
                maxOutputTokens: maximumOutputTokens,
                temperatureThousandths: try OpenAiResponsesTranslation.translateThousandths(
                    requestParts.temperature, parameterName: "temperature"),
                topPThousandths: try OpenAiResponsesTranslation.translateThousandths(
                    requestParts.topP, parameterName: "top_p"),
                seed: nil,
                // Thinking-enabled Qwen otherwise spends max_output_tokens
                // inside reasoning and never emits the structured JSON the
                // caller asked for.
                thinkingBudget: thinkingBudgetTokens),
            structuredGeneration: requestParts.enforcedStructuredGeneration.map(
                ChatSchemaConstraint.constraintFromEnforcedGeneration));
        do {
            try chatGenerationCommand.validate();
        } catch let ipcRejection as ChatGenerationValidationError {
            throw OpenAiResponsesTranslationError.ipcValidation(ipcRejection);
        }
        return chatGenerationCommand;
    }

    /// One assistant turn still being assembled from adjacent reasoning,
    /// assistant-message, and function-call input items.
    private struct PendingAssistantMessage {

        var content: String = String();
        var reasoningContent: String = String();
        var toolCalls: Array<ChatAssistantToolCall> = Array();

        mutating func flushInto(
            _ translatedMessages: inout Array<ChatMessage>
        ) -> Void {
            if self.content.isEmpty && self.reasoningContent.isEmpty && self.toolCalls.isEmpty {
                return;
            }
            var flushedContent: String? = nil;
            if self.content.isEmpty == false {
                flushedContent = self.content;
                self.content = String();
            }
            var flushedReasoningContent: String? = nil;
            if self.reasoningContent.isEmpty == false {
                flushedReasoningContent = self.reasoningContent;
                self.reasoningContent = String();
            }
            let flushedToolCalls: Array<ChatAssistantToolCall> = self.toolCalls;
            self.toolCalls = Array();
            translatedMessages.append(.assistant(
                content: flushedContent,
                reasoningContent: flushedReasoningContent,
                toolCalls: flushedToolCalls));
        }
    }

    private static func translateResponseInputItems(
        _ responseInputItems: Array<OpenAiResponseInputItemParts>,
        _ translatedMessages: inout Array<ChatMessage>
    ) -> Void {
        var pendingAssistantMessage: PendingAssistantMessage = PendingAssistantMessage();
        for responseInputItem: OpenAiResponseInputItemParts in responseInputItems {
            switch (responseInputItem) {
            case let .systemMessage(instructionContent):
                pendingAssistantMessage.flushInto(&translatedMessages);
                OpenAiResponsesTranslation.appendInstruction(
                    &translatedMessages, instructionContent: instructionContent);
            case let .developerMessage(instructionContent):
                pendingAssistantMessage.flushInto(&translatedMessages);
                OpenAiResponsesTranslation.appendInstruction(
                    &translatedMessages, instructionContent: instructionContent);
            case let .userMessage(userContent, userImages):
                pendingAssistantMessage.flushInto(&translatedMessages);
                translatedMessages.append(.user(
                    content: userContent,
                    images: userImages.map({ (imageInput: ImageInput.OpenAiImageInput) -> ChatImageInput in
                        return ChatImageInput(
                            mimeType: imageInput.mimeType(),
                            decodedBytes: imageInput.decodedBytes());
                    })));
            case let .assistantMessage(assistantContent):
                pendingAssistantMessage.content += assistantContent;
            case let .reasoning(reasoningContent):
                pendingAssistantMessage.reasoningContent += reasoningContent;
            case let .functionCall(callId, functionName, argumentsJson):
                pendingAssistantMessage.toolCalls.append(ChatAssistantToolCall(
                    id: callId,
                    function: ChatAssistantToolFunction(
                        name: functionName,
                        argumentsJson: argumentsJson)));
            case let .functionCallOutput(callId, toolOutput):
                pendingAssistantMessage.flushInto(&translatedMessages);
                translatedMessages.append(.tool(toolCallId: callId, content: toolOutput));
            }
        }
        pendingAssistantMessage.flushInto(&translatedMessages);
    }

    /// A system-flavored instruction after the first message becomes a
    /// chronological user update so prompt templates keep their
    /// root-instruction assumption while mid-conversation instructions survive.
    private static func appendInstruction(
        _ translatedMessages: inout Array<ChatMessage>,
        instructionContent: String
    ) -> Void {
        if translatedMessages.isEmpty {
            translatedMessages.append(.system(content: instructionContent));
            return;
        }
        if case let .user(priorUserContent, priorUserImages) = translatedMessages.last {
            let appendedUserContent: String = priorUserContent + "\n"
                + OpenAiResponsesTranslation.escapedInstructionText(instructionContent);
            translatedMessages[translatedMessages.count - 1] = .user(
                content: appendedUserContent,
                images: priorUserImages);
            return;
        }
        translatedMessages.append(.user(
            content: OpenAiResponsesTranslation.escapedInstructionText(instructionContent),
            images: Array()));
    }

    private static func escapedInstructionText(_ instructionContent: String) -> String {
        var escapedInstruction: String = chronologicalInstructionOpeningTag;
        for instructionCharacter: Character in instructionContent {
            switch (instructionCharacter) {
            case "&": escapedInstruction += "&amp;";
            case "<": escapedInstruction += "&lt;";
            case ">": escapedInstruction += "&gt;";
            default: escapedInstruction.append(instructionCharacter);
            }
        }
        escapedInstruction += chronologicalInstructionClosingTag;
        return escapedInstruction;
    }

    private static func translateTools(
        _ toolParts: Array<OpenAiResponseToolDefinitionParts>
    ) throws -> Array<ChatToolDefinition> {
        var translatedTools: Array<ChatToolDefinition> = Array();
        translatedTools.reserveCapacity(toolParts.count);
        for functionTool: OpenAiResponseToolDefinitionParts in toolParts {
            let parametersJsonText: String;
            do {
                parametersJsonText = try functionTool.canonicalParametersJson();
            } catch {
                throw OpenAiResponsesTranslationError.toolSchemaSerialization(String(describing: error));
            }
            translatedTools.append(ChatToolDefinition(
                name: functionTool.name,
                description: functionTool.description,
                parametersJson: parametersJsonText));
        }
        return translatedTools;
    }

    private static func translateToolChoice(
        _ toolChoiceParts: OpenAiResponseToolChoiceParts
    ) -> ChatToolChoice {
        switch (toolChoiceParts) {
        case .auto:
            return .auto;
        case .none:
            return .none;
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
            throw OpenAiResponsesTranslationError.samplingPrecisionUnsupported(
                parameterName: parameterName,
                requestedValue: samplingParameterValue);
        }
        guard roundedSamplingParameter.isFinite, roundedSamplingParameter >= 0,
              let truncatedThousandths: UInt32 = UInt32(exactly: roundedSamplingParameter),
              let thousandthsValue: UInt16 = UInt16(exactly: truncatedThousandths) else {
            throw OpenAiResponsesTranslationError.samplingPrecisionUnsupported(
                parameterName: parameterName,
                requestedValue: samplingParameterValue);
        }
        return thousandthsValue;
    }
}
