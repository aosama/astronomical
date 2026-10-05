// ResponsesInput+Parts.swift — RestContract
//
// Validation half of crates/rest-contract/src/openai_responses_input.rs: the
// into_parts translations that consume the wire shapes from
// ResponsesInput.swift into protocol-neutral parts.

import Foundation;
import IpcProtocol;

/// Validated Responses input ready for supervisor translation.
public enum OpenAiResponseInputParts: Equatable {
    /// One compact user input string.
    case text(String);
    /// Ordered validated input items.
    case items(Array<OpenAiResponseInputItemParts>);
}

/// One validated input item ready for supervisor translation.
public enum OpenAiResponseInputItemParts: Equatable {
    case systemMessage(content: String);
    case developerMessage(content: String);
    case userMessage(content: String, images: Array<ImageInput.OpenAiImageInput>);
    case assistantMessage(content: String);
    case reasoning(content: String);
    case functionCall(callId: String, name: String, argumentsJson: String);
    case functionCallOutput(callId: String, output: String);

    public func kindName() -> String {
        switch self {
        case .systemMessage: return "system_message";
        case .developerMessage: return "developer_message";
        case .userMessage: return "user_message";
        case .assistantMessage: return "assistant_message";
        case .reasoning: return "reasoning";
        case .functionCall: return "function_call";
        case .functionCallOutput: return "function_call_output";
        }
    }
}

extension OpenAiResponseInput {

    /// Validates and consumes this public input into protocol-neutral parts.
    internal func intoParts() throws -> OpenAiResponseInputParts {
        switch self {
        case .text(let inputText):
            return .text(inputText);
        case .items(let inputItems):
            if inputItems.isEmpty {
                throw OpenAiResponsesValidationError.emptyInputItems;
            }
            var translatedItems: Array<OpenAiResponseInputItemParts> =
                Array<OpenAiResponseInputItemParts>();
            translatedItems.reserveCapacity(inputItems.count);
            for inputItem: OpenAiResponseInputItem in inputItems {
                translatedItems.append(try inputItem.intoParts());
            }
            return .items(translatedItems);
        }
    }
}

extension OpenAiResponseInputItem {

    internal func intoParts() throws -> OpenAiResponseInputItemParts {
        switch self {
        case .message(let messageInput):
            return try messageInput.intoParts();
        case .reasoning(let reasoningInput):
            return try reasoningInput.intoParts();
        case .functionCall(let functionCallInput):
            return .functionCall(
                callId: functionCallInput.callId,
                name: functionCallInput.name,
                argumentsJson: functionCallInput.argumentsJson);
        case .functionCallOutput(let functionCallOutputInput):
            return .functionCallOutput(
                callId: functionCallOutputInput.callId,
                output: functionCallOutputInput.output);
        case .unsupported(let unsupportedInputItem):
            var inputItemType: String = "unknown";
            if case let .object(unsupportedObject) = unsupportedInputItem {
                if let typeWireValue: JsonWireValue = unsupportedObject.value(forKey: "type") {
                    if case let .string(typeText) = typeWireValue {
                        inputItemType = typeText;
                    }
                }
            }
            throw OpenAiResponsesValidationError.unsupportedInputItem(inputItemType: inputItemType);
        }
    }
}

extension OpenAiResponseMessageInput {

    internal func intoParts() throws -> OpenAiResponseInputItemParts {
        switch self.role {
        case .user:
            let (contentText, decodedImages): (String, Array<ImageInput.OpenAiImageInput>) =
                try self.content.intoUserContent();
            return .userMessage(content: contentText, images: decodedImages);
        case .system:
            return .systemMessage(content: try self.content.intoText());
        case .developer:
            return .developerMessage(content: try self.content.intoText());
        case .assistant:
            return .assistantMessage(content: try self.content.intoText());
        }
    }
}

extension OpenAiResponseMessageContent {

    /// Folds this content into plain text; image parts are only valid inside
    /// user messages, where `intoUserContent` decodes them instead.
    internal func intoText() throws -> String {
        switch self {
        case .text(let contentText):
            return contentText;
        case .parts(let contentParts):
            if contentParts.isEmpty {
                throw OpenAiResponsesValidationError.emptyContentParts;
            }
            var combinedText: String = "";
            for contentPart: OpenAiResponseContentPart in contentParts {
                switch contentPart {
                case .inputText(let text), .outputText(let text, _, _):
                    combinedText = combinedText + text;
                case .inputImage:
                    throw OpenAiResponsesValidationError.imageInputOutsideUserMessage;
                }
            }
            return combinedText;
        }
    }

    /// Folds user message content into concatenated text plus ordered images.
    internal func intoUserContent() throws -> (String, Array<ImageInput.OpenAiImageInput>) {
        switch self {
        case .text(let contentText):
            return (contentText, Array<ImageInput.OpenAiImageInput>());
        case .parts(let contentParts):
            if contentParts.isEmpty {
                throw OpenAiResponsesValidationError.emptyContentParts;
            }
            var combinedText: String = "";
            var decodedImages: Array<ImageInput.OpenAiImageInput> =
                Array<ImageInput.OpenAiImageInput>();
            for contentPart: OpenAiResponseContentPart in contentParts {
                switch contentPart {
                case .inputText(let text), .outputText(let text, _, _):
                    combinedText = combinedText + text;
                case .inputImage(let imageUrl, _):
                    do {
                        try ImageInput.validateImageUrlScheme(imageUrl: imageUrl);
                        let decodedImage: ImageInput.OpenAiImageInput =
                            try ImageInput.decodeImageUrl(imageUrl: imageUrl);
                        decodedImages.append(decodedImage);
                    } catch let imageError as OpenAiChatCompletionValidationError {
                        throw OpenAiResponsesValidationError.imageInput(imageError);
                    }
                }
            }
            return (combinedText, decodedImages);
        }
    }
}

extension OpenAiResponseReasoningInput {

    internal func intoParts() throws -> OpenAiResponseInputItemParts {
        if self.hasEncryptedContent {
            throw OpenAiResponsesValidationError.unsupportedReasoningReplay;
        }
        var combinedReasoningText: String = "";
        for reasoningSummary: OpenAiResponseReasoningSummaryInput in self.summaries {
            if case .summaryText(let summaryText) = reasoningSummary {
                combinedReasoningText = combinedReasoningText + summaryText;
            }
        }
        for reasoningContentEntry: OpenAiResponseReasoningContent in self.contents {
            combinedReasoningText = combinedReasoningText + reasoningContentEntry.text;
        }
        return .reasoning(content: combinedReasoningText);
    }
}
