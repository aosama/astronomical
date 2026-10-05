import Foundation;
import IpcProtocol;

/// A chat history message supported by the initial local text-only endpoint.
///
/// Unknown message fields are ignored: mainstream harnesses replay history produced by
/// other providers (for example OpenAI's `refusal` marker), and one third-party field
/// must not reject the whole conversation (#772). The `role` tag stays the strict
/// structural discriminator. Port of the message half of
/// crates/rest-contract/src/openai_chat_types.rs.
public enum OpenAiChatMessage: Equatable {
    /// An initial system instruction.
    case system(content: OpenAiMessageContent);
    /// A user message.
    case user(content: OpenAiMessageContent);
    /// A prior assistant response, possibly containing tool calls.
    case assistant(
        content: OpenAiMessageContent?, reasoningContent: String?,
        toolCalls: Array<OpenAiAssistantToolCall>, refusal: String?);
    /// A result returned by one prior assistant tool call.
    case tool(content: OpenAiMessageContent, toolCallId: String);

    /// Decodes the role-tagged wire object exactly as serde's internally
    /// tagged representation does; unknown fields inside a variant are
    /// absorbed because the Rust enum carries no deny_unknown_fields.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiChatMessage {
        let messageObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let roleTag: String = try messageObject.decodeTaggedVariantName(
            tagFieldName: "role", expectedVariantNames: ["system", "user", "assistant", "tool"]);
        switch roleTag {
        case "system":
            return .system(content: try OpenAiMessageContent.decoded(
                wireValue: try messageObject.requireObjectValue(fieldName: "content")));
        case "user":
            return .user(content: try OpenAiMessageContent.decoded(
                wireValue: try messageObject.requireObjectValue(fieldName: "content")));
        case "assistant":
            return .assistant(
                content: try messageObject.decodeOptionalRawValueAllowingAbsent(fieldName: "content")
                    .map({ (contentWireValue: JsonWireValue) throws -> OpenAiMessageContent in
                        return try OpenAiMessageContent.decoded(wireValue: contentWireValue);
                    }),
                reasoningContent: try messageObject.decodeOptionalStringAllowingAbsent(fieldName: "reasoning_content"),
                toolCalls: try messageObject.decodeArrayAllowingAbsent(
                    fieldName: "tool_calls",
                    mappedElement: { (toolCallWireValue: JsonWireValue) throws -> OpenAiAssistantToolCall in
                        return try OpenAiAssistantToolCall.decoded(wireValue: toolCallWireValue);
                    }),
                refusal: try messageObject.decodeOptionalStringAllowingAbsent(fieldName: "refusal"));
        default:
            return .tool(
                content: try OpenAiMessageContent.decoded(
                    wireValue: try messageObject.requireObjectValue(fieldName: "content")),
                toolCallId: try messageObject.decodeString(fieldName: "tool_call_id"));
        }
    }

    internal func validate() throws -> Void {
        switch self {
        case .system(let content), .user(let content):
            try content.validate();
        case .assistant(let content, let reasoningContent, let toolCalls, let refusal):
            if content == nil && reasoningContent == nil && toolCalls.isEmpty
                && (refusal == nil || refusal?.isEmpty == true) {
                throw OpenAiChatCompletionValidationError.emptyAssistantMessage;
            }
            if let unwrappedContent: OpenAiMessageContent = content {
                try unwrappedContent.validate();
            }
            for assistantToolCall: OpenAiAssistantToolCall in toolCalls {
                try assistantToolCall.validate();
            }
        case .tool(let content, let toolCallId):
            try ChatToolValidation.validateNonEmptyString(fieldName: "tool_call_id", stringValue: toolCallId);
            try content.validate();
        }
    }

    internal func intoParts() throws -> OpenAiChatMessageParts {
        try self.validate();
        switch self {
        case .system(let content):
            return .system(content: try content.intoText());
        case .user(let content):
            let (textContent, images): (String, Array<ImageInput.OpenAiImageInput>) = try content.intoUserContent();
            return .user(content: textContent, images: images);
        case .assistant(let content, let reasoningContent, let toolCalls, let refusal):
            // A replayed upstream refusal with no visible content becomes the message
            // content so the harness keeps the full conversation fidelity (#772).
            let foldedContent: String? = try content.map({ (unwrappedContent: OpenAiMessageContent) throws -> String in
                return try unwrappedContent.intoText();
            });
            var visibleRefusal: String? = nil;
            if let unwrappedRefusal: String = refusal, unwrappedRefusal.isEmpty == false {
                visibleRefusal = unwrappedRefusal;
            }
            return .assistant(
                content: foldedContent ?? visibleRefusal,
                reasoningContent: reasoningContent,
                toolCalls: toolCalls.map({ (assistantToolCall: OpenAiAssistantToolCall) -> OpenAiAssistantToolCallParts in
                    return assistantToolCall.intoParts();
                }));
        case .tool(let content, let toolCallId):
            return .tool(toolCallId: toolCallId, content: try content.intoText());
        }
    }
}

/// One validated, text-or-image chat message ready for protocol translation.
public enum OpenAiChatMessageParts: Equatable {
    /// An initial system instruction.
    case system(content: String);
    /// A user message, possibly carrying decoded image inputs.
    case user(content: String, images: Array<ImageInput.OpenAiImageInput>);
    /// A prior assistant answer, reasoning, and function calls.
    case assistant(
        content: String?, reasoningContent: String?, toolCalls: Array<OpenAiAssistantToolCallParts>);
    /// A result produced by an earlier function call.
    case tool(toolCallId: String, content: String);
}

/// Text-only message content accepted by the initial endpoint. Untagged on
/// the wire: a bare string or an array of typed content parts.
public enum OpenAiMessageContent: Equatable {
    /// A compact string message.
    case text(String);
    /// A list of explicitly typed content parts.
    case parts(Array<OpenAiContentPart>);

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiMessageContent {
        if case let .string(textValue) = wireValue {
            return .text(textValue);
        }
        if case let .array(partValues) = wireValue {
            return .parts(try partValues.map({ (partWireValue: JsonWireValue) throws -> OpenAiContentPart in
                return try OpenAiContentPart.decoded(wireValue: partWireValue);
            }));
        }
        throw JsonWireProblem.malformedDocument(
            problem: "data did not match any variant of untagged enum OpenAiMessageContent");
    }

    internal func validate() throws -> Void {
        switch self {
        case .text:
            return;
        case .parts(let contentParts):
            if contentParts.isEmpty {
                throw OpenAiChatCompletionValidationError.emptyContentParts;
            }
            for contentPart: OpenAiContentPart in contentParts {
                try contentPart.validate();
            }
        }
    }

    internal func intoText() throws -> String {
        try self.validate();
        switch self {
        case .text(let content):
            return content;
        case .parts(let contentParts):
            return contentParts.reduce(into: "") { (combinedContent: inout String, contentPart: OpenAiContentPart) in
                switch contentPart {
                case .text(let text):
                    combinedContent = combinedContent + text;
                // A replayed refusal fragment is assistant text for history fidelity.
                case .refusal(let refusal):
                    combinedContent = combinedContent + refusal;
                case .imageUrl, .inputAudio, .videoUrl:
                    break;
                }
            };
        }
    }

    /// Decodes user message content into concatenated text and ordered image inputs.
    internal func intoUserContent() throws -> (String, Array<ImageInput.OpenAiImageInput>) {
        try self.validate();
        switch self {
        case .text(let content):
            return (content, Array<ImageInput.OpenAiImageInput>());
        case .parts(let contentParts):
            var combinedText: String = "";
            var decodedImages: Array<ImageInput.OpenAiImageInput> = Array();
            for contentPart: OpenAiContentPart in contentParts {
                switch contentPart {
                case .text(let text):
                    combinedText = combinedText + text;
                case .refusal(let refusal):
                    combinedText = combinedText + refusal;
                case .imageUrl(let imageUrl):
                    decodedImages.append(try ImageInput.decodeImageUrl(imageUrl: imageUrl.url));
                case .inputAudio, .videoUrl:
                    // Already rejected by validate(); unreachable here.
                    break;
                }
            }
            return (combinedText, decodedImages);
        }
    }
}

/// A typed content part. Text, refusal, and data-URI images are supported by the
/// local endpoint; unknown part types stay a hard error because the part type is a
/// structural discriminator, not a provider extension field.
public enum OpenAiContentPart: Equatable {
    /// A text fragment.
    case text(text: String);
    /// An image input encoded as a `data:image/...;base64,...` URI.
    case imageUrl(imageUrl: OpenAiImageUrl);
    /// An OpenAI safety-refusal fragment; folded into the message text on replay.
    case refusal(refusal: String);
    /// An audio input, intentionally unsupported by the initial endpoint.
    case inputAudio(inputAudio: JsonWireValue);
    /// A video input, intentionally unsupported by the initial endpoint.
    case videoUrl(videoUrl: JsonWireValue);

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiContentPart {
        let partObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let partType: String = try partObject.decodeTaggedVariantName(
            tagFieldName: "type",
            expectedVariantNames: ["text", "image_url", "refusal", "input_audio", "video_url"]);
        switch partType {
        case "text":
            return .text(text: try partObject.decodeString(fieldName: "text"));
        case "image_url":
            return .imageUrl(imageUrl: try OpenAiImageUrl.decoded(
                wireValue: try partObject.requireObjectValue(fieldName: "image_url")));
        case "refusal":
            return .refusal(refusal: try partObject.decodeString(fieldName: "refusal"));
        case "input_audio":
            return .inputAudio(inputAudio: try partObject.requireObjectValue(fieldName: "input_audio"));
        default:
            return .videoUrl(videoUrl: try partObject.requireObjectValue(fieldName: "video_url"));
        }
    }

    internal func validate() throws -> Void {
        switch self {
        case .text, .refusal:
            return;
        case .imageUrl(let imageUrl):
            try ImageInput.validateImageUrlScheme(imageUrl: imageUrl.url);
        case .inputAudio:
            throw OpenAiChatCompletionValidationError.unsupportedContentPart(contentPartType: "input_audio");
        case .videoUrl:
            throw OpenAiChatCompletionValidationError.unsupportedContentPart(contentPartType: "video_url");
        }
    }
}

/// The `image_url` object inside an `image_url` content part.
public struct OpenAiImageUrl: Equatable {
    /// The image URL. Only `data:image/...;base64,...` URIs are accepted.
    private let urlText: String;

    fileprivate init(url: String) {
        self.urlText = url;
    }

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiImageUrl {
        let urlObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        return OpenAiImageUrl(url: try urlObject.decodeString(fieldName: "url"));
    }

    /// The image URL. Only `data:image/...;base64,...` URIs are accepted.
    internal var url: String {
        return self.urlText;
    }
}
