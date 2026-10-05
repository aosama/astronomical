import Foundation;

/// A request rejected before worker admission by the public OpenAI contract.
/// Port of the OpenAiChatCompletionValidationError enum from
/// crates/rest-contract/src/openai_chat_completion_request.rs, kept in its
/// own owner file so every wire module can reject without pulling the full
/// request unit in.
public enum OpenAiChatCompletionValidationError: Error, Equatable {
    /// A required string was empty.
    case emptyString(fieldName: String);
    /// The request did not contain any messages.
    case emptyMessages;
    /// A multipart content list was empty.
    case emptyContentParts;
    /// The text-only endpoint received another modality.
    case unsupportedContentPart(contentPartType: String);
    /// An image URL used a scheme other than `data:image/...;base64,...`.
    case unsupportedImageUrlScheme;
    /// An image URL had a non-image MIME type.
    case unsupportedImageMimeType(actualMimeType: String);
    /// A data URI was malformed (missing comma or metadata).
    case malformedDataUri;
    /// A data URI base64 payload could not be decoded.
    case invalidBase64;
    /// A decoded image exceeded the maximum accepted byte size.
    case imageTooLarge(actualBytes: Int, maximumBytes: Int);
    /// An assistant history record contained no answer, reasoning, or tool call.
    case emptyAssistantMessage;
    /// A tool used an unsupported type.
    case unsupportedToolType;
    /// A tool name did not match the strict portable function-name grammar.
    case invalidToolName(toolName: String);
    /// A tool schema was too deeply nested.
    case toolSchemaNestingTooDeep(actualSchemaNestingDepth: Int, maximumSchemaNestingDepth: Int);
    /// A locally decoded schema unexpectedly could not serialize for its byte check.
    case toolSchemaSerializationFailed;
    /// The client selected an unsupported automatic tool mode.
    case unsupportedToolChoice(mode: String);
    /// A forced function was not among the declared tools.
    case toolChoiceNamesUnknownFunction(functionName: String);
    /// A declared function cannot yet be deterministically forced.
    case unsupportedForcedToolChoice(functionName: String);
    /// `max_tokens` and `max_completion_tokens` disagreed.
    case conflictingOutputTokenLimits(maxTokens: UInt32, maxCompletionTokens: UInt32);
    /// A reasoning-control spelling was contradictory or unrecognized.
    case thinkingControls(ThinkingControlsError);
    /// The output token budget was zero or too large for the worker representation.
    case outputTokenCountOutOfRange(actualOutputTokens: UInt32, maximumOutputTokens: UInt32);
    /// A sampling setting fell outside its OpenAI-compatible range.
    case samplingParameterOutOfRange(parameterName: String, minimum: String, maximum: String);
    /// Caller-defined stop sequences are not implemented by the initial endpoint.
    case unsupportedStopSequences;
    /// A recognized OpenAI-compatible request option is not implemented yet.
    case unsupportedOption(optionName: String);
    /// `response_format` failed public structured-output validation.
    case structuredOutput(OpenAiStructuredOutputValidationError);
    /// The extra-body structured-outputs surface failed validation.
    case structuredOutputs(OpenAiStructuredOutputsValidationError);

    public var errorDescription: String? {
        switch self {
        case .emptyString(let fieldName):
            return "\(fieldName) must not be empty";
        case .emptyMessages:
            return "messages must not be empty";
        case .emptyContentParts:
            return "content parts must not be empty";
        case .unsupportedContentPart(let contentPartType):
            return "content part type '\(contentPartType)' is not supported by the text-only endpoint";
        case .unsupportedImageUrlScheme:
            return "only data:image base64 URIs are supported for image input";
        case .unsupportedImageMimeType(let actualMimeType):
            return "image MIME type '\(actualMimeType)' is not an image type";
        case .malformedDataUri:
            return "the data URI is malformed";
        case .invalidBase64:
            return "the data URI base64 payload is invalid";
        case .imageTooLarge(let actualBytes, let maximumBytes):
            return "decoded image is \(actualBytes) bytes, exceeding the \(maximumBytes) byte limit";
        case .emptyAssistantMessage:
            return "assistant messages must contain content, reasoning, or tool calls";
        case .unsupportedToolType:
            return "only function tools are supported";
        case .invalidToolName(let toolName):
            return "tool name '\(toolName)' is invalid";
        case .toolSchemaNestingTooDeep(let actualSchemaNestingDepth, let maximumSchemaNestingDepth):
            return "tool schema nesting depth is \(actualSchemaNestingDepth), exceeding \(maximumSchemaNestingDepth)";
        case .toolSchemaSerializationFailed:
            return "tool schema could not be serialized for bounded validation";
        case .unsupportedToolChoice(let mode):
            return "tool choice mode '\(mode)' is unsupported";
        case .toolChoiceNamesUnknownFunction(let functionName):
            return "tool choice names undeclared function '\(functionName)'";
        case .unsupportedForcedToolChoice(let functionName):
            return "forcing function '\(functionName)' is unsupported";
        case .conflictingOutputTokenLimits(let maxTokens, let maxCompletionTokens):
            return "max_tokens (\(maxTokens)) conflicts with max_completion_tokens (\(maxCompletionTokens))";
        case .thinkingControls(let thinkingControlsError):
            return thinkingControlsError.errorDescription;
        case .outputTokenCountOutOfRange(let actualOutputTokens, let maximumOutputTokens):
            return "output token count is \(actualOutputTokens), outside the 1..=\(maximumOutputTokens) token range";
        case .samplingParameterOutOfRange(let parameterName, let minimum, let maximum):
            return "\(parameterName) is outside the supported range \(minimum)..=\(maximum)";
        case .unsupportedStopSequences:
            return "caller-supplied stop sequences are unsupported";
        case .unsupportedOption(let optionName):
            return "request option '\(optionName)' is unsupported";
        case .structuredOutput(let structuredOutputError):
            return structuredOutputError.errorDescription;
        case .structuredOutputs(let structuredOutputsError):
            return structuredOutputsError.errorDescription;
        }
    }
}
