import Foundation;
import IpcProtocol;

/// Port of crates/rest-contract/src/openai_models_and_errors.rs.

/// Validated input used to construct one OpenAI-compatible model description.
public struct OpenAiModelParts: Equatable {
    /// Model identifier advertised by the local server.
    public let modelId: String;
    /// Unix timestamp associated with this server's model advertisement.
    public let created: UInt64;
    /// Local owner label for the advertised model.
    public let ownedBy: String;
    /// Total prompt plus generation position capacity of the model.
    public let contextWindow: UInt32;
    /// Maximum prompt tokens a client may send when reserving the minimum
    /// generation position. Request admission applies the selected output budget.
    public let maxInputTokens: UInt32;
    /// Maximum output tokens a client may request when the prompt leaves enough
    /// positions. This is independent from the maximum prompt length.
    public let maxOutputTokens: UInt32;
    /// Modalities accepted in request input.
    public let inputModalities: Array<String>;
    /// Modalities emitted in response output.
    public let outputModalities: Array<String>;
    /// Whether the advertised generation endpoints support streaming.
    public let supportsStreaming: Bool;
    /// Whether the model emits reasoning separately from visible assistant text.
    public let supportsReasoning: Bool;
    /// Public wire format used when reasoning is supported.
    public let reasoningFormat: String?;
    /// Whether the model emits validated function calls.
    public let supportsToolCalls: Bool;
    /// Public wire format used when function calls are supported.
    public let toolCallFormat: String?;
    /// Generation endpoints supported by this model.
    public let supportedEndpoints: Array<String>;
    /// Whether Chat Completions and Responses accept `response_format`.
    public let supportsStructuredOutputs: Bool;
    /// How structured output is enforced, when supported.
    public let structuredOutputEnforcement: String?;

    public init(
        modelId: String, created: UInt64, ownedBy: String, contextWindow: UInt32,
        maxInputTokens: UInt32, maxOutputTokens: UInt32, inputModalities: Array<String>,
        outputModalities: Array<String>, supportsStreaming: Bool, supportsReasoning: Bool,
        reasoningFormat: String?, supportsToolCalls: Bool, toolCallFormat: String?,
        supportedEndpoints: Array<String>, supportsStructuredOutputs: Bool,
        structuredOutputEnforcement: String?) {
        self.modelId = modelId;
        self.created = created;
        self.ownedBy = ownedBy;
        self.contextWindow = contextWindow;
        self.maxInputTokens = maxInputTokens;
        self.maxOutputTokens = maxOutputTokens;
        self.inputModalities = inputModalities;
        self.outputModalities = outputModalities;
        self.supportsStreaming = supportsStreaming;
        self.supportsReasoning = supportsReasoning;
        self.reasoningFormat = reasoningFormat;
        self.supportsToolCalls = supportsToolCalls;
        self.toolCallFormat = toolCallFormat;
        self.supportedEndpoints = supportedEndpoints;
        self.supportsStructuredOutputs = supportsStructuredOutputs;
        self.structuredOutputEnforcement = structuredOutputEnforcement;
    }

    /// Validates relationships that must hold between advertised model capabilities.
    public func validate() throws -> Void {
        if self.contextWindow == 0 {
            throw OpenAiModelValidationError.contextWindowMustBePositive;
        }
        if self.maxInputTokens >= self.contextWindow {
            throw OpenAiModelValidationError.inputTokenBudgetMustLeaveGenerationPosition(
                maxInputTokens: self.maxInputTokens, contextWindow: self.contextWindow);
        }
        if self.maxOutputTokens >= self.contextWindow {
            throw OpenAiModelValidationError.outputTokenBudgetMustLeavePromptPosition(
                maxOutputTokens: self.maxOutputTokens, contextWindow: self.contextWindow);
        }
        if self.inputModalities.contains("text") == false {
            throw OpenAiModelValidationError.inputModalitiesMustContainText;
        }
        if self.outputModalities.contains("text") == false {
            throw OpenAiModelValidationError.outputModalitiesMustContainText;
        }
        if self.supportsReasoning != (self.reasoningFormat != nil) {
            throw OpenAiModelValidationError.reasoningFormatMustMatchSupport;
        }
        if self.supportsToolCalls != (self.toolCallFormat != nil) {
            throw OpenAiModelValidationError.toolCallFormatMustMatchSupport;
        }
        if self.supportsStructuredOutputs != (self.structuredOutputEnforcement != nil) {
            throw OpenAiModelValidationError.structuredOutputEnforcementMustMatchSupport;
        }
        if self.supportedEndpoints.isEmpty {
            throw OpenAiModelValidationError.supportedEndpointsMustNotBeEmpty;
        }
    }
}

/// Capability metadata for an image-output model without autoregressive token limits.
public struct OpenAiImageModelParts: Equatable {
    /// Model identifier advertised by the local server.
    public let modelId: String;
    /// Unix timestamp associated with this server's model advertisement.
    public let created: UInt64;
    /// Local owner label for the advertised model.
    public let ownedBy: String;
    /// Modalities accepted in request input.
    public let inputModalities: Array<String>;
    /// Modalities emitted in response output.
    public let outputModalities: Array<String>;
    /// Generation endpoints supported by this model.
    public let supportedEndpoints: Array<String>;

    public init(
        modelId: String, created: UInt64, ownedBy: String, inputModalities: Array<String>,
        outputModalities: Array<String>, supportedEndpoints: Array<String>) {
        self.modelId = modelId;
        self.created = created;
        self.ownedBy = ownedBy;
        self.inputModalities = inputModalities;
        self.outputModalities = outputModalities;
        self.supportedEndpoints = supportedEndpoints;
    }
}

/// Capability metadata for an embedding model without autoregressive token limits.
public struct OpenAiEmbeddingModelParts: Equatable {
    /// Model identifier advertised by the local server.
    public let modelId: String;
    /// Unix timestamp associated with this server's model advertisement.
    public let created: UInt64;
    /// Local owner label for the advertised model.
    public let ownedBy: String;
    /// Native vector width returned by the loaded embedding artifact.
    public let vectorWidth: UInt32;
    /// Maximum accepted prompt tokens per embedding input.
    public let maxInputTokens: UInt32;

    public init(
        modelId: String, created: UInt64, ownedBy: String, vectorWidth: UInt32,
        maxInputTokens: UInt32) {
        self.modelId = modelId;
        self.created = created;
        self.ownedBy = ownedBy;
        self.vectorWidth = vectorWidth;
        self.maxInputTokens = maxInputTokens;
    }
}

/// Rejection reason for internally inconsistent advertised model capabilities.
public enum OpenAiModelValidationError: Error, Equatable {
    /// A model must have at least one context position.
    case contextWindowMustBePositive;
    /// The advertised prompt budget must leave one position for generation.
    case inputTokenBudgetMustLeaveGenerationPosition(maxInputTokens: UInt32, contextWindow: UInt32);
    /// The advertised output budget must leave one position for prompt input.
    case outputTokenBudgetMustLeavePromptPosition(maxOutputTokens: UInt32, contextWindow: UInt32);
    /// Input must always support text for the current API contract.
    case inputModalitiesMustContainText;
    /// Output must always support text for the current API contract.
    case outputModalitiesMustContainText;
    /// Image generation must advertise its image result modality.
    case imageOutputModalitiesMustContainImage;
    /// A reasoning format must be present exactly when reasoning is supported.
    case reasoningFormatMustMatchSupport;
    /// A tool-call format must be present exactly when tool calling is supported.
    case toolCallFormatMustMatchSupport;
    /// At least one generation endpoint must be advertised.
    case supportedEndpointsMustNotBeEmpty;
    /// A loaded embedding artifact must report a positive vector width.
    case embeddingVectorWidthMustBePositive;
    /// A loaded embedding artifact must report a positive input budget.
    case embeddingInputBudgetMustBePositive;
    /// Structured-output enforcement must be present exactly when structured outputs are supported.
    case structuredOutputEnforcementMustMatchSupport;

    public var errorDescription: String? {
        switch self {
        case .contextWindowMustBePositive:
            return "context window must be positive";
        case .inputTokenBudgetMustLeaveGenerationPosition(let maxInputTokens, let contextWindow):
            return "maximum input tokens \(maxInputTokens) must be smaller than context window \(contextWindow)";
        case .outputTokenBudgetMustLeavePromptPosition(let maxOutputTokens, let contextWindow):
            return "maximum output tokens \(maxOutputTokens) must be smaller than context window \(contextWindow)";
        case .inputModalitiesMustContainText:
            return "input modalities must contain text";
        case .outputModalitiesMustContainText:
            return "output modalities must contain text";
        case .imageOutputModalitiesMustContainImage:
            return "image model output modalities must contain image";
        case .reasoningFormatMustMatchSupport:
            return "reasoning format must be present exactly when reasoning is supported";
        case .toolCallFormatMustMatchSupport:
            return "tool-call format must be present exactly when tool calling is supported";
        case .supportedEndpointsMustNotBeEmpty:
            return "supported endpoints must not be empty";
        case .embeddingVectorWidthMustBePositive:
            return "embedding vector width must be positive";
        case .embeddingInputBudgetMustBePositive:
            return "embedding input token budget must be positive";
        case .structuredOutputEnforcementMustMatchSupport:
            return "structured-output enforcement must be present exactly when structured outputs are supported";
        }
    }
}

/// One model visible through the standard OpenAI models endpoint.
public struct OpenAiModel: Equatable {
    private let idText: String;
    private let objectKind: String;
    private let createdTimestamp: UInt64;
    private let ownedByText: String;
    private let contextWindowTokens: UInt32?;
    private let maxInputTokensBudget: UInt32?;
    private let maxOutputTokensBudget: UInt32?;
    private let inputModalityList: Array<String>?;
    private let outputModalityList: Array<String>?;
    private let supportsStreamingFlag: Bool?;
    private let supportsReasoningFlag: Bool?;
    private let reasoningFormatName: String?;
    private let supportsToolCallsFlag: Bool?;
    private let toolCallFormatName: String?;
    private let supportedEndpointList: Array<String>?;
    private let supportsStructuredOutputsFlag: Bool?;
    private let structuredOutputEnforcementMode: String?;

    private init(
        id: String, objectKind: String, created: UInt64, ownedBy: String,
        contextWindow: UInt32?, maxInputTokens: UInt32?, maxOutputTokens: UInt32?,
        inputModalities: Array<String>?, outputModalities: Array<String>?,
        supportsStreaming: Bool?, supportsReasoning: Bool?, reasoningFormat: String?,
        supportsToolCalls: Bool?, toolCallFormat: String?, supportedEndpoints: Array<String>?,
        supportsStructuredOutputs: Bool?, structuredOutputEnforcement: String?) {
        self.idText = id;
        self.objectKind = objectKind;
        self.createdTimestamp = created;
        self.ownedByText = ownedBy;
        self.contextWindowTokens = contextWindow;
        self.maxInputTokensBudget = maxInputTokens;
        self.maxOutputTokensBudget = maxOutputTokens;
        self.inputModalityList = inputModalities;
        self.outputModalityList = outputModalities;
        self.supportsStreamingFlag = supportsStreaming;
        self.supportsReasoningFlag = supportsReasoning;
        self.reasoningFormatName = reasoningFormat;
        self.supportsToolCallsFlag = supportsToolCalls;
        self.toolCallFormatName = toolCallFormat;
        self.supportedEndpointList = supportedEndpoints;
        self.supportsStructuredOutputsFlag = supportsStructuredOutputs;
        self.structuredOutputEnforcementMode = structuredOutputEnforcement;
    }

    /// Returns the exact model identifier advertised through the REST API.
    public func id() -> String {
        return self.idText;
    }

    /// Builds one model entry from validated advertised model capabilities.
    public static func fromParts(modelParts: OpenAiModelParts) throws -> OpenAiModel {
        try modelParts.validate();
        return OpenAiModel(
            id: modelParts.modelId,
            objectKind: "model",
            created: modelParts.created,
            ownedBy: modelParts.ownedBy,
            contextWindow: modelParts.contextWindow,
            maxInputTokens: modelParts.maxInputTokens,
            maxOutputTokens: modelParts.maxOutputTokens,
            inputModalities: modelParts.inputModalities,
            outputModalities: modelParts.outputModalities,
            supportsStreaming: modelParts.supportsStreaming,
            supportsReasoning: modelParts.supportsReasoning,
            reasoningFormat: modelParts.reasoningFormat,
            supportsToolCalls: modelParts.supportsToolCalls,
            toolCallFormat: modelParts.toolCallFormat,
            supportedEndpoints: modelParts.supportedEndpoints,
            supportsStructuredOutputs: modelParts.supportsStructuredOutputs,
            structuredOutputEnforcement: modelParts.structuredOutputEnforcement);
    }

    /// Builds one non-streaming image model without fabricating token-generation limits.
    public static func fromImageParts(imageModelParts: OpenAiImageModelParts) throws -> OpenAiModel {
        if imageModelParts.inputModalities.contains("text") == false {
            throw OpenAiModelValidationError.inputModalitiesMustContainText;
        }
        if imageModelParts.outputModalities.contains("image") == false {
            throw OpenAiModelValidationError.imageOutputModalitiesMustContainImage;
        }
        if imageModelParts.supportedEndpoints.isEmpty {
            throw OpenAiModelValidationError.supportedEndpointsMustNotBeEmpty;
        }
        return OpenAiModel(
            id: imageModelParts.modelId,
            objectKind: "model",
            created: imageModelParts.created,
            ownedBy: imageModelParts.ownedBy,
            contextWindow: nil,
            maxInputTokens: nil,
            maxOutputTokens: nil,
            inputModalities: imageModelParts.inputModalities,
            outputModalities: imageModelParts.outputModalities,
            supportsStreaming: false,
            supportsReasoning: false,
            reasoningFormat: nil,
            supportsToolCalls: false,
            toolCallFormat: nil,
            supportedEndpoints: imageModelParts.supportedEndpoints,
            supportsStructuredOutputs: false,
            structuredOutputEnforcement: nil);
    }

    /// Builds one non-streaming embedding model without fabricating token limits.
    public static func fromEmbeddingParts(embeddingModelParts: OpenAiEmbeddingModelParts) throws -> OpenAiModel {
        if embeddingModelParts.vectorWidth == 0 {
            throw OpenAiModelValidationError.embeddingVectorWidthMustBePositive;
        }
        if embeddingModelParts.maxInputTokens == 0 {
            throw OpenAiModelValidationError.embeddingInputBudgetMustBePositive;
        }
        return OpenAiModel(
            id: embeddingModelParts.modelId,
            objectKind: "model",
            created: embeddingModelParts.created,
            ownedBy: embeddingModelParts.ownedBy,
            contextWindow: nil,
            maxInputTokens: nil,
            maxOutputTokens: nil,
            inputModalities: ["text"],
            outputModalities: ["embedding"],
            supportsStreaming: false,
            supportsReasoning: false,
            reasoningFormat: nil,
            supportsToolCalls: false,
            toolCallFormat: nil,
            supportedEndpoints: ["/v1/embeddings"],
            supportsStructuredOutputs: false,
            structuredOutputEnforcement: nil);
    }

    /// Serializes the model entry exactly as the serde derive does: fields in
    /// declaration order, absent Optional fields skipped.
    public func wireValue() -> JsonWireValue {
        var modelObject: JsonWireObject = JsonWireObject(entries: Array());
        modelObject.appendEntry(key: "id", value: .string(self.idText));
        modelObject.appendEntry(key: "object", value: .string(self.objectKind));
        modelObject.appendEntry(key: "created", value: .unsignedInteger(self.createdTimestamp));
        modelObject.appendEntry(key: "owned_by", value: .string(self.ownedByText));
        if let contextWindow: UInt32 = self.contextWindowTokens {
            modelObject.appendEntry(key: "context_window", value: .unsignedInteger(UInt64(contextWindow)));
        }
        if let maxInputTokens: UInt32 = self.maxInputTokensBudget {
            modelObject.appendEntry(key: "max_input_tokens", value: .unsignedInteger(UInt64(maxInputTokens)));
        }
        if let maxOutputTokens: UInt32 = self.maxOutputTokensBudget {
            modelObject.appendEntry(key: "max_output_tokens", value: .unsignedInteger(UInt64(maxOutputTokens)));
        }
        if let inputModalities: Array<String> = self.inputModalityList {
            modelObject.appendEntry(key: "input_modalities", value: JsonWireValue.stringArray(inputModalities));
        }
        if let outputModalities: Array<String> = self.outputModalityList {
            modelObject.appendEntry(key: "output_modalities", value: JsonWireValue.stringArray(outputModalities));
        }
        if let supportsStreaming: Bool = self.supportsStreamingFlag {
            modelObject.appendEntry(key: "supports_streaming", value: .boolean(supportsStreaming));
        }
        if let supportsReasoning: Bool = self.supportsReasoningFlag {
            modelObject.appendEntry(key: "supports_reasoning", value: .boolean(supportsReasoning));
        }
        if let reasoningFormat: String = self.reasoningFormatName {
            modelObject.appendEntry(key: "reasoning_format", value: .string(reasoningFormat));
        }
        if let supportsToolCalls: Bool = self.supportsToolCallsFlag {
            modelObject.appendEntry(key: "supports_tool_calls", value: .boolean(supportsToolCalls));
        }
        if let toolCallFormat: String = self.toolCallFormatName {
            modelObject.appendEntry(key: "tool_call_format", value: .string(toolCallFormat));
        }
        if let supportedEndpoints: Array<String> = self.supportedEndpointList {
            modelObject.appendEntry(key: "supported_endpoints", value: JsonWireValue.stringArray(supportedEndpoints));
        }
        if let supportsStructuredOutputs: Bool = self.supportsStructuredOutputsFlag {
            modelObject.appendEntry(key: "supports_structured_outputs", value: .boolean(supportsStructuredOutputs));
        }
        if let structuredOutputEnforcement: String = self.structuredOutputEnforcementMode {
            modelObject.appendEntry(key: "structured_output_enforcement", value: .string(structuredOutputEnforcement));
        }
        return .object(modelObject);
    }
}

/// A standard OpenAI-compatible list of the exact models ready in the local worker.
public struct OpenAiModelList: Equatable {
    private let objectKind: String;
    private let dataModels: Array<OpenAiModel>;

    private init(objectKind: String, data: Array<OpenAiModel>) {
        self.objectKind = objectKind;
        self.dataModels = data;
    }

    /// Builds an empty response while no local worker can safely advertise a model.
    public static func empty() -> OpenAiModelList {
        return OpenAiModelList(objectKind: "list", data: Array());
    }

    /// Builds a one-model response from validated model capabilities.
    public static func singleModel(modelParts: OpenAiModelParts) throws -> OpenAiModelList {
        return OpenAiModelList(objectKind: "list", data: [try OpenAiModel.fromParts(modelParts: modelParts)]);
    }

    /// Builds a multi-model response listing all discovered models.
    public static func fromModels(models: Array<OpenAiModel>) -> OpenAiModelList {
        return OpenAiModelList(objectKind: "list", data: models);
    }

    public func wireValue() -> JsonWireValue {
        var listObject: JsonWireObject = JsonWireObject(entries: Array());
        listObject.appendEntry(key: "object", value: .string(self.objectKind));
        listObject.appendEntry(
            key: "data",
            value: JsonWireValue.mappedArray(self.dataModels, mappedWireValue: { (model: OpenAiModel) -> JsonWireValue in
                return model.wireValue();
            }));
        return .object(listObject);
    }
}

/// A standard OpenAI-compatible error response.
public struct OpenAiErrorResponse: Equatable, Sendable {
    private let errorContent: OpenAiError;

    private init(error: OpenAiError) {
        self.errorContent = error;
    }

    /// Builds an invalid-request response with optional public field and stable code context.
    public static func invalidRequest(message: String, parameter: String?, code: String?) -> OpenAiErrorResponse {
        return OpenAiErrorResponse(error: OpenAiError(
            message: message, errorType: "invalid_request_error", param: parameter, code: code));
    }

    /// Builds a service-unavailable response when no safe worker can serve the request.
    public static func serviceUnavailable(message: String, code: String?) -> OpenAiErrorResponse {
        return OpenAiErrorResponse(error: OpenAiError(
            message: message, errorType: "server_error", param: nil, code: code));
    }

    /// Builds a service-unavailable response for a rejected model load.
    public static func modelLoadFailed(modelLoadFailureReason: String) -> OpenAiErrorResponse {
        return OpenAiErrorResponse.serviceUnavailable(
            message: "the requested model could not be loaded: \(modelLoadFailureReason)",
            code: "model_load_failed");
    }

    /// Builds a capacity response when the one-worker scheduler cannot admit another request.
    public static func capacityUnavailable(message: String) -> OpenAiErrorResponse {
        return OpenAiErrorResponse(error: OpenAiError(
            message: message, errorType: "server_error", param: nil, code: "server_capacity"));
    }

    public func wireValue() -> JsonWireValue {
        var responseObject: JsonWireObject = JsonWireObject(entries: Array());
        responseObject.appendEntry(key: "error", value: self.errorContent.wireValue());
        return .object(responseObject);
    }
}

/// Error content nested inside an OpenAI-compatible error response.
public struct OpenAiError: Equatable, Sendable {
    private let messageText: String;
    private let errorTypeName: String;
    private let parameterName: String?;
    private let codeName: String?;

    fileprivate init(message: String, errorType: String, param: String?, code: String?) {
        self.messageText = message;
        self.errorTypeName = errorType;
        self.parameterName = param;
        self.codeName = code;
    }

    public func wireValue() -> JsonWireValue {
        var errorObject: JsonWireObject = JsonWireObject(entries: Array());
        errorObject.appendEntry(key: "message", value: .string(self.messageText));
        errorObject.appendEntry(key: "type", value: .string(self.errorTypeName));
        if let parameterName: String = self.parameterName {
            errorObject.appendEntry(key: "param", value: .string(parameterName));
        }
        if let codeName: String = self.codeName {
            errorObject.appendEntry(key: "code", value: .string(codeName));
        }
        return .object(errorObject);
    }
}
