// ResponsesResponse.swift — RestContract
//
// Port of crates/rest-contract/src/openai_responses_response.rs (response
// object, request echo configuration, status, and usage; the output item
// shapes live in ResponsesResponse+OutputItems.swift).

import Foundation;
import IpcProtocol;

/// One ordered metadata entry. The Rust contract stores metadata in a
/// `BTreeMap<String, String>` whose iteration is byte-sorted by key; this
/// ordered entry array reproduces that determinism for validation and
/// serialization.
public struct OpenAiMetadataEntry: Equatable {
    public let metadataKey: String;
    public let metadataValue: String;

    public init(metadataKey: String, metadataValue: String) {
        self.metadataKey = metadataKey;
        self.metadataValue = metadataValue;
    }
}

/// One complete response returned by the local Responses endpoint.
public struct OpenAiResponse: Equatable {
    private var responseIdentifier: String;
    private var objectKind: String;
    private var createdTimestamp: UInt64;
    private var completedTimestamp: UInt64?;
    private var lifecycleStatus: OpenAiResponseStatus;
    private var responseError: OpenAiResponseError?;
    private var incompleteDetails: OpenAiResponseIncompleteDetails?;
    private var instructionsText: String?;
    private var metadataEntries: Array<OpenAiMetadataEntry>;
    private var modelIdentifier: String;
    private var outputItems: Array<OpenAiResponseOutputItem>;
    private var outputText: String;
    private var parallelToolCallsEnabled: Bool;
    private var temperatureValue: Float?;
    private var toolChoiceName: String;
    private var declaredTools: Array<OpenAiResponseFunctionTool>;
    private var topPValue: Float?;
    private var maxOutputTokensBudget: UInt32?;
    private var previousResponseIdentifier: String?;
    private var truncationMode: String;
    private var usageCounts: OpenAiResponseUsage?;

    fileprivate init(
        responseIdentifier: String, objectKind: String, createdTimestamp: UInt64,
        completedTimestamp: UInt64?, lifecycleStatus: OpenAiResponseStatus,
        responseError: OpenAiResponseError?, incompleteDetails: OpenAiResponseIncompleteDetails?,
        instructionsText: String?, metadataEntries: Array<OpenAiMetadataEntry>,
        modelIdentifier: String, outputItems: Array<OpenAiResponseOutputItem>,
        outputText: String, parallelToolCallsEnabled: Bool, temperatureValue: Float?,
        toolChoiceName: String, declaredTools: Array<OpenAiResponseFunctionTool>,
        topPValue: Float?, maxOutputTokensBudget: UInt32?,
        previousResponseIdentifier: String?, truncationMode: String,
        usageCounts: OpenAiResponseUsage?) {
        self.responseIdentifier = responseIdentifier;
        self.objectKind = objectKind;
        self.createdTimestamp = createdTimestamp;
        self.completedTimestamp = completedTimestamp;
        self.lifecycleStatus = lifecycleStatus;
        self.responseError = responseError;
        self.incompleteDetails = incompleteDetails;
        self.instructionsText = instructionsText;
        self.metadataEntries = metadataEntries;
        self.modelIdentifier = modelIdentifier;
        self.outputItems = outputItems;
        self.outputText = outputText;
        self.parallelToolCallsEnabled = parallelToolCallsEnabled;
        self.temperatureValue = temperatureValue;
        self.toolChoiceName = toolChoiceName;
        self.declaredTools = declaredTools;
        self.topPValue = topPValue;
        self.maxOutputTokensBudget = maxOutputTokensBudget;
        self.previousResponseIdentifier = previousResponseIdentifier;
        self.truncationMode = truncationMode;
        self.usageCounts = usageCounts;
    }

    /// Applies the validated request settings represented by this response.
    public func withRequestConfiguration(
        requestConfiguration: OpenAiResponseRequestConfiguration) -> OpenAiResponse {
        var configuredResponse: OpenAiResponse = self;
        configuredResponse.metadataEntries = requestConfiguration.metadata;
        configuredResponse.temperatureValue = requestConfiguration.temperature;
        configuredResponse.topPValue = requestConfiguration.topP;
        configuredResponse.maxOutputTokensBudget = requestConfiguration.maxOutputTokens;
        configuredResponse.toolChoiceName = requestConfiguration.toolChoice;
        configuredResponse.declaredTools = requestConfiguration.tools;
        return configuredResponse;
    }

    /// Creates the empty snapshot carried by initial streaming lifecycle events.
    public static func inProgress(
        responseId: String, createdAt: UInt64, modelId: String, instructions: String?
    ) -> OpenAiResponse {
        return OpenAiResponse(
            responseIdentifier: responseId,
            objectKind: "response",
            createdTimestamp: createdAt,
            completedTimestamp: nil,
            lifecycleStatus: .inProgress,
            responseError: nil,
            incompleteDetails: nil,
            instructionsText: instructions,
            metadataEntries: Array<OpenAiMetadataEntry>(),
            modelIdentifier: modelId,
            outputItems: Array<OpenAiResponseOutputItem>(),
            outputText: "",
            parallelToolCallsEnabled: true,
            temperatureValue: nil,
            toolChoiceName: "auto",
            declaredTools: Array<OpenAiResponseFunctionTool>(),
            topPValue: nil,
            maxOutputTokensBudget: nil,
            previousResponseIdentifier: nil,
            truncationMode: "disabled",
            usageCounts: nil);
    }

    /// Creates one completed local response with default request-echo fields.
    public static func completed(
        responseId: String, createdAt: UInt64, completedAt: UInt64, modelId: String,
        instructions: String?, output: Array<OpenAiResponseOutputItem>, usage: OpenAiResponseUsage
    ) -> OpenAiResponse {
        var outputText: String = "";
        for outputItem: OpenAiResponseOutputItem in output {
            if let itemMessageText: String = outputItem.messageText() {
                outputText = outputText + itemMessageText;
            }
        }
        return OpenAiResponse(
            responseIdentifier: responseId,
            objectKind: "response",
            createdTimestamp: createdAt,
            completedTimestamp: completedAt,
            lifecycleStatus: .completed,
            responseError: nil,
            incompleteDetails: nil,
            instructionsText: instructions,
            metadataEntries: Array<OpenAiMetadataEntry>(),
            modelIdentifier: modelId,
            outputItems: output,
            outputText: outputText,
            parallelToolCallsEnabled: true,
            temperatureValue: nil,
            toolChoiceName: "auto",
            declaredTools: Array<OpenAiResponseFunctionTool>(),
            topPValue: nil,
            maxOutputTokensBudget: nil,
            previousResponseIdentifier: nil,
            truncationMode: "disabled",
            usageCounts: usage);
    }

    /// Creates one response interrupted by its generated-token ceiling.
    public static func incompleteAtOutputTokenLimit(
        responseId: String, createdAt: UInt64, modelId: String, instructions: String?,
        output: Array<OpenAiResponseOutputItem>, usage: OpenAiResponseUsage
    ) -> OpenAiResponse {
        var incompleteOutput: Array<OpenAiResponseOutputItem> = output;
        for outputItemIndex in incompleteOutput.indices {
            incompleteOutput[outputItemIndex].markIncomplete();
        }
        var interruptedResponse: OpenAiResponse = OpenAiResponse.completed(
            responseId: responseId,
            createdAt: createdAt,
            completedAt: createdAt,
            modelId: modelId,
            instructions: instructions,
            output: incompleteOutput,
            usage: usage);
        interruptedResponse.completedTimestamp = nil;
        interruptedResponse.lifecycleStatus = .incomplete;
        interruptedResponse.incompleteDetails =
            OpenAiResponseIncompleteDetails(reason: "max_output_tokens");
        return interruptedResponse;
    }

    /// Creates one failed response after the local worker has reported a request failure.
    public static func failed(
        responseId: String, createdAt: UInt64, modelId: String, instructions: String?,
        output: Array<OpenAiResponseOutputItem>, errorCode: String, errorMessage: String
    ) -> OpenAiResponse {
        var failedOutput: Array<OpenAiResponseOutputItem> = output;
        for outputItemIndex in failedOutput.indices {
            failedOutput[outputItemIndex].markIncomplete();
        }
        var outputText: String = "";
        for outputItem: OpenAiResponseOutputItem in failedOutput {
            if let itemMessageText: String = outputItem.messageText() {
                outputText = outputText + itemMessageText;
            }
        }
        return OpenAiResponse(
            responseIdentifier: responseId,
            objectKind: "response",
            createdTimestamp: createdAt,
            completedTimestamp: nil,
            lifecycleStatus: .failed,
            responseError: OpenAiResponseError(code: errorCode, message: errorMessage),
            incompleteDetails: nil,
            instructionsText: instructions,
            metadataEntries: Array<OpenAiMetadataEntry>(),
            modelIdentifier: modelId,
            outputItems: failedOutput,
            outputText: outputText,
            parallelToolCallsEnabled: true,
            temperatureValue: nil,
            toolChoiceName: "auto",
            declaredTools: Array<OpenAiResponseFunctionTool>(),
            topPValue: nil,
            maxOutputTokensBudget: nil,
            previousResponseIdentifier: nil,
            truncationMode: "disabled",
            usageCounts: nil);
    }

    /// Serializes the response exactly as the serde derive does: fields in
    /// declaration order, `None` Options emitted as JSON null.
    public func wireValue() -> JsonWireValue {
        var responseObject: JsonWireObject = JsonWireObject(entries: Array());
        responseObject.appendEntry(key: "id", value: .string(self.responseIdentifier));
        responseObject.appendEntry(key: "object", value: .string(self.objectKind));
        responseObject.appendEntry(key: "created_at", value: .unsignedInteger(self.createdTimestamp));
        responseObject.appendEntry(
            key: "completed_at",
            value: ResponsesWireSupport.optionalUnsignedInteger(self.completedTimestamp));
        responseObject.appendEntry(key: "status", value: self.lifecycleStatus.wireValue());
        responseObject.appendEntry(
            key: "error",
            value: ResponsesWireSupport.optionalEncoded(self.responseError));
        responseObject.appendEntry(
            key: "incomplete_details",
            value: ResponsesWireSupport.optionalEncoded(self.incompleteDetails));
        responseObject.appendEntry(
            key: "instructions",
            value: ResponsesWireSupport.optionalString(self.instructionsText));
        responseObject.appendEntry(key: "metadata", value: self.metadataEntries.wireValue());
        responseObject.appendEntry(key: "model", value: .string(self.modelIdentifier));
        responseObject.appendEntry(
            key: "output",
            value: JsonWireValue.mappedArray(
                self.outputItems,
                mappedWireValue: { (outputItem: OpenAiResponseOutputItem) -> JsonWireValue in
                    return outputItem.wireValue();
                }));
        responseObject.appendEntry(key: "output_text", value: .string(self.outputText));
        responseObject.appendEntry(
            key: "parallel_tool_calls", value: .boolean(self.parallelToolCallsEnabled));
        responseObject.appendEntry(
            key: "temperature", value: ResponsesWireSupport.optionalFloat32(self.temperatureValue));
        responseObject.appendEntry(key: "tool_choice", value: .string(self.toolChoiceName));
        responseObject.appendEntry(
            key: "tools",
            value: JsonWireValue.mappedArray(
                self.declaredTools,
                mappedWireValue: { (declaredTool: OpenAiResponseFunctionTool) -> JsonWireValue in
                    return declaredTool.wireValue();
                }));
        responseObject.appendEntry(
            key: "top_p", value: ResponsesWireSupport.optionalFloat32(self.topPValue));
        responseObject.appendEntry(
            key: "max_output_tokens",
            value: ResponsesWireSupport.optionalUnsignedInteger32(self.maxOutputTokensBudget));
        responseObject.appendEntry(
            key: "previous_response_id",
            value: ResponsesWireSupport.optionalString(self.previousResponseIdentifier));
        responseObject.appendEntry(key: "truncation", value: .string(self.truncationMode));
        responseObject.appendEntry(
            key: "usage", value: ResponsesWireSupport.optionalEncoded(self.usageCounts));
        return .object(responseObject);
    }
}

/// Validated request settings echoed by every lifecycle snapshot of a response.
public struct OpenAiResponseRequestConfiguration: Equatable {
    public let metadata: Array<OpenAiMetadataEntry>;
    public let temperature: Float?;
    public let topP: Float?;
    public let maxOutputTokens: UInt32?;
    public let toolChoice: String;
    public let tools: Array<OpenAiResponseFunctionTool>;

    /// Mirrors the Rust `Default` impl through the initializer defaults:
    /// empty metadata, no sampling limits, `auto` tool choice, no tools.
    public init(
        metadata: Array<OpenAiMetadataEntry> = Array<OpenAiMetadataEntry>(),
        temperature: Float? = nil, topP: Float? = nil, maxOutputTokens: UInt32? = nil,
        toolChoice: String = "auto",
        tools: Array<OpenAiResponseFunctionTool> = Array<OpenAiResponseFunctionTool>()) {
        self.metadata = metadata;
        self.temperature = temperature;
        self.topP = topP;
        self.maxOutputTokens = maxOutputTokens;
        self.toolChoice = toolChoice;
        self.tools = tools;
    }
}

/// The lifecycle state of a Responses object.
public enum OpenAiResponseStatus: Equatable {
    case inProgress;
    case completed;
    case incomplete;
    case failed;
    case cancelled;

    /// Serializes with the serde `rename_all = "snake_case"` spelling.
    public func wireValue() -> JsonWireValue {
        switch self {
        case .inProgress: return .string("in_progress");
        case .completed: return .string("completed");
        case .incomplete: return .string("incomplete");
        case .failed: return .string("failed");
        case .cancelled: return .string("cancelled");
        }
    }
}

/// Checked token accounting for one local response.
public struct OpenAiResponseUsage: Equatable {
    private let promptInputTokens: UInt32;
    private let inputTokenDetails: OpenAiResponseInputTokenDetails;
    private let generatedOutputTokens: UInt32;
    private let outputTokenDetails: OpenAiResponseOutputTokenDetails;
    private let totalTokenCount: UInt32;

    fileprivate init(
        promptInputTokens: UInt32, inputTokenDetails: OpenAiResponseInputTokenDetails,
        generatedOutputTokens: UInt32,
        outputTokenDetails: OpenAiResponseOutputTokenDetails, totalTokenCount: UInt32) {
        self.promptInputTokens = promptInputTokens;
        self.inputTokenDetails = inputTokenDetails;
        self.generatedOutputTokens = generatedOutputTokens;
        self.outputTokenDetails = outputTokenDetails;
        self.totalTokenCount = totalTokenCount;
    }

    /// Mirrors the Rust constructor: the total comes from a checked add, so
    /// overflow yields `None` instead of wrapping.
    public static func new(
        inputTokens: UInt32, outputTokens: UInt32, cachedTokens: UInt32,
        reasoningTokens: UInt32) -> OpenAiResponseUsage? {
        let (totalTokens, didOverflow) = inputTokens.addingReportingOverflow(outputTokens);
        if didOverflow {
            return nil;
        }
        return OpenAiResponseUsage(
            promptInputTokens: inputTokens,
            inputTokenDetails: OpenAiResponseInputTokenDetails(
                cacheWriteTokenCount: 0, cachedTokenCount: cachedTokens),
            generatedOutputTokens: outputTokens,
            outputTokenDetails: OpenAiResponseOutputTokenDetails(
                reasoningTokenCount: reasoningTokens),
            totalTokenCount: totalTokens);
    }

    /// Serializes the usage counts exactly as the serde derive does.
    public func wireValue() -> JsonWireValue {
        var usageObject: JsonWireObject = JsonWireObject(entries: Array());
        usageObject.appendEntry(key: "input_tokens", value: .unsignedInteger(UInt64(self.promptInputTokens)));
        usageObject.appendEntry(key: "input_tokens_details", value: self.inputTokenDetails.wireValue());
        usageObject.appendEntry(key: "output_tokens", value: .unsignedInteger(UInt64(self.generatedOutputTokens)));
        usageObject.appendEntry(key: "output_tokens_details", value: self.outputTokenDetails.wireValue());
        usageObject.appendEntry(key: "total_tokens", value: .unsignedInteger(UInt64(self.totalTokenCount)));
        return .object(usageObject);
    }
}

/// Cache-related input token accounting; private in the Rust contract.
internal struct OpenAiResponseInputTokenDetails: Equatable {
    private let cacheWriteTokenCount: UInt32;
    private let cachedTokenCount: UInt32;

    fileprivate init(cacheWriteTokenCount: UInt32, cachedTokenCount: UInt32) {
        self.cacheWriteTokenCount = cacheWriteTokenCount;
        self.cachedTokenCount = cachedTokenCount;
    }

    internal func wireValue() -> JsonWireValue {
        var detailsObject: JsonWireObject = JsonWireObject(entries: Array());
        detailsObject.appendEntry(key: "cache_write_tokens", value: .unsignedInteger(UInt64(self.cacheWriteTokenCount)));
        detailsObject.appendEntry(key: "cached_tokens", value: .unsignedInteger(UInt64(self.cachedTokenCount)));
        return .object(detailsObject);
    }
}

/// Reasoning-related output token accounting; private in the Rust contract.
internal struct OpenAiResponseOutputTokenDetails: Equatable {
    private let reasoningTokenCount: UInt32;

    fileprivate init(reasoningTokenCount: UInt32) {
        self.reasoningTokenCount = reasoningTokenCount;
    }

    internal func wireValue() -> JsonWireValue {
        var detailsObject: JsonWireObject = JsonWireObject(entries: Array());
        detailsObject.appendEntry(key: "reasoning_tokens", value: .unsignedInteger(UInt64(self.reasoningTokenCount)));
        return .object(detailsObject);
    }
}

/// Why a response finished without completion.
public struct OpenAiResponseIncompleteDetails: Equatable {
    private let incompleteReason: String;

    fileprivate init(reason: String) {
        self.incompleteReason = reason;
    }

    /// Serializes the reason exactly as the serde derive does.
    public func wireValue() -> JsonWireValue {
        var detailsObject: JsonWireObject = JsonWireObject(entries: Array());
        detailsObject.appendEntry(key: "reason", value: .string(self.incompleteReason));
        return .object(detailsObject);
    }
}

/// The failure carried by one failed response.
public struct OpenAiResponseError: Equatable {
    private let errorCodeName: String;
    private let errorMessageText: String;

    fileprivate init(code: String, message: String) {
        self.errorCodeName = code;
        self.errorMessageText = message;
    }

    /// Serializes the failure exactly as the serde derive does.
    public func wireValue() -> JsonWireValue {
        var errorObject: JsonWireObject = JsonWireObject(entries: Array());
        errorObject.appendEntry(key: "code", value: .string(self.errorCodeName));
        errorObject.appendEntry(key: "message", value: .string(self.errorMessageText));
        return .object(errorObject);
    }
}

/// Module-private serialization helpers for the response shapes.
fileprivate enum ResponsesWireSupport {

    fileprivate static func optionalUnsignedInteger(_ optionalValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue: UInt64 = optionalValue else {
            return .null;
        }
        return .unsignedInteger(unwrappedValue);
    }

    fileprivate static func optionalUnsignedInteger32(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue: UInt32 = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }

    fileprivate static func optionalFloat32(_ optionalValue: Float?) -> JsonWireValue {
        guard let unwrappedValue: Float = optionalValue else {
            return .null;
        }
        return .float32(unwrappedValue);
    }

    fileprivate static func optionalString(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue: String = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }

    fileprivate static func optionalEncoded(_ optionalPayload: OpenAiResponseError?) -> JsonWireValue {
        guard let unwrappedPayload: OpenAiResponseError = optionalPayload else {
            return .null;
        }
        return unwrappedPayload.wireValue();
    }

    fileprivate static func optionalEncoded(
        _ optionalPayload: OpenAiResponseIncompleteDetails?) -> JsonWireValue {
        guard let unwrappedPayload: OpenAiResponseIncompleteDetails = optionalPayload else {
            return .null;
        }
        return unwrappedPayload.wireValue();
    }

    fileprivate static func optionalEncoded(_ optionalPayload: OpenAiResponseUsage?) -> JsonWireValue {
        guard let unwrappedPayload: OpenAiResponseUsage = optionalPayload else {
            return .null;
        }
        return unwrappedPayload.wireValue();
    }
}

extension Array where Element == OpenAiMetadataEntry {

    /// Serializes metadata entries as a JSON object in byte-sorted key order,
    /// reproducing serde's `BTreeMap<String, String>` serialization.
    internal func wireValue() -> JsonWireValue {
        var metadataObject: JsonWireObject =
            JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        for metadataEntry: OpenAiMetadataEntry in self {
            metadataObject.appendEntry(
                key: metadataEntry.metadataKey, value: .string(metadataEntry.metadataValue));
        }
        return .object(metadataObject);
    }
}
