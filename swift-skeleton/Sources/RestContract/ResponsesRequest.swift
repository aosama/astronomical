// ResponsesRequest.swift — RestContract
//
// Port of crates/rest-contract/src/openai_responses_request.rs.

import Foundation;
import IpcProtocol;

/// One unknown request field absorbed by serde's `#[serde(flatten)]`. Entries
/// are kept byte-sorted by field name so first-key rejections match the Rust
/// `BTreeMap<String, Value>` ordering.
public struct OpenAiUnknownRequestField: Equatable {
    public let fieldName: String;
    public let fieldValue: JsonWireValue;

    public init(fieldName: String, fieldValue: JsonWireValue) {
        self.fieldName = fieldName;
        self.fieldValue = fieldValue;
    }
}

/// One bounded request to the local OpenAI-compatible Responses endpoint.
public struct OpenAiResponsesRequest: Equatable {
    private let modelText: String;
    private let inputValue: OpenAiResponseInput;
    private let instructionsText: String?;
    private let declaredTools: Array<OpenAiResponseToolDefinition>;
    private let toolChoiceSelection: OpenAiResponseToolChoice?;
    private let metadataEntries: Array<OpenAiMetadataEntry>;
    private let storeFlag: Bool?;
    private let backgroundFlag: Bool?;
    private let truncationMode: String?;
    private let serviceTierName: String?;
    private let userLabel: String?;
    private let safetyIdentifier: String?;
    private let promptCacheKey: String?;
    private let previousResponseIdentifier: String?;
    private let conversationValue: JsonWireValue?;
    private let contextManagementValue: JsonWireValue?;
    private let includeValue: JsonWireValue?;
    private let moderationValue: JsonWireValue?;
    private let promptValue: JsonWireValue?;
    private let promptCacheOptionsValue: JsonWireValue?;
    private let promptCacheRetentionMode: String?;
    private let reasoningValue: ReasoningRequestObject?;
    /// OpenAI `reasoning_effort` level name; resolved through the shared
    /// thinking-control resolution.
    private let reasoningEffortName: String?;
    /// Qwen-style flat switch for the thinking channel.
    private let enableThinkingFlag: Bool?;
    /// vLLM-style template-kwarg block carrying the thinking toggle.
    private let chatTemplateKwargsValue: ChatTemplateKwargsRequestObject?;
    private let streamOptionsValue: JsonWireValue?;
    private let textValue: JsonWireValue?;
    private let topLogprobsCount: UInt8?;
    private let parallelToolCallsFlag: Bool?;
    private let maxOutputTokensBudget: UInt32?;
    private let temperatureValue: Float?;
    private let topPValue: Float?;
    private let streamFlag: Bool;
    private let thinkingBudgetValue: UInt32?;
    /// Coding-agent spelling of the thinking budget: the agent resolves this
    /// field name from its provider compatibility configuration and falls back
    /// to it when the server declares thinking-budget support.
    private let thinkingTokenBudgetValue: UInt32?;
    /// Documented OpenAI-compatible alias for the thinking budget used by
    /// coding-agent traffic alongside `thinking_token_budget`.
    private let thinkingBudgetTokensValue: UInt32?;
    private let responseFormatValue: OpenAiResponseFormat?;
    private let structuredOutputsValue: OpenAiStructuredOutputs?;
    private let guidedGrammarText: String?;
    private let unknownFields: Array<OpenAiUnknownRequestField>;

    internal init(
        modelText: String, inputValue: OpenAiResponseInput, instructionsText: String?,
        declaredTools: Array<OpenAiResponseToolDefinition>,
        toolChoiceSelection: OpenAiResponseToolChoice?,
        metadataEntries: Array<OpenAiMetadataEntry>, storeFlag: Bool?, backgroundFlag: Bool?,
        truncationMode: String?, serviceTierName: String?, userLabel: String?,
        safetyIdentifier: String?, promptCacheKey: String?,
        previousResponseIdentifier: String?, conversationValue: JsonWireValue?,
        contextManagementValue: JsonWireValue?, includeValue: JsonWireValue?,
        moderationValue: JsonWireValue?, promptValue: JsonWireValue?,
        promptCacheOptionsValue: JsonWireValue?, promptCacheRetentionMode: String?,
        reasoningValue: ReasoningRequestObject?, reasoningEffortName: String?,
        enableThinkingFlag: Bool?, chatTemplateKwargsValue: ChatTemplateKwargsRequestObject?,
        streamOptionsValue: JsonWireValue?, textValue: JsonWireValue?,
        topLogprobsCount: UInt8?, parallelToolCallsFlag: Bool?,
        maxOutputTokensBudget: UInt32?, temperatureValue: Float?, topPValue: Float?,
        streamFlag: Bool, thinkingBudgetValue: UInt32?, thinkingTokenBudgetValue: UInt32?,
        thinkingBudgetTokensValue: UInt32?, responseFormatValue: OpenAiResponseFormat?,
        structuredOutputsValue: OpenAiStructuredOutputs?, guidedGrammarText: String?,
        unknownFields: Array<OpenAiUnknownRequestField>) {
        self.modelText = modelText;
        self.inputValue = inputValue;
        self.instructionsText = instructionsText;
        self.declaredTools = declaredTools;
        self.toolChoiceSelection = toolChoiceSelection;
        self.metadataEntries = metadataEntries;
        self.storeFlag = storeFlag;
        self.backgroundFlag = backgroundFlag;
        self.truncationMode = truncationMode;
        self.serviceTierName = serviceTierName;
        self.userLabel = userLabel;
        self.safetyIdentifier = safetyIdentifier;
        self.promptCacheKey = promptCacheKey;
        self.previousResponseIdentifier = previousResponseIdentifier;
        self.conversationValue = conversationValue;
        self.contextManagementValue = contextManagementValue;
        self.includeValue = includeValue;
        self.moderationValue = moderationValue;
        self.promptValue = promptValue;
        self.promptCacheOptionsValue = promptCacheOptionsValue;
        self.promptCacheRetentionMode = promptCacheRetentionMode;
        self.reasoningValue = reasoningValue;
        self.reasoningEffortName = reasoningEffortName;
        self.enableThinkingFlag = enableThinkingFlag;
        self.chatTemplateKwargsValue = chatTemplateKwargsValue;
        self.streamOptionsValue = streamOptionsValue;
        self.textValue = textValue;
        self.topLogprobsCount = topLogprobsCount;
        self.parallelToolCallsFlag = parallelToolCallsFlag;
        self.maxOutputTokensBudget = maxOutputTokensBudget;
        self.temperatureValue = temperatureValue;
        self.topPValue = topPValue;
        self.streamFlag = streamFlag;
        self.thinkingBudgetValue = thinkingBudgetValue;
        self.thinkingTokenBudgetValue = thinkingTokenBudgetValue;
        self.thinkingBudgetTokensValue = thinkingBudgetTokensValue;
        self.responseFormatValue = responseFormatValue;
        self.structuredOutputsValue = structuredOutputsValue;
        self.guidedGrammarText = guidedGrammarText;
        self.unknownFields = unknownFields;
    }

    /// Resolves every submitted thinking-control spelling into one hard budget
    /// plus the stream exclusion preference, mirroring the output-token budget
    /// rule: several spellings are fine while they agree, and disagreement is
    /// a caller bug that must fail loudly instead of silently picking one.
    private func resolveThinkingControls() throws -> ThinkingControls {
        do {
            return try ThinkingControlsInputs(
                thinkingBudget: self.thinkingBudgetValue,
                thinkingTokenBudget: self.thinkingTokenBudgetValue,
                thinkingBudgetTokens: self.thinkingBudgetTokensValue,
                reasoning: self.reasoningValue,
                reasoningEffort: self.reasoningEffortName,
                enableThinking: self.enableThinkingFlag,
                chatTemplateKwargs: self.chatTemplateKwargsValue
            ).resolve();
        } catch let thinkingControlsError as ThinkingControlsError {
            throw OpenAiResponsesValidationError.thinkingControls(thinkingControlsError);
        }
    }

    /// Validates and consumes this public request into protocol-neutral parts.
    public func intoParts() throws -> OpenAiResponsesRequestParts {
        if let firstUnknownField: OpenAiUnknownRequestField = self.unknownFields.first {
            throw OpenAiResponsesValidationError.unknownField(fieldName: firstUnknownField.fieldName);
        }
        if self.modelText.isEmpty {
            throw OpenAiResponsesValidationError.emptyModel;
        }
        let maximumOutputTokens: UInt32 =
            self.maxOutputTokensBudget ?? ChatCompletionLimits.DEFAULT_OPENAI_OUTPUT_TOKENS;
        let requestedMaximumOutputTokens: UInt32? = self.maxOutputTokensBudget;
        if maximumOutputTokens == 0 || maximumOutputTokens > ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS {
            throw OpenAiResponsesValidationError.outputTokenCountOutOfRange(
                actualOutputTokens: maximumOutputTokens,
                maximumOutputTokens: ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS);
        }
        try OpenAiResponsesRequest.validateSamplingParameter(
            parameterName: "temperature", parameterValue: self.temperatureValue,
            minimum: 0.0, maximum: 2.0);
        try OpenAiResponsesRequest.validateSamplingParameter(
            parameterName: "top_p", parameterValue: self.topPValue, minimum: 0.0, maximum: 1.0);
        let thinkingControls: ThinkingControls = try self.resolveThinkingControls();
        var responseFormatStructuredOutput: OpenAiStructuredOutput? = nil;
        if let responseFormat: OpenAiResponseFormat = self.responseFormatValue {
            responseFormatStructuredOutput = try OpenAiResponsesRequest.wrappedStructuredOutput({
                return try responseFormat.intoStructuredOutput();
            });
        }
        let textFormatStructuredOutput: OpenAiStructuredOutput? =
            try OpenAiResponsesRequest.wrappedStructuredOutput({
                return try OpenAiStructuredOutput.structured_output_from_responses_text_format(self.textValue);
            });
        let structuredOutput: OpenAiStructuredOutput? =
            try OpenAiResponsesRequest.wrappedStructuredOutputMerge(
                responseFormat: responseFormatStructuredOutput,
                textFormat: textFormatStructuredOutput);
        let enforcedStructuredGeneration: EnforcedStructuredGeneration? =
            try OpenAiResponsesRequest.wrappedEnforcedGeneration(
                structuredOutputs: self.structuredOutputsValue,
                guidedGrammar: self.guidedGrammarText);
        try OpenAiResponsesRequest.validateCompatibilityFields(request: self);
        var translatedTools: Array<OpenAiResponseToolDefinitionParts> =
            Array<OpenAiResponseToolDefinitionParts>();
        translatedTools.reserveCapacity(self.declaredTools.count);
        for declaredTool: OpenAiResponseToolDefinition in self.declaredTools {
            translatedTools.append(try declaredTool.intoParts());
        }
        var translatedToolChoice: OpenAiResponseToolChoiceParts = .auto;
        if let declaredToolChoice: OpenAiResponseToolChoice = self.toolChoiceSelection {
            translatedToolChoice = try declaredToolChoice.intoParts();
        }
        return OpenAiResponsesRequestParts(
            model: self.modelText,
            input: try self.inputValue.intoParts(),
            instructions: self.instructionsText,
            tools: translatedTools,
            toolChoice: translatedToolChoice,
            metadata: self.metadataEntries,
            maximumOutputTokens: maximumOutputTokens,
            requestedMaximumOutputTokens: requestedMaximumOutputTokens,
            temperature: self.temperatureValue,
            topP: self.topPValue,
            stream: self.streamFlag,
            thinkingBudget: thinkingControls.budget,
            reasoningExcluded: thinkingControls.reasoningExcluded,
            structuredOutput: structuredOutput,
            enforcedStructuredGeneration: enforcedStructuredGeneration);
    }

    /// Rust wraps structured-output rejections at the Responses boundary via
    /// `?` plus `#[from]`; these helpers reproduce that transparent wrapping so
    /// callers always see the OpenAiResponsesValidationError shape.
    private static func wrappedStructuredOutput(
        _ decodeBody: () throws -> OpenAiStructuredOutput?) throws -> OpenAiStructuredOutput? {
        do {
            return try decodeBody();
        } catch let structuredOutputError as OpenAiStructuredOutputValidationError {
            throw OpenAiResponsesValidationError.structuredOutput(structuredOutputError);
        }
    }

    private static func wrappedStructuredOutputMerge(
        responseFormat: OpenAiStructuredOutput?, textFormat: OpenAiStructuredOutput?) throws -> OpenAiStructuredOutput? {
        do {
            return try OpenAiStructuredOutput.merge_structured_output_requests(
                responseFormat: responseFormat, textFormat: textFormat);
        } catch let structuredOutputError as OpenAiStructuredOutputValidationError {
            throw OpenAiResponsesValidationError.structuredOutput(structuredOutputError);
        }
    }

    private static func wrappedEnforcedGeneration(
        structuredOutputs: OpenAiStructuredOutputs?, guidedGrammar: String?) throws -> EnforcedStructuredGeneration? {
        do {
            return try EnforcedStructuredGeneration.enforced_generation_from_extra_body(
                structuredOutputs: structuredOutputs, guidedGrammar: guidedGrammar);
        } catch let structuredOutputsError as OpenAiStructuredOutputsValidationError {
            throw OpenAiResponsesValidationError.structuredOutputs(structuredOutputsError);
        }
    }

    /// Module-private translation of the free
    /// `validate_compatibility_fields` helper: every recognized option this
    /// endpoint cannot honor is rejected by its exact request spelling.
    private static func validateCompatibilityFields(
        request: OpenAiResponsesRequest) throws -> Void {
        if let storeFlag: Bool = request.storeFlag {
            if storeFlag {
                throw OpenAiResponsesValidationError.unsupportedOption(optionName: "store=true");
            }
        }
        let rejectedWhenPresent: Array<(optionName: String, isPresent: Bool)> = [
            ("previous_response_id", request.previousResponseIdentifier != nil),
            ("conversation", request.conversationValue != nil),
            ("context_management", request.contextManagementValue != nil),
            ("moderation", request.moderationValue != nil),
            ("prompt", request.promptValue != nil),
            ("prompt_cache_options", request.promptCacheOptionsValue != nil),
            ("prompt_cache_retention", request.promptCacheRetentionMode != nil),
        ];
        for rejectedOption: (optionName: String, isPresent: Bool) in rejectedWhenPresent {
            if rejectedOption.isPresent {
                throw OpenAiResponsesValidationError.unsupportedOption(
                    optionName: rejectedOption.optionName);
            }
        }
        if let parallelToolCallsFlag: Bool = request.parallelToolCallsFlag {
            if parallelToolCallsFlag == false {
                throw OpenAiResponsesValidationError.unsupportedOption(
                    optionName: "parallel_tool_calls=false");
            }
        }
        if let topLogprobsCount: UInt8 = request.topLogprobsCount {
            if topLogprobsCount > 0 {
                throw OpenAiResponsesValidationError.unsupportedOption(optionName: "top_logprobs");
            }
        }
        if let textConfiguration: JsonWireValue = request.textValue {
            if case let .object(textObject) = textConfiguration {
                if textObject.value(forKey: "verbosity") != nil {
                    throw OpenAiResponsesValidationError.unsupportedOption(
                        optionName: "text.verbosity");
                }
            }
        }
        if let backgroundFlag: Bool = request.backgroundFlag {
            if backgroundFlag {
                throw OpenAiResponsesValidationError.unsupportedOption(optionName: "background=true");
            }
        }
        if let truncationMode: String = request.truncationMode {
            if truncationMode != "disabled" {
                throw OpenAiResponsesValidationError.unsupportedOption(optionName: "truncation");
            }
        }
        if let serviceTierName: String = request.serviceTierName {
            if serviceTierName != "auto" && serviceTierName != "default" {
                throw OpenAiResponsesValidationError.unsupportedOption(optionName: "service_tier");
            }
        }
        if request.metadataEntries.count > 16 {
            throw OpenAiResponsesValidationError.metadataEntryCountExceeded;
        }
        for metadataEntry: OpenAiMetadataEntry in request.metadataEntries {
            // Rust measures `str::len` in UTF-8 bytes.
            if metadataEntry.metadataKey.utf8.count > 64
                || metadataEntry.metadataValue.utf8.count > 512 {
                throw OpenAiResponsesValidationError.metadataTextTooLong;
            }
        }
    }

    /// Module-private translation of the free `validate_sampling_parameter`
    /// helper: finite values inside the closed range pass, everything else
    /// fails with the Rust `Display` rendering of the range bounds.
    private static func validateSamplingParameter(
        parameterName: String, parameterValue: Float?, minimum: Float, maximum: Float
    ) throws -> Void {
        guard let resolvedParameterValue: Float = parameterValue else {
            return;
        }
        if resolvedParameterValue.isFinite
            && resolvedParameterValue >= minimum
            && resolvedParameterValue <= maximum {
            return;
        }
        throw OpenAiResponsesValidationError.samplingParameterOutOfRange(
            parameterName: parameterName,
            minimum: rustFloatDisplay(minimum),
            maximum: rustFloatDisplay(maximum));
    }

    /// Rust renders integral `f32` values without a trailing `.0` in error
    /// text (`0..=2`, not `0.0..=2.0`).
    private static func rustFloatDisplay(_ floatValue: Float) -> String {
        if floatValue.isFinite && floatValue == floatValue.rounded(.towardZero) {
            return String(Int(floatValue));
        }
        return String(floatValue);
    }
}

/// Validated Responses request data ready for supervisor translation.
public struct OpenAiResponsesRequestParts: Equatable {
    public let model: String;
    public let input: OpenAiResponseInputParts;
    public let instructions: String?;
    public let tools: Array<OpenAiResponseToolDefinitionParts>;
    public let toolChoice: OpenAiResponseToolChoiceParts;
    public let metadata: Array<OpenAiMetadataEntry>;
    public let maximumOutputTokens: UInt32;
    public let requestedMaximumOutputTokens: UInt32?;
    public let temperature: Float?;
    public let topP: Float?;
    public let stream: Bool;
    public let thinkingBudget: UInt32?;
    /// Whether reasoning deltas must be withheld from the response stream.
    public let reasoningExcluded: Bool;
    public let structuredOutput: OpenAiStructuredOutput?;
    public let enforcedStructuredGeneration: EnforcedStructuredGeneration?;

    public init(
        model: String, input: OpenAiResponseInputParts, instructions: String?,
        tools: Array<OpenAiResponseToolDefinitionParts>,
        toolChoice: OpenAiResponseToolChoiceParts, metadata: Array<OpenAiMetadataEntry>,
        maximumOutputTokens: UInt32, requestedMaximumOutputTokens: UInt32?,
        temperature: Float?, topP: Float?, stream: Bool, thinkingBudget: UInt32?,
        reasoningExcluded: Bool, structuredOutput: OpenAiStructuredOutput?,
        enforcedStructuredGeneration: EnforcedStructuredGeneration?) {
        self.model = model;
        self.input = input;
        self.instructions = instructions;
        self.tools = tools;
        self.toolChoice = toolChoice;
        self.metadata = metadata;
        self.maximumOutputTokens = maximumOutputTokens;
        self.requestedMaximumOutputTokens = requestedMaximumOutputTokens;
        self.temperature = temperature;
        self.topP = topP;
        self.stream = stream;
        self.thinkingBudget = thinkingBudget;
        self.reasoningExcluded = reasoningExcluded;
        self.structuredOutput = structuredOutput;
        self.enforcedStructuredGeneration = enforcedStructuredGeneration;
    }

    /// Copies only the bounded settings required in returned Response objects.
    public func responseConfiguration() -> OpenAiResponseRequestConfiguration {
        var echoedTools: Array<OpenAiResponseFunctionTool> = Array<OpenAiResponseFunctionTool>();
        echoedTools.reserveCapacity(self.tools.count);
        for functionTool: OpenAiResponseToolDefinitionParts in self.tools {
            echoedTools.append(OpenAiResponseFunctionTool.new(
                name: functionTool.name,
                description: functionTool.description,
                parameters: functionTool.parameters,
                strict: functionTool.strict));
        }
        return OpenAiResponseRequestConfiguration(
            metadata: self.metadata,
            temperature: self.temperature,
            topP: self.topP,
            maxOutputTokens: self.requestedMaximumOutputTokens,
            toolChoice: self.toolChoice.kindName(),
            tools: echoedTools);
    }
}

/// A request rejected before worker admission by the Responses contract.
public enum OpenAiResponsesValidationError: Error, Equatable {
    case emptyModel;
    case emptyInputItems;
    case emptyContentParts;
    case imageInputOutsideUserMessage;
    /// A reasoning-control spelling was contradictory or unrecognized.
    case thinkingControls(ThinkingControlsError);
    case imageInput(OpenAiChatCompletionValidationError);
    case unsupportedReasoningReplay;
    case unsupportedInputItem(inputItemType: String);
    case invalidToolName(toolName: String);
    case toolSchemaNestingTooDeep(actualSchemaNestingDepth: Int, maximumSchemaNestingDepth: Int);
    case unsupportedOption(optionName: String);
    case metadataEntryCountExceeded;
    case metadataTextTooLong;
    case outputTokenCountOutOfRange(actualOutputTokens: UInt32, maximumOutputTokens: UInt32);
    case samplingParameterOutOfRange(parameterName: String, minimum: String, maximum: String);
    case unknownField(fieldName: String);
    case structuredOutput(OpenAiStructuredOutputValidationError);
    case structuredOutputs(OpenAiStructuredOutputsValidationError);

    public var errorDescription: String? {
        switch self {
        case .emptyModel:
            return "model must not be empty";
        case .emptyInputItems:
            return "input items must not be empty";
        case .emptyContentParts:
            return "message content parts must not be empty";
        case .imageInputOutsideUserMessage:
            return "image input is supported only in user messages";
        case .thinkingControls(let thinkingControlsError):
            return thinkingControlsError.errorDescription;
        case .imageInput(let imageError):
            return "invalid image input: \(imageError.errorDescription ?? "")";
        case .unsupportedReasoningReplay:
            return "encrypted foreign reasoning cannot be replayed locally";
        case .unsupportedInputItem(let inputItemType):
            return "response input item type '\(inputItemType)' is unsupported";
        case .invalidToolName(let toolName):
            return "tool name '\(toolName)' is invalid";
        case .toolSchemaNestingTooDeep(let actualSchemaNestingDepth, let maximumSchemaNestingDepth):
            return "tool schema nesting depth is \(actualSchemaNestingDepth), exceeding \(maximumSchemaNestingDepth)";
        case .unsupportedOption(let optionName):
            return "request option '\(optionName)' is unsupported";
        case .metadataEntryCountExceeded:
            return "metadata exceeds the supported 16-entry limit";
        case .metadataTextTooLong:
            return "metadata key or value exceeds the supported length";
        case .outputTokenCountOutOfRange(let actualOutputTokens, let maximumOutputTokens):
            return "output token count is \(actualOutputTokens), outside the 1..=\(maximumOutputTokens) token range";
        case .samplingParameterOutOfRange(let parameterName, let minimum, let maximum):
            return "\(parameterName) is outside the supported range \(minimum)..=\(maximum)";
        case .unknownField(let fieldName):
            return "request field '\(fieldName)' is unknown";
        case .structuredOutput(let structuredOutputError):
            return structuredOutputError.errorDescription;
        case .structuredOutputs(let structuredOutputsError):
            return structuredOutputsError.errorDescription;
        }
    }
}

