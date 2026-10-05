import Foundation;
import IpcProtocol;

/// One bounded OpenAI-compatible Chat Completions request. Deserialize-only
/// port of crates/rest-contract/src/openai_chat_completion_request.rs (the
/// public constants live in ChatCompletionLimits.swift and the companion
/// error enum lives in ChatCompletionValidationError.swift). Unknown
/// top-level fields are absorbed by the flattened field map (#772).
public struct OpenAiChatCompletionRequest: Equatable {
    private let modelText: String;
    private let messageHistory: Array<OpenAiChatMessage>;
    private let declaredTools: Array<OpenAiToolDefinition>;
    private let requestedToolChoice: OpenAiToolChoice?;
    private let maxTokensBudget: UInt32?;
    private let maxCompletionTokensBudget: UInt32?;
    private let temperatureSetting: Float?;
    private let nucleusTopP: Float?;
    private let frequencyPenaltySetting: Float?;
    private let presencePenaltySetting: Float?;
    private let storeRequest: Bool?;
    private let reasoningEffortName: String?;
    /// Maximum tokens the model may spend inside the thinking block before being
    /// forced to close it and start the visible response. When `nil`, the model
    /// thinks freely up to `max_tokens`.
    private let thinkingBudgetLimit: UInt32?;
    /// Coding-agent spelling of the thinking budget: the agent resolves this
    /// field name from its provider compatibility configuration and falls back
    /// to it when the server declares thinking-budget support.
    private let thinkingTokenBudgetLimit: UInt32?;
    /// Documented OpenAI-compatible alias for the thinking budget used by
    /// coding-agent traffic alongside `thinking_token_budget`.
    private let thinkingBudgetTokensAlias: UInt32?;
    /// OpenRouter-style reasoning object carrying an effort level, a direct
    /// token limit, an on/off switch, and a stream-exclusion preference.
    private let reasoningObject: ReasoningRequestObject?;
    /// Qwen-style flat switch for the thinking channel.
    private let enableThinkingFlag: Bool?;
    /// vLLM-style template-kwarg block carrying the thinking toggle.
    private let chatTemplateKwargsBlock: ChatTemplateKwargsRequestObject?;
    private let stopSequenceChoice: OpenAiStopSequences?;
    private let samplingSeed: UInt64?;
    private let streamRequested: Bool;
    private let streamOptionsChoice: OpenAiStreamOptions?;
    private let responseFormatChoice: OpenAiResponseFormat?;
    private let structuredOutputsBlock: OpenAiStructuredOutputs?;
    private let guidedGrammarText: String?;
    /// Unknown top-level fields captured by serde's flatten, kept in
    /// BTreeMap (UTF-8 byte-ordered) sequence so equality and behavior
    /// match the Rust contract exactly.
    private let unknownFields: Array<(fieldName: String, fieldValue: JsonWireValue)>;

    internal init(
        model: String, messages: Array<OpenAiChatMessage>, tools: Array<OpenAiToolDefinition>,
        toolChoice: OpenAiToolChoice?, maxTokens: UInt32?, maxCompletionTokens: UInt32?,
        temperature: Float?, topP: Float?, frequencyPenalty: Float?, presencePenalty: Float?,
        store: Bool?, reasoningEffort: String?, thinkingBudget: UInt32?,
        thinkingTokenBudget: UInt32?, thinkingBudgetTokens: UInt32?,
        reasoning: ReasoningRequestObject?, enableThinking: Bool?,
        chatTemplateKwargs: ChatTemplateKwargsRequestObject?, stop: OpenAiStopSequences?,
        seed: UInt64?, stream: Bool, streamOptions: OpenAiStreamOptions?,
        responseFormat: OpenAiResponseFormat?, structuredOutputs: OpenAiStructuredOutputs?,
        guidedGrammar: String?, unknownFields: Array<(fieldName: String, fieldValue: JsonWireValue)>) {
        self.modelText = model;
        self.messageHistory = messages;
        self.declaredTools = tools;
        self.requestedToolChoice = toolChoice;
        self.maxTokensBudget = maxTokens;
        self.maxCompletionTokensBudget = maxCompletionTokens;
        self.temperatureSetting = temperature;
        self.nucleusTopP = topP;
        self.frequencyPenaltySetting = frequencyPenalty;
        self.presencePenaltySetting = presencePenalty;
        self.storeRequest = store;
        self.reasoningEffortName = reasoningEffort;
        self.thinkingBudgetLimit = thinkingBudget;
        self.thinkingTokenBudgetLimit = thinkingTokenBudget;
        self.thinkingBudgetTokensAlias = thinkingBudgetTokens;
        self.reasoningObject = reasoning;
        self.enableThinkingFlag = enableThinking;
        self.chatTemplateKwargsBlock = chatTemplateKwargs;
        self.stopSequenceChoice = stop;
        self.samplingSeed = seed;
        self.streamRequested = stream;
        self.streamOptionsChoice = streamOptions;
        self.responseFormatChoice = responseFormat;
        self.structuredOutputsBlock = structuredOutputs;
        self.guidedGrammarText = guidedGrammar;
        self.unknownFields = unknownFields;
    }

    public static func == (
        lhsValue: OpenAiChatCompletionRequest, rhsValue: OpenAiChatCompletionRequest
    ) -> Bool {
        return lhsValue.modelText == rhsValue.modelText
            && lhsValue.messageHistory == rhsValue.messageHistory
            && lhsValue.declaredTools == rhsValue.declaredTools
            && lhsValue.requestedToolChoice == rhsValue.requestedToolChoice
            && lhsValue.maxTokensBudget == rhsValue.maxTokensBudget
            && lhsValue.maxCompletionTokensBudget == rhsValue.maxCompletionTokensBudget
            && lhsValue.temperatureSetting == rhsValue.temperatureSetting
            && lhsValue.nucleusTopP == rhsValue.nucleusTopP
            && lhsValue.frequencyPenaltySetting == rhsValue.frequencyPenaltySetting
            && lhsValue.presencePenaltySetting == rhsValue.presencePenaltySetting
            && lhsValue.storeRequest == rhsValue.storeRequest
            && lhsValue.reasoningEffortName == rhsValue.reasoningEffortName
            && lhsValue.thinkingBudgetLimit == rhsValue.thinkingBudgetLimit
            && lhsValue.thinkingTokenBudgetLimit == rhsValue.thinkingTokenBudgetLimit
            && lhsValue.thinkingBudgetTokensAlias == rhsValue.thinkingBudgetTokensAlias
            && lhsValue.reasoningObject == rhsValue.reasoningObject
            && lhsValue.enableThinkingFlag == rhsValue.enableThinkingFlag
            && lhsValue.chatTemplateKwargsBlock == rhsValue.chatTemplateKwargsBlock
            && lhsValue.stopSequenceChoice == rhsValue.stopSequenceChoice
            && lhsValue.samplingSeed == rhsValue.samplingSeed
            && lhsValue.streamRequested == rhsValue.streamRequested
            && lhsValue.streamOptionsChoice == rhsValue.streamOptionsChoice
            && lhsValue.responseFormatChoice == rhsValue.responseFormatChoice
            && lhsValue.structuredOutputsBlock == rhsValue.structuredOutputsBlock
            && lhsValue.guidedGrammarText == rhsValue.guidedGrammarText
            && OpenAiChatCompletionRequest.unknownFieldsEqual(
                lhsValue.unknownFields, rhsValue.unknownFields);
    }

    private static func unknownFieldsEqual(
        _ leftUnknownFields: Array<(fieldName: String, fieldValue: JsonWireValue)>,
        _ rightUnknownFields: Array<(fieldName: String, fieldValue: JsonWireValue)>
    ) -> Bool {
        if leftUnknownFields.count != rightUnknownFields.count {
            return false;
        }
        for unknownFieldIndex: Int in leftUnknownFields.indices {
            let leftUnknownField: (fieldName: String, fieldValue: JsonWireValue) =
                leftUnknownFields[unknownFieldIndex];
            let rightUnknownField: (fieldName: String, fieldValue: JsonWireValue) =
                rightUnknownFields[unknownFieldIndex];
            if leftUnknownField.fieldName != rightUnknownField.fieldName
                || leftUnknownField.fieldValue != rightUnknownField.fieldValue {
                return false;
            }
        }
        return true;
    }

    /// Validates request bounds before the supervisor sends structured data to the worker.
    /// Unknown top-level fields are deliberately absorbed and ignored: callers replay
    /// provider-specific options that the local endpoint has no use for, and one
    /// unsupported option must not reject the whole request (#772).
    public func validate() throws -> Void {
        try ChatCompletionRequestValidation.validateNonEmptyString(
            fieldName: "model", stringValue: self.modelText);
        if self.messageHistory.isEmpty {
            throw OpenAiChatCompletionValidationError.emptyMessages;
        }
        for chatMessage: OpenAiChatMessage in self.messageHistory {
            try chatMessage.validate();
        }
        for toolDefinition: OpenAiToolDefinition in self.declaredTools {
            _ = try toolDefinition.validate();
        }
        try self.validateToolChoice();
        try self.validateOutputTokenBudget();
        _ = try self.resolveThinkingControls();
        try ChatCompletionRequestValidation.validateSamplingParameter(
            parameterName: "temperature", parameterValue: self.temperatureSetting,
            minimum: 0.0, maximum: 2.0);
        try ChatCompletionRequestValidation.validateSamplingParameter(
            parameterName: "top_p", parameterValue: self.nucleusTopP,
            minimum: 0.0, maximum: 1.0);
        try self.validateUnsupportedOptions();
        if self.stopSequenceChoice != nil {
            throw OpenAiChatCompletionValidationError.unsupportedStopSequences;
        }
        if let requestedResponseFormat: OpenAiResponseFormat = self.responseFormatChoice {
            _ = try OpenAiChatCompletionRequest.structuredOutputFrom(requestedResponseFormat);
        }
        _ = try OpenAiChatCompletionRequest.enforcedGenerationFromExtraBody(
            structuredOutputs: self.structuredOutputsBlock, guidedGrammar: self.guidedGrammarText);
    }

    /// Returns the requested model ID after request validation.
    public func model() -> String {
        return self.modelText;
    }

    /// Returns ordered conversation history after request validation.
    public func messages() -> Array<OpenAiChatMessage> {
        return self.messageHistory;
    }

    /// Returns declared callable tools after request validation.
    public func tools() -> Array<OpenAiToolDefinition> {
        return self.declaredTools;
    }

    /// Returns whether the caller requested an SSE response.
    public func stream() -> Bool {
        return self.streamRequested;
    }

    /// Returns whether the final streamed chunk must contain usage information.
    public func includesUsageInStream() -> Bool {
        if let resolvedStreamOptions: OpenAiStreamOptions = self.streamOptionsChoice {
            return resolvedStreamOptions.includeUsage;
        }
        return false;
    }

    /// Returns the selected generated-token budget after request validation.
    public func maximumOutputTokens() -> UInt32 {
        return self.maxCompletionTokensBudget ?? self.maxTokensBudget
            ?? ChatCompletionLimits.DEFAULT_OPENAI_OUTPUT_TOKENS;
    }

    /// Returns the optional deterministic sampling seed after request validation.
    public func seed() -> UInt64? {
        return self.samplingSeed;
    }

    /// Resolves every submitted thinking-control spelling into one hard budget
    /// plus the stream exclusion preference, mirroring the output-token budget
    /// rule: several spellings are fine while they agree, and disagreement is
    /// a caller bug that must fail loudly instead of silently picking one.
    private func resolveThinkingControls() throws -> ThinkingControls {
        do {
            return try ThinkingControlsInputs(
                thinkingBudget: self.thinkingBudgetLimit,
                thinkingTokenBudget: self.thinkingTokenBudgetLimit,
                thinkingBudgetTokens: self.thinkingBudgetTokensAlias,
                reasoning: self.reasoningObject,
                reasoningEffort: self.reasoningEffortName,
                enableThinking: self.enableThinkingFlag,
                chatTemplateKwargs: self.chatTemplateKwargsBlock
            ).resolve();
        } catch let resolutionError as ThinkingControlsError {
            throw OpenAiChatCompletionValidationError.thinkingControls(resolutionError);
        }
    }

    /// Compiles `response_format` into its validated structured output,
    /// wrapping the public structured-output rejection the way Rust's `?` on
    /// `into_structured_output()` does through the `#[from]` conversion.
    private static func structuredOutputFrom(
        _ requestedResponseFormat: OpenAiResponseFormat
    ) throws -> OpenAiStructuredOutput? {
        do {
            return try requestedResponseFormat.intoStructuredOutput();
        } catch let structuredOutputError as OpenAiStructuredOutputValidationError {
            throw OpenAiChatCompletionValidationError.structuredOutput(structuredOutputError);
        }
    }

    /// Compiles extra-body `structured_outputs` or `guided_grammar`, wrapping
    /// the extra-body rejection the way Rust's `?` on
    /// `enforced_generation_from_extra_body()` does through `#[from]`.
    private static func enforcedGenerationFromExtraBody(
        structuredOutputs structuredOutputsBlock: OpenAiStructuredOutputs?,
        guidedGrammar guidedGrammarText: String?
    ) throws -> EnforcedStructuredGeneration? {
        do {
            return try EnforcedStructuredGeneration.enforced_generation_from_extra_body(
                structuredOutputs: structuredOutputsBlock, guidedGrammar: guidedGrammarText);
        } catch let structuredOutputsError as OpenAiStructuredOutputsValidationError {
            throw OpenAiChatCompletionValidationError.structuredOutputs(structuredOutputsError);
        }
    }

    /// Validates and consumes this REST DTO into protocol-neutral request parts.
    public func intoParts() throws -> OpenAiChatCompletionRequestParts {
        try self.validate();
        let maximumOutputTokensBudget: UInt32 = self.maximumOutputTokens();
        let requestedMaximumOutputTokensBudget: UInt32? =
            self.maxCompletionTokensBudget ?? self.maxTokensBudget;
        let streamUsageIncluded: Bool = self.includesUsageInStream();
        var structuredOutputChoice: OpenAiStructuredOutput? = nil;
        if let requestedResponseFormat: OpenAiResponseFormat = self.responseFormatChoice {
            structuredOutputChoice = try OpenAiChatCompletionRequest.structuredOutputFrom(
                requestedResponseFormat);
        }
        let enforcedStructuredGenerationChoice: EnforcedStructuredGeneration? =
            try OpenAiChatCompletionRequest.enforcedGenerationFromExtraBody(
                structuredOutputs: self.structuredOutputsBlock, guidedGrammar: self.guidedGrammarText);
        let resolvedThinkingControls: ThinkingControls = try self.resolveThinkingControls();
        var translatedMessageParts: Array<OpenAiChatMessageParts> = Array();
        for chatMessage: OpenAiChatMessage in self.messageHistory {
            translatedMessageParts.append(try chatMessage.intoParts());
        }
        var translatedToolParts: Array<OpenAiToolDefinitionParts> = Array();
        for toolDefinition: OpenAiToolDefinition in self.declaredTools {
            translatedToolParts.append(try toolDefinition.intoParts());
        }
        var resolvedToolChoice: OpenAiToolChoiceMode = OpenAiToolChoiceMode.auto;
        if let requestedToolChoice: OpenAiToolChoice = self.requestedToolChoice {
            resolvedToolChoice = requestedToolChoice.intoMode();
        }
        return OpenAiChatCompletionRequestParts(
            model: self.modelText,
            messages: translatedMessageParts,
            tools: translatedToolParts,
            toolChoice: resolvedToolChoice,
            maximumOutputTokens: maximumOutputTokensBudget,
            requestedMaximumOutputTokens: requestedMaximumOutputTokensBudget,
            temperature: self.temperatureSetting,
            topP: self.nucleusTopP,
            seed: self.samplingSeed,
            thinkingBudget: resolvedThinkingControls.budget,
            reasoningExcluded: resolvedThinkingControls.reasoningExcluded,
            stream: self.streamRequested,
            includesUsageInStream: streamUsageIncluded,
            structuredOutput: structuredOutputChoice,
            enforcedStructuredGeneration: enforcedStructuredGenerationChoice);
    }

    private func validateOutputTokenBudget() throws -> Void {
        if let requestedMaxTokens: UInt32 = self.maxTokensBudget {
            if let requestedMaxCompletionTokens: UInt32 = self.maxCompletionTokensBudget {
                if requestedMaxTokens != requestedMaxCompletionTokens {
                    throw OpenAiChatCompletionValidationError.conflictingOutputTokenLimits(
                        maxTokens: requestedMaxTokens,
                        maxCompletionTokens: requestedMaxCompletionTokens);
                }
            }
        }
        let actualOutputTokens: UInt32 = self.maximumOutputTokens();
        if actualOutputTokens == 0
            || actualOutputTokens > ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS {
            throw OpenAiChatCompletionValidationError.outputTokenCountOutOfRange(
                actualOutputTokens: actualOutputTokens,
                maximumOutputTokens: ChatCompletionLimits.MAX_OPENAI_OUTPUT_TOKENS);
        }
    }

    private func validateToolChoice() throws -> Void {
        guard let selectedToolChoice: OpenAiToolChoice = self.requestedToolChoice else {
            return;
        }
        var declaredToolNames: Array<String> = Array();
        for toolDefinition: OpenAiToolDefinition in self.declaredTools {
            declaredToolNames.append(toolDefinition.name());
        }
        switch selectedToolChoice {
        case .mode(let selectedMode) where selectedMode == "auto" || selectedMode == "none":
            return;
        case .mode(let selectedMode):
            throw OpenAiChatCompletionValidationError.unsupportedToolChoice(mode: selectedMode);
        case .function(_, let selectedFunction):
            if declaredToolNames.contains(selectedFunction.name()) {
                throw OpenAiChatCompletionValidationError.unsupportedForcedToolChoice(
                    functionName: selectedFunction.name());
            }
            throw OpenAiChatCompletionValidationError.toolChoiceNamesUnknownFunction(
                functionName: selectedFunction.name());
        }
    }

    private func validateUnsupportedOptions() throws -> Void {
        if self.frequencyPenaltySetting != nil {
            throw OpenAiChatCompletionValidationError.unsupportedOption(
                optionName: "frequency_penalty");
        }
        if self.presencePenaltySetting != nil {
            throw OpenAiChatCompletionValidationError.unsupportedOption(
                optionName: "presence_penalty");
        }
        if self.storeRequest != nil {
            throw OpenAiChatCompletionValidationError.unsupportedOption(optionName: "store");
        }
    }
}

/// Validated OpenAI request data ready for translation at the supervisor boundary.
public struct OpenAiChatCompletionRequestParts: Equatable {
    /// Exact model ID targeted by the client.
    public let model: String;
    /// Ordered text-only chat history.
    public let messages: Array<OpenAiChatMessageParts>;
    /// Declared function tools.
    public let tools: Array<OpenAiToolDefinitionParts>;
    /// Requested tool-selection policy.
    public let toolChoice: OpenAiToolChoiceMode;
    /// Bounded generated-token budget.
    public let maximumOutputTokens: UInt32;
    /// Caller-supplied generated-token budget, preserving omission for model defaults.
    public let requestedMaximumOutputTokens: UInt32?;
    /// Optional OpenAI-compatible temperature.
    public let temperature: Float?;
    /// Optional OpenAI-compatible nucleus threshold.
    public let topP: Float?;
    /// Optional deterministic sampler seed.
    public let seed: UInt64?;
    /// Maximum tokens the model may spend inside the thinking block; `Optional(0)`
    /// is an explicit disable that renders the thinking channel closed.
    public let thinkingBudget: UInt32?;
    /// Whether reasoning deltas must be withheld from the response stream.
    public let reasoningExcluded: Bool;
    /// Whether the client requested SSE streaming.
    public let stream: Bool;
    /// Whether the stream's terminal event must carry usage.
    public let includesUsageInStream: Bool;
    /// Validated OpenAI structured-output request, when the client asked for JSON.
    public let structuredOutput: OpenAiStructuredOutput?;
    /// Extra-body constraint that must be token-masked or the request fails.
    public let enforcedStructuredGeneration: EnforcedStructuredGeneration?;

    public init(
        model: String, messages: Array<OpenAiChatMessageParts>,
        tools: Array<OpenAiToolDefinitionParts>, toolChoice: OpenAiToolChoiceMode,
        maximumOutputTokens: UInt32, requestedMaximumOutputTokens: UInt32?,
        temperature: Float?, topP: Float?, seed: UInt64?, thinkingBudget: UInt32?,
        reasoningExcluded: Bool, stream: Bool, includesUsageInStream: Bool,
        structuredOutput: OpenAiStructuredOutput?,
        enforcedStructuredGeneration: EnforcedStructuredGeneration?) {
        self.model = model;
        self.messages = messages;
        self.tools = tools;
        self.toolChoice = toolChoice;
        self.maximumOutputTokens = maximumOutputTokens;
        self.requestedMaximumOutputTokens = requestedMaximumOutputTokens;
        self.temperature = temperature;
        self.topP = topP;
        self.seed = seed;
        self.thinkingBudget = thinkingBudget;
        self.reasoningExcluded = reasoningExcluded;
        self.stream = stream;
        self.includesUsageInStream = includesUsageInStream;
        self.structuredOutput = structuredOutput;
        self.enforcedStructuredGeneration = enforcedStructuredGeneration;
    }
}

/// Free validation helpers mirroring the private functions at the bottom of
/// crates/rest-contract/src/openai_chat_completion_request.rs.
fileprivate enum ChatCompletionRequestValidation {

    fileprivate static func validateNonEmptyString(fieldName: String, stringValue: String) throws -> Void {
        if stringValue.isEmpty {
            throw OpenAiChatCompletionValidationError.emptyString(fieldName: fieldName);
        }
    }

    fileprivate static func validateSamplingParameter(
        parameterName: String, parameterValue: Float?, minimum: Float, maximum: Float
    ) throws -> Void {
        guard let resolvedParameterValue: Float = parameterValue else {
            return;
        }
        if resolvedParameterValue.isFinite
            && resolvedParameterValue >= minimum && resolvedParameterValue <= maximum {
            return;
        }
        throw OpenAiChatCompletionValidationError.samplingParameterOutOfRange(
            parameterName: parameterName,
            minimum: ChatCompletionRequestValidation.rustFloat32Display(minimum),
            maximum: ChatCompletionRequestValidation.rustFloat32Display(maximum));
    }

    /// Rust `f32` Display formatting (`0.0f32.to_string()` prints `0`) so the
    /// error text stays byte-identical to the thiserror output; Swift prints
    /// the same numbers as `0.0`.
    fileprivate static func rustFloat32Display(_ floatValue: Float) -> String {
        let isIntegralNumber: Bool =
            floatValue.isFinite && floatValue == floatValue.rounded();
        if isIntegralNumber && floatValue >= -2_147_483_648.0 && floatValue <= 2_147_483_647.0 {
            return String(Int(floatValue));
        }
        return String(floatValue);
    }
}
