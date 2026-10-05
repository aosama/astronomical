// ResponsesRequest+Decoded.swift — RestContract
//
// Wire decoding half of crates/rest-contract/src/openai_responses_request.rs:
// the serde-shaped `decoded` entry point for OpenAiResponsesRequest.

import Foundation;
import IpcProtocol;

extension OpenAiResponsesRequest {

    /// Mirrors the serde derive for a struct with `#[serde(flatten)]`: exact
    /// wire field names with duplicate rejection, missing-or-null `Option`
    /// decoding to `None`, `#[serde(default)]` fields surviving absence, and
    /// every other key collected into the flattened map.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponsesRequest {
        let requestObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var decodedModelText: String? = nil;
        var decodedInput: OpenAiResponseInput? = nil;
        var decodedInstructionsText: String? = nil;
        var decodedTools: Array<OpenAiResponseToolDefinition>? = nil;
        var decodedToolChoice: OpenAiResponseToolChoice? = nil;
        var decodedMetadataEntries: Array<OpenAiMetadataEntry>? = nil;
        var decodedStoreFlag: Bool? = nil;
        var decodedBackgroundFlag: Bool? = nil;
        var decodedTruncationMode: String? = nil;
        var decodedServiceTierName: String? = nil;
        var decodedUserLabel: String? = nil;
        var decodedSafetyIdentifier: String? = nil;
        var decodedPromptCacheKey: String? = nil;
        var decodedPreviousResponseIdentifier: String? = nil;
        var decodedConversationValue: JsonWireValue? = nil;
        var decodedContextManagementValue: JsonWireValue? = nil;
        var decodedIncludeValue: JsonWireValue? = nil;
        var decodedModerationValue: JsonWireValue? = nil;
        var decodedPromptValue: JsonWireValue? = nil;
        var decodedPromptCacheOptionsValue: JsonWireValue? = nil;
        var decodedPromptCacheRetentionMode: String? = nil;
        var decodedReasoningValue: ReasoningRequestObject? = nil;
        var decodedReasoningEffortName: String? = nil;
        var decodedEnableThinkingFlag: Bool? = nil;
        var decodedChatTemplateKwargsValue: ChatTemplateKwargsRequestObject? = nil;
        var decodedStreamOptionsValue: JsonWireValue? = nil;
        var decodedTextValue: JsonWireValue? = nil;
        var decodedTopLogprobsCount: UInt8? = nil;
        var decodedParallelToolCallsFlag: Bool? = nil;
        var decodedMaxOutputTokensBudget: UInt32? = nil;
        var decodedTemperatureValue: Float? = nil;
        var decodedTopPValue: Float? = nil;
        var decodedStreamFlag: Bool? = nil;
        var decodedThinkingBudgetValue: UInt32? = nil;
        var decodedThinkingTokenBudgetValue: UInt32? = nil;
        var decodedThinkingBudgetTokensValue: UInt32? = nil;
        var decodedResponseFormatValue: OpenAiResponseFormat? = nil;
        var decodedStructuredOutputsValue: OpenAiStructuredOutputs? = nil;
        var decodedGuidedGrammarText: String? = nil;
        var unknownFields: Array<OpenAiUnknownRequestField> = Array();
        for entry in requestObject.entries {
            switch entry.key {
            case "model":
                if decodedModelText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "model");
                }
                decodedModelText = try JsonWireValue.extractString(entry.value);
            case "input":
                if decodedInput != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "input");
                }
                decodedInput = try OpenAiResponseInput.decoded(wireValue: entry.value);
            case "instructions":
                if decodedInstructionsText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "instructions");
                }
                if entry.value.isNull == false {
                    decodedInstructionsText = try JsonWireValue.extractString(entry.value);
                }
            case "tools":
                if decodedTools != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "tools");
                }
                decodedTools = try JsonWireValue.extractArray(
                    entry.value,
                    mappedElement: { (toolWireValue: JsonWireValue) throws -> OpenAiResponseToolDefinition in
                        return try OpenAiResponseToolDefinition.decoded(wireValue: toolWireValue);
                    });
            case "tool_choice":
                if decodedToolChoice != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "tool_choice");
                }
                if entry.value.isNull == false {
                    decodedToolChoice = try OpenAiResponseToolChoice.decoded(wireValue: entry.value);
                }
            case "metadata":
                if decodedMetadataEntries != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "metadata");
                }
                decodedMetadataEntries = try ResponsesRequestWire.decodedMetadataEntries(entry.value);
            case "store":
                if decodedStoreFlag != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "store");
                }
                if entry.value.isNull == false {
                    decodedStoreFlag = try JsonWireValue.extractBool(entry.value);
                }
            case "background":
                if decodedBackgroundFlag != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "background");
                }
                if entry.value.isNull == false {
                    decodedBackgroundFlag = try JsonWireValue.extractBool(entry.value);
                }
            case "truncation":
                if decodedTruncationMode != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "truncation");
                }
                if entry.value.isNull == false {
                    decodedTruncationMode = try JsonWireValue.extractString(entry.value);
                }
            case "service_tier":
                if decodedServiceTierName != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "service_tier");
                }
                if entry.value.isNull == false {
                    decodedServiceTierName = try JsonWireValue.extractString(entry.value);
                }
            case "user":
                if decodedUserLabel != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "user");
                }
                if entry.value.isNull == false {
                    decodedUserLabel = try JsonWireValue.extractString(entry.value);
                }
            case "safety_identifier":
                if decodedSafetyIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "safety_identifier");
                }
                if entry.value.isNull == false {
                    decodedSafetyIdentifier = try JsonWireValue.extractString(entry.value);
                }
            case "prompt_cache_key":
                if decodedPromptCacheKey != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "prompt_cache_key");
                }
                if entry.value.isNull == false {
                    decodedPromptCacheKey = try JsonWireValue.extractString(entry.value);
                }
            case "previous_response_id":
                if decodedPreviousResponseIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "previous_response_id");
                }
                if entry.value.isNull == false {
                    decodedPreviousResponseIdentifier = try JsonWireValue.extractString(entry.value);
                }
            case "conversation":
                if decodedConversationValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "conversation");
                }
                if entry.value.isNull == false {
                    decodedConversationValue = entry.value;
                }
            case "context_management":
                if decodedContextManagementValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "context_management");
                }
                if entry.value.isNull == false {
                    decodedContextManagementValue = entry.value;
                }
            case "include":
                if decodedIncludeValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "include");
                }
                if entry.value.isNull == false {
                    decodedIncludeValue = entry.value;
                }
            case "moderation":
                if decodedModerationValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "moderation");
                }
                if entry.value.isNull == false {
                    decodedModerationValue = entry.value;
                }
            case "prompt":
                if decodedPromptValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "prompt");
                }
                if entry.value.isNull == false {
                    decodedPromptValue = entry.value;
                }
            case "prompt_cache_options":
                if decodedPromptCacheOptionsValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "prompt_cache_options");
                }
                if entry.value.isNull == false {
                    decodedPromptCacheOptionsValue = entry.value;
                }
            case "prompt_cache_retention":
                if decodedPromptCacheRetentionMode != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "prompt_cache_retention");
                }
                if entry.value.isNull == false {
                    decodedPromptCacheRetentionMode = try JsonWireValue.extractString(entry.value);
                }
            case "reasoning":
                if decodedReasoningValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "reasoning");
                }
                if entry.value.isNull == false {
                    decodedReasoningValue = try ReasoningRequestObject.decoded(wireValue: entry.value);
                }
            case "reasoning_effort":
                if decodedReasoningEffortName != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "reasoning_effort");
                }
                if entry.value.isNull == false {
                    decodedReasoningEffortName = try JsonWireValue.extractString(entry.value);
                }
            case "enable_thinking":
                if decodedEnableThinkingFlag != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "enable_thinking");
                }
                if entry.value.isNull == false {
                    decodedEnableThinkingFlag = try JsonWireValue.extractBool(entry.value);
                }
            case "chat_template_kwargs":
                if decodedChatTemplateKwargsValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "chat_template_kwargs");
                }
                if entry.value.isNull == false {
                    decodedChatTemplateKwargsValue = try ChatTemplateKwargsRequestObject.decoded(
                        wireValue: entry.value);
                }
            case "stream_options":
                if decodedStreamOptionsValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "stream_options");
                }
                if entry.value.isNull == false {
                    decodedStreamOptionsValue = entry.value;
                }
            case "text":
                if decodedTextValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "text");
                }
                if entry.value.isNull == false {
                    decodedTextValue = entry.value;
                }
            case "top_logprobs":
                if decodedTopLogprobsCount != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "top_logprobs");
                }
                if entry.value.isNull == false {
                    decodedTopLogprobsCount = try JsonWireValue.clampToUInt8(
                        try JsonWireValue.extractUInt64(entry.value));
                }
            case "parallel_tool_calls":
                if decodedParallelToolCallsFlag != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "parallel_tool_calls");
                }
                if entry.value.isNull == false {
                    decodedParallelToolCallsFlag = try JsonWireValue.extractBool(entry.value);
                }
            case "max_output_tokens":
                if decodedMaxOutputTokensBudget != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "max_output_tokens");
                }
                if entry.value.isNull == false {
                    decodedMaxOutputTokensBudget = try JsonWireValue.clampToUInt32(
                        try JsonWireValue.extractUInt64(entry.value));
                }
            case "temperature":
                if decodedTemperatureValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "temperature");
                }
                if entry.value.isNull == false {
                    decodedTemperatureValue = try JsonWireValue.extractFloat32(entry.value);
                }
            case "top_p":
                if decodedTopPValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "top_p");
                }
                if entry.value.isNull == false {
                    decodedTopPValue = try JsonWireValue.extractFloat32(entry.value);
                }
            case "stream":
                if decodedStreamFlag != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "stream");
                }
                decodedStreamFlag = try JsonWireValue.extractBool(entry.value);
            case "thinking_budget":
                if decodedThinkingBudgetValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "thinking_budget");
                }
                if entry.value.isNull == false {
                    decodedThinkingBudgetValue = try JsonWireValue.clampToUInt32(
                        try JsonWireValue.extractUInt64(entry.value));
                }
            case "thinking_token_budget":
                if decodedThinkingTokenBudgetValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "thinking_token_budget");
                }
                if entry.value.isNull == false {
                    decodedThinkingTokenBudgetValue = try JsonWireValue.clampToUInt32(
                        try JsonWireValue.extractUInt64(entry.value));
                }
            case "thinking_budget_tokens":
                if decodedThinkingBudgetTokensValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "thinking_budget_tokens");
                }
                if entry.value.isNull == false {
                    decodedThinkingBudgetTokensValue = try JsonWireValue.clampToUInt32(
                        try JsonWireValue.extractUInt64(entry.value));
                }
            case "response_format":
                if decodedResponseFormatValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "response_format");
                }
                if entry.value.isNull == false {
                    decodedResponseFormatValue = try OpenAiResponseFormat.decoded(
                        wireValue: entry.value);
                }
            case "structured_outputs":
                if decodedStructuredOutputsValue != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "structured_outputs");
                }
                if entry.value.isNull == false {
                    decodedStructuredOutputsValue = try OpenAiStructuredOutputs.decoded(
                        wireValue: entry.value);
                }
            case "guided_grammar":
                if decodedGuidedGrammarText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "guided_grammar");
                }
                if entry.value.isNull == false {
                    decodedGuidedGrammarText = try JsonWireValue.extractString(entry.value);
                }
            default:
                unknownFields.append(OpenAiUnknownRequestField(
                    fieldName: entry.key, fieldValue: entry.value));
            }
        }
        guard let resolvedModelText: String = decodedModelText else {
            throw JsonWireProblem.missingField(fieldName: "model");
        }
        guard let resolvedInput: OpenAiResponseInput = decodedInput else {
            throw JsonWireProblem.missingField(fieldName: "input");
        }
        // Rust collects flattened unknown fields in a BTreeMap, which orders
        // keys by UTF-8 bytes; reproduce that ordering for the first-key rule.
        unknownFields.sort(by: { (leftField: OpenAiUnknownRequestField, rightField: OpenAiUnknownRequestField) -> Bool in
            return Array(leftField.fieldName.utf8).lexicographicallyPrecedes(Array(rightField.fieldName.utf8));
        });
        return OpenAiResponsesRequest(
            modelText: resolvedModelText,
            inputValue: resolvedInput,
            instructionsText: decodedInstructionsText,
            declaredTools: decodedTools ?? Array<OpenAiResponseToolDefinition>(),
            toolChoiceSelection: decodedToolChoice,
            metadataEntries: decodedMetadataEntries ?? Array<OpenAiMetadataEntry>(),
            storeFlag: decodedStoreFlag,
            backgroundFlag: decodedBackgroundFlag,
            truncationMode: decodedTruncationMode,
            serviceTierName: decodedServiceTierName,
            userLabel: decodedUserLabel,
            safetyIdentifier: decodedSafetyIdentifier,
            promptCacheKey: decodedPromptCacheKey,
            previousResponseIdentifier: decodedPreviousResponseIdentifier,
            conversationValue: decodedConversationValue,
            contextManagementValue: decodedContextManagementValue,
            includeValue: decodedIncludeValue,
            moderationValue: decodedModerationValue,
            promptValue: decodedPromptValue,
            promptCacheOptionsValue: decodedPromptCacheOptionsValue,
            promptCacheRetentionMode: decodedPromptCacheRetentionMode,
            reasoningValue: decodedReasoningValue,
            reasoningEffortName: decodedReasoningEffortName,
            enableThinkingFlag: decodedEnableThinkingFlag,
            chatTemplateKwargsValue: decodedChatTemplateKwargsValue,
            streamOptionsValue: decodedStreamOptionsValue,
            textValue: decodedTextValue,
            topLogprobsCount: decodedTopLogprobsCount,
            parallelToolCallsFlag: decodedParallelToolCallsFlag,
            maxOutputTokensBudget: decodedMaxOutputTokensBudget,
            temperatureValue: decodedTemperatureValue,
            topPValue: decodedTopPValue,
            streamFlag: decodedStreamFlag ?? false,
            thinkingBudgetValue: decodedThinkingBudgetValue,
            thinkingTokenBudgetValue: decodedThinkingTokenBudgetValue,
            thinkingBudgetTokensValue: decodedThinkingBudgetTokensValue,
            responseFormatValue: decodedResponseFormatValue,
            structuredOutputsValue: decodedStructuredOutputsValue,
            guidedGrammarText: decodedGuidedGrammarText,
            unknownFields: unknownFields);
    }

}

/// Module-private decode helpers mirroring the Rust private functions and the
/// serde flattened-map behavior of the Responses request struct.
fileprivate enum ResponsesRequestWire {

    /// Decodes `metadata: BTreeMap<String, String>`: an object with string
    /// values, kept byte-sorted by key for BTreeMap iteration parity.
    fileprivate static func decodedMetadataEntries(
        _ metadataWireValue: JsonWireValue) throws -> Array<OpenAiMetadataEntry> {
        let metadataObject: JsonWireObject = try JsonWireValue.extractObject(metadataWireValue);
        var metadataEntries: Array<OpenAiMetadataEntry> = Array<OpenAiMetadataEntry>();
        metadataEntries.reserveCapacity(metadataObject.entries.count);
        for metadataField: (key: String, value: JsonWireValue) in metadataObject.entries {
            metadataEntries.append(OpenAiMetadataEntry(
                metadataKey: metadataField.key,
                metadataValue: try JsonWireValue.extractString(metadataField.value)));
        }
        metadataEntries.sort(by: { (leftEntry: OpenAiMetadataEntry, rightEntry: OpenAiMetadataEntry) -> Bool in
            return Array(leftEntry.metadataKey.utf8)
                .lexicographicallyPrecedes(Array(rightEntry.metadataKey.utf8));
        });
        return metadataEntries;
    }
}
