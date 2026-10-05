import Foundation;
import IpcProtocol;

/// Serde decode for the Chat Completions request, split from the request
/// owner file at the 500-line marker. Mirrors the derived Deserialize of
/// crates/rest-contract/src/openai_chat_completion_request.rs: unknown
/// top-level fields flatten into a byte-ordered map (#772), Optional fields
/// accept missing or null, and defaulted booleans false when absent.
extension OpenAiChatCompletionRequest {

    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiChatCompletionRequest {
        let requestObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let knownFieldNames: Array<String> = [
            "model", "messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens",
            "temperature", "top_p", "frequency_penalty", "presence_penalty", "store",
            "reasoning_effort", "thinking_budget", "thinking_token_budget",
            "thinking_budget_tokens", "reasoning", "enable_thinking", "chat_template_kwargs",
            "stop", "seed", "stream", "stream_options", "response_format",
            "structured_outputs", "guided_grammar"];
        var unknownFields: Array<(fieldName: String, fieldValue: JsonWireValue)> = Array();
        for propertyName: String in requestObject.keyNames {
            if knownFieldNames.contains(propertyName) == false {
                unknownFields.append((propertyName, requestObject.value(forKey: propertyName)!));
            }
        }
        // The Rust flatten collects unknowns in a BTreeMap, so both equality
        // and the absorbed order follow UTF-8 byte ordering.
        unknownFields.sort { (leftEntry: (fieldName: String, fieldValue: JsonWireValue), rightEntry: (fieldName: String, fieldValue: JsonWireValue)) -> Bool in
            return Array(leftEntry.fieldName.utf8).lexicographicallyPrecedes(Array(rightEntry.fieldName.utf8));
        };
        return OpenAiChatCompletionRequest(
            model: try requestObject.decodeString(fieldName: "model"),
            messages: try requestObject.decodeArray(
                fieldName: "messages",
                mappedElement: { (messageWireValue: JsonWireValue) throws -> OpenAiChatMessage in
                    return try OpenAiChatMessage.decoded(wireValue: messageWireValue);
                }),
            tools: try requestObject.decodeArrayAllowingAbsent(
                fieldName: "tools",
                mappedElement: { (toolWireValue: JsonWireValue) throws -> OpenAiToolDefinition in
                    return try OpenAiToolDefinition.decoded(wireValue: toolWireValue);
                }),
            toolChoice: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "tool_choice")
                .map({ (choiceWireValue: JsonWireValue) throws -> OpenAiToolChoice in
                    return try OpenAiToolChoice.decoded(wireValue: choiceWireValue);
                }),
            maxTokens: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "max_tokens"),
            maxCompletionTokens: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "max_completion_tokens"),
            temperature: try ChatCompletionRequestDecoding.optionalFloat32(requestObject, fieldName: "temperature"),
            topP: try ChatCompletionRequestDecoding.optionalFloat32(requestObject, fieldName: "top_p"),
            frequencyPenalty: try ChatCompletionRequestDecoding.optionalFloat32(requestObject, fieldName: "frequency_penalty"),
            presencePenalty: try ChatCompletionRequestDecoding.optionalFloat32(requestObject, fieldName: "presence_penalty"),
            store: try requestObject.decodeOptionalBoolAllowingAbsent(fieldName: "store"),
            reasoningEffort: try requestObject.decodeOptionalStringAllowingAbsent(fieldName: "reasoning_effort"),
            thinkingBudget: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "thinking_budget"),
            thinkingTokenBudget: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "thinking_token_budget"),
            thinkingBudgetTokens: try requestObject.decodeOptionalUInt32AllowingAbsent(fieldName: "thinking_budget_tokens"),
            reasoning: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "reasoning")
                .map({ (reasoningWireValue: JsonWireValue) throws -> ReasoningRequestObject in
                    return try ReasoningRequestObject.decoded(wireValue: reasoningWireValue);
                }),
            enableThinking: try requestObject.decodeOptionalBoolAllowingAbsent(fieldName: "enable_thinking"),
            chatTemplateKwargs: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "chat_template_kwargs")
                .map({ (kwargsWireValue: JsonWireValue) throws -> ChatTemplateKwargsRequestObject in
                    return try ChatTemplateKwargsRequestObject.decoded(wireValue: kwargsWireValue);
                }),
            stop: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "stop")
                .map({ (stopWireValue: JsonWireValue) throws -> OpenAiStopSequences in
                    return try OpenAiStopSequences.decoded(wireValue: stopWireValue);
                }),
            seed: try requestObject.decodeOptionalUInt64AllowingAbsent(fieldName: "seed"),
            stream: try requestObject.decodeBoolAllowingAbsent(fieldName: "stream"),
            streamOptions: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "stream_options")
                .map({ (optionsWireValue: JsonWireValue) throws -> OpenAiStreamOptions in
                    return try OpenAiStreamOptions.decoded(wireValue: optionsWireValue);
                }),
            responseFormat: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "response_format")
                .map({ (formatWireValue: JsonWireValue) throws -> OpenAiResponseFormat in
                    return try OpenAiResponseFormat.decoded(wireValue: formatWireValue);
                }),
            structuredOutputs: try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: "structured_outputs")
                .map({ (structuredWireValue: JsonWireValue) throws -> OpenAiStructuredOutputs in
                    return try OpenAiStructuredOutputs.decoded(wireValue: structuredWireValue);
                }),
            guidedGrammar: try requestObject.decodeOptionalStringAllowingAbsent(fieldName: "guided_grammar"),
            unknownFields: unknownFields);
    }
}

/// Option<f32> decoding: serde accepts missing, null, and every JSON number
/// shape for a float field, narrowing integers losslessly like serde_json.
enum ChatCompletionRequestDecoding {

    static func optionalFloat32(_ requestObject: JsonWireObject, fieldName propertyName: String) throws -> Float? {
        guard let fieldValue: JsonWireValue = try requestObject.decodeOptionalRawValueAllowingAbsent(fieldName: propertyName) else {
            return nil;
        }
        return try JsonWireValue.extractFloat32(fieldValue);
    }
}
