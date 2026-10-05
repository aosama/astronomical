import Foundation;
import IpcProtocol;

/// One OpenAI-compatible Server-Sent Events chat completion chunk.
/// Serialize-only port of the chunk half of
/// crates/rest-contract/src/openai_chat_completion_response.rs; wire keys
/// follow the exact Rust declaration order and `skip_serializing_if` rules.
public struct OpenAiChatCompletionChunk: Equatable {
    private let identifier: String;
    private let objectKind: String;
    private let createdTimestamp: UInt64;
    private let modelText: String;
    private let choiceList: Array<OpenAiChatCompletionChunkChoice>;
    private var usageAccount: OpenAiTokenUsage?;

    internal init(
        id: String, created: UInt64, model: String,
        delta: OpenAiChatCompletionDelta, finishReason: OpenAiFinishReason?) {
        self.identifier = id;
        self.objectKind = "chat.completion.chunk";
        self.createdTimestamp = created;
        self.modelText = model;
        self.choiceList = [OpenAiChatCompletionChunkChoice(
            index: 0, delta: delta, finishReason: finishReason)];
        self.usageAccount = nil;
    }

    /// Creates the initial assistant-role chunk for one stream.
    public static func assistantRole(id: String, created: UInt64, model: String) -> OpenAiChatCompletionChunk {
        return OpenAiChatCompletionChunk(
            id: id, created: created, model: model,
            delta: OpenAiChatCompletionDelta(
                role: "assistant", content: nil, reasoningContent: nil,
                toolCalls: Array<OpenAiToolCallDelta>()),
            finishReason: nil);
    }

    /// Creates one generated text delta.
    public static func textDelta(id: String, created: UInt64, model: String, text: String) -> OpenAiChatCompletionChunk {
        return OpenAiChatCompletionChunk(
            id: id, created: created, model: model,
            delta: OpenAiChatCompletionDelta(
                role: nil, content: text, reasoningContent: nil,
                toolCalls: Array<OpenAiToolCallDelta>()),
            finishReason: nil);
    }

    /// Creates one generated reasoning delta.
    public static func reasoningDelta(
        id: String, created: UInt64, model: String, reasoningContent: String
    ) -> OpenAiChatCompletionChunk {
        return OpenAiChatCompletionChunk(
            id: id, created: created, model: model,
            delta: OpenAiChatCompletionDelta(
                role: nil, content: nil, reasoningContent: reasoningContent,
                toolCalls: Array<OpenAiToolCallDelta>()),
            finishReason: nil);
    }

    /// Creates one complete tool-call delta at its stable output index.
    public static func toolCallDelta(
        id: String, created: UInt64, model: String, toolCallIndex: UInt16,
        toolCallId: String, functionName: String, functionArguments: String
    ) -> OpenAiChatCompletionChunk {
        return OpenAiChatCompletionChunk(
            id: id, created: created, model: model,
            delta: OpenAiChatCompletionDelta(
                role: nil, content: nil, reasoningContent: nil,
                toolCalls: [OpenAiToolCallDelta(
                    index: toolCallIndex,
                    id: toolCallId,
                    function: OpenAiToolCallFunctionDelta(
                        name: functionName, arguments: functionArguments))]),
            finishReason: nil);
    }

    /// Creates the terminal chunk with the public completion reason.
    public static func finished(
        id: String, created: UInt64, model: String, finishReason: OpenAiFinishReason
    ) -> OpenAiChatCompletionChunk {
        return OpenAiChatCompletionChunk(
            id: id, created: created, model: model,
            delta: OpenAiChatCompletionDelta.empty(),
            finishReason: finishReason);
    }

    /// Adds token usage to a terminal chunk when the client requested it.
    public func withUsage(usage: OpenAiTokenUsage) -> OpenAiChatCompletionChunk {
        var updatedChunk: OpenAiChatCompletionChunk = self;
        updatedChunk.usageAccount = usage;
        return updatedChunk;
    }

    public func wireValue() -> JsonWireValue {
        var chunkObject: JsonWireObject = JsonWireObject(entries: Array());
        chunkObject.appendEntry(key: "id", value: .string(self.identifier));
        chunkObject.appendEntry(key: "object", value: .string(self.objectKind));
        chunkObject.appendEntry(key: "created", value: .unsignedInteger(self.createdTimestamp));
        chunkObject.appendEntry(key: "model", value: .string(self.modelText));
        chunkObject.appendEntry(
            key: "choices",
            value: JsonWireValue.mappedArray(
                self.choiceList,
                mappedWireValue: { (streamChoice: OpenAiChatCompletionChunkChoice) -> JsonWireValue in
                    return streamChoice.wireValue();
                }));
        if let terminalUsage: OpenAiTokenUsage = self.usageAccount {
            chunkObject.appendEntry(key: "usage", value: terminalUsage.wireValue());
        }
        return .object(chunkObject);
    }
}

/// One complete non-streaming OpenAI-compatible chat completion response.
public struct OpenAiChatCompletionResponse: Equatable {
    private let identifier: String;
    private let objectKind: String;
    private let createdTimestamp: UInt64;
    private let modelText: String;
    private let choiceList: Array<OpenAiChatCompletionChoice>;
    private let usageAccount: OpenAiTokenUsage;

    /// Creates the single-choice response returned by the initial local endpoint.
    public init(
        id: String, created: UInt64, model: String, message: OpenAiAssistantMessage,
        finishReason: OpenAiFinishReason, usage: OpenAiTokenUsage) {
        self.identifier = id;
        self.objectKind = "chat.completion";
        self.createdTimestamp = created;
        self.modelText = model;
        self.choiceList = [OpenAiChatCompletionChoice(
            index: 0, message: message, finishReason: finishReason)];
        self.usageAccount = usage;
    }

    public func wireValue() -> JsonWireValue {
        var responseObject: JsonWireObject = JsonWireObject(entries: Array());
        responseObject.appendEntry(key: "id", value: .string(self.identifier));
        responseObject.appendEntry(key: "object", value: .string(self.objectKind));
        responseObject.appendEntry(key: "created", value: .unsignedInteger(self.createdTimestamp));
        responseObject.appendEntry(key: "model", value: .string(self.modelText));
        responseObject.appendEntry(
            key: "choices",
            value: JsonWireValue.mappedArray(
                self.choiceList,
                mappedWireValue: { (completionChoice: OpenAiChatCompletionChoice) -> JsonWireValue in
                    return completionChoice.wireValue();
                }));
        responseObject.appendEntry(key: "usage", value: self.usageAccount.wireValue());
        return .object(responseObject);
    }
}

/// The one complete choice in an initial non-streaming response.
public struct OpenAiChatCompletionChoice: Equatable {
    private let choiceIndex: UInt8;
    private let assistantMessage: OpenAiAssistantMessage;
    private let finishReasonName: OpenAiFinishReason;

    internal init(index: UInt8, message: OpenAiAssistantMessage, finishReason: OpenAiFinishReason) {
        self.choiceIndex = index;
        self.assistantMessage = message;
        self.finishReasonName = finishReason;
    }

    public func wireValue() -> JsonWireValue {
        var choiceObject: JsonWireObject = JsonWireObject(entries: Array());
        choiceObject.appendEntry(key: "index", value: .unsignedInteger(UInt64(self.choiceIndex)));
        choiceObject.appendEntry(key: "message", value: self.assistantMessage.wireValue());
        choiceObject.appendEntry(key: "finish_reason", value: self.finishReasonName.wireValue());
        return .object(choiceObject);
    }
}

/// Complete assistant output assembled from ordered worker events.
public struct OpenAiAssistantMessage: Equatable {
    private let roleLabel: String;
    private let contentText: String?;
    private let reasoningContentText: String?;
    private let toolCallList: Array<OpenAiResponseToolCall>;

    /// Creates one assistant message from bounded response parts.
    public init(content: String?, reasoningContent: String?, toolCalls: Array<OpenAiResponseToolCall>) {
        self.roleLabel = "assistant";
        self.contentText = content;
        self.reasoningContentText = reasoningContent;
        self.toolCallList = toolCalls;
    }

    public func wireValue() -> JsonWireValue {
        var messageObject: JsonWireObject = JsonWireObject(entries: Array());
        messageObject.appendEntry(key: "role", value: .string(self.roleLabel));
        if let visibleContent: String = self.contentText {
            messageObject.appendEntry(key: "content", value: .string(visibleContent));
        } else {
            messageObject.appendEntry(key: "content", value: .null);
        }
        if let reasoningContent: String = self.reasoningContentText {
            messageObject.appendEntry(key: "reasoning_content", value: .string(reasoningContent));
        }
        if self.toolCallList.isEmpty == false {
            messageObject.appendEntry(
                key: "tool_calls",
                value: JsonWireValue.mappedArray(
                    self.toolCallList,
                    mappedWireValue: { (responseToolCall: OpenAiResponseToolCall) -> JsonWireValue in
                        return responseToolCall.wireValue();
                    }));
        }
        return .object(messageObject);
    }
}

/// One complete function call in a non-streaming assistant message.
public struct OpenAiResponseToolCall: Equatable {
    private let identifier: String;
    private let toolTypeName: String;
    private let functionData: OpenAiToolCallFunctionDelta;

    internal init(id: String, function: OpenAiToolCallFunctionDelta) {
        self.identifier = id;
        self.toolTypeName = "function";
        self.functionData = function;
    }

    /// Creates one complete OpenAI function call.
    public static func function(id: String, name: String, arguments: String) -> OpenAiResponseToolCall {
        return OpenAiResponseToolCall(
            id: id,
            function: OpenAiToolCallFunctionDelta(name: name, arguments: arguments));
    }

    public func wireValue() -> JsonWireValue {
        var toolCallObject: JsonWireObject = JsonWireObject(entries: Array());
        toolCallObject.appendEntry(key: "id", value: .string(self.identifier));
        toolCallObject.appendEntry(key: "type", value: .string(self.toolTypeName));
        toolCallObject.appendEntry(key: "function", value: self.functionData.wireValue());
        return .object(toolCallObject);
    }
}

/// The one supported choice in the initial single-request stream.
public struct OpenAiChatCompletionChunkChoice: Equatable {
    private let choiceIndex: UInt8;
    private let streamDelta: OpenAiChatCompletionDelta;
    private let finishReasonName: OpenAiFinishReason?;

    internal init(index: UInt8, delta: OpenAiChatCompletionDelta, finishReason: OpenAiFinishReason?) {
        self.choiceIndex = index;
        self.streamDelta = delta;
        self.finishReasonName = finishReason;
    }

    public func wireValue() -> JsonWireValue {
        var choiceObject: JsonWireObject = JsonWireObject(entries: Array());
        choiceObject.appendEntry(key: "index", value: .unsignedInteger(UInt64(self.choiceIndex)));
        choiceObject.appendEntry(key: "delta", value: self.streamDelta.wireValue());
        if let resolvedFinishReason: OpenAiFinishReason = self.finishReasonName {
            choiceObject.appendEntry(key: "finish_reason", value: resolvedFinishReason.wireValue());
        } else {
            choiceObject.appendEntry(key: "finish_reason", value: .null);
        }
        return .object(choiceObject);
    }
}

/// The incremental assistant output carried by one chunk.
public struct OpenAiChatCompletionDelta: Equatable {
    private let roleLabel: String?;
    private let contentText: String?;
    private let reasoningContentText: String?;
    private let toolCallList: Array<OpenAiToolCallDelta>;

    internal init(
        role: String?, content: String?, reasoningContent: String?,
        toolCalls: Array<OpenAiToolCallDelta>) {
        self.roleLabel = role;
        self.contentText = content;
        self.reasoningContentText = reasoningContent;
        self.toolCallList = toolCalls;
    }

    fileprivate static func empty() -> OpenAiChatCompletionDelta {
        return OpenAiChatCompletionDelta(
            role: nil, content: nil, reasoningContent: nil,
            toolCalls: Array<OpenAiToolCallDelta>());
    }

    public func wireValue() -> JsonWireValue {
        var deltaObject: JsonWireObject = JsonWireObject(entries: Array());
        if let streamRole: String = self.roleLabel {
            deltaObject.appendEntry(key: "role", value: .string(streamRole));
        }
        if let deltaContent: String = self.contentText {
            deltaObject.appendEntry(key: "content", value: .string(deltaContent));
        }
        if let deltaReasoningContent: String = self.reasoningContentText {
            deltaObject.appendEntry(key: "reasoning_content", value: .string(deltaReasoningContent));
        }
        if self.toolCallList.isEmpty == false {
            deltaObject.appendEntry(
                key: "tool_calls",
                value: JsonWireValue.mappedArray(
                    self.toolCallList,
                    mappedWireValue: { (deltaToolCall: OpenAiToolCallDelta) -> JsonWireValue in
                        return deltaToolCall.wireValue();
                    }));
        }
        return .object(deltaObject);
    }
}

/// One complete function call surfaced in an OpenAI stream chunk.
public struct OpenAiToolCallDelta: Equatable {
    private let toolCallIndex: UInt16;
    private let identifier: String;
    private let toolTypeName: String;
    private let functionData: OpenAiToolCallFunctionDelta;

    internal init(index: UInt16, id: String, function: OpenAiToolCallFunctionDelta) {
        self.toolCallIndex = index;
        self.identifier = id;
        self.toolTypeName = "function";
        self.functionData = function;
    }

    public func wireValue() -> JsonWireValue {
        var toolCallObject: JsonWireObject = JsonWireObject(entries: Array());
        toolCallObject.appendEntry(key: "index", value: .unsignedInteger(UInt64(self.toolCallIndex)));
        toolCallObject.appendEntry(key: "id", value: .string(self.identifier));
        toolCallObject.appendEntry(key: "type", value: .string(self.toolTypeName));
        toolCallObject.appendEntry(key: "function", value: self.functionData.wireValue());
        return .object(toolCallObject);
    }
}

/// The function data attached to an OpenAI tool-call delta.
public struct OpenAiToolCallFunctionDelta: Equatable {
    private let nameText: String;
    private let argumentsText: String;

    internal init(name: String, arguments: String) {
        self.nameText = name;
        self.argumentsText = arguments;
    }

    public func wireValue() -> JsonWireValue {
        var functionObject: JsonWireObject = JsonWireObject(entries: Array());
        functionObject.appendEntry(key: "name", value: .string(self.nameText));
        functionObject.appendEntry(key: "arguments", value: .string(self.argumentsText));
        return .object(functionObject);
    }
}

/// A terminal reason recognized by OpenAI-compatible chat clients.
public enum OpenAiFinishReason: Equatable {
    /// The model reached a normal end-of-sequence marker.
    case stop;
    /// The configured output-token cap was reached.
    case length;
    /// The model emitted one or more complete function calls.
    case toolCalls;

    public func wireValue() -> JsonWireValue {
        switch self {
        case .stop:
            return .string("stop");
        case .length:
            return .string("length");
        case .toolCalls:
            return .string("tool_calls");
        }
    }
}

/// Token accounting returned in complete responses and optionally on terminal stream chunks.
public struct OpenAiTokenUsage: Equatable {
    private let promptTokensCount: UInt32;
    private let completionTokensCount: UInt32;
    private let totalTokensCount: UInt32;
    private var promptTokenDetails: OpenAiPromptTokenDetails?;

    internal init(
        promptTokens: UInt32, completionTokens: UInt32, totalTokens: UInt32,
        promptTokenDetails: OpenAiPromptTokenDetails?) {
        self.promptTokensCount = promptTokens;
        self.completionTokensCount = completionTokens;
        self.totalTokensCount = totalTokens;
        self.promptTokenDetails = promptTokenDetails;
    }

    /// Builds checked token accounting from worker-observed counts.
    public static func new(promptTokens: UInt32, completionTokens: UInt32) -> OpenAiTokenUsage? {
        let (totalTokensCount, totalTokensOverflowed) =
            promptTokens.addingReportingOverflow(completionTokens);
        if totalTokensOverflowed {
            return nil;
        }
        return OpenAiTokenUsage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: totalTokensCount,
            promptTokenDetails: nil);
    }

    /// Attaches the number of prompt tokens served from the persistent cache.
    ///
    /// When zero, the `prompt_tokens_details` field is omitted entirely
    /// so existing clients see no change in the response shape.
    public func withCachedTokens(cachedTokens: UInt32) -> OpenAiTokenUsage {
        var updatedUsage: OpenAiTokenUsage = self;
        if cachedTokens > 0 {
            updatedUsage.promptTokenDetails = OpenAiPromptTokenDetails(cachedTokens: cachedTokens);
        } else {
            updatedUsage.promptTokenDetails = nil;
        }
        return updatedUsage;
    }

    public func wireValue() -> JsonWireValue {
        var usageObject: JsonWireObject = JsonWireObject(entries: Array());
        usageObject.appendEntry(key: "prompt_tokens", value: .unsignedInteger(UInt64(self.promptTokensCount)));
        usageObject.appendEntry(key: "completion_tokens", value: .unsignedInteger(UInt64(self.completionTokensCount)));
        usageObject.appendEntry(key: "total_tokens", value: .unsignedInteger(UInt64(self.totalTokensCount)));
        if let attachedPromptTokenDetails: OpenAiPromptTokenDetails = self.promptTokenDetails {
            usageObject.appendEntry(
                key: "prompt_tokens_details", value: attachedPromptTokenDetails.wireValue());
        }
        return .object(usageObject);
    }
}

/// Breakdown of prompt token costs, following the OpenAI convention.
internal struct OpenAiPromptTokenDetails: Equatable {
    private let cachedTokensCount: UInt32;

    internal init(cachedTokens: UInt32) {
        self.cachedTokensCount = cachedTokens;
    }

    public func wireValue() -> JsonWireValue {
        var detailsObject: JsonWireObject = JsonWireObject(entries: Array());
        detailsObject.appendEntry(key: "cached_tokens", value: .unsignedInteger(UInt64(self.cachedTokensCount)));
        return .object(detailsObject);
    }
}
