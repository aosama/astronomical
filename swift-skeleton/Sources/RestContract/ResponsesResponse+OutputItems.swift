// ResponsesResponse+OutputItems.swift — RestContract
//
// Output item shapes of crates/rest-contract/src/openai_responses_response.rs:
// the model-produced output items, content parts, and the function tool echo
// consumed by response objects and stream events.

import Foundation;
import IpcProtocol;

/// One model-produced output item, discriminated on the wire by `type`.
public enum OpenAiResponseOutputItem: Equatable {
    case reasoning(
        identifier: String, summaryEntries: Array<OpenAiResponseReasoningSummary>,
        contentEntries: Array<OpenAiResponseReasoningContent>,
        itemStatus: OpenAiResponseItemStatus);
    case message(
        identifier: String, role: String,
        contentParts: Array<OpenAiResponseOutputContent>,
        itemStatus: OpenAiResponseItemStatus);
    case functionCall(
        identifier: String, callId: String, functionName: String, argumentsJson: String,
        itemStatus: OpenAiResponseItemStatus);

    /// Creates the empty reasoning item opened by a streaming lifecycle event.
    public static func reasoningInProgress(id: String) -> OpenAiResponseOutputItem {
        return .reasoning(
            identifier: id,
            summaryEntries: Array<OpenAiResponseReasoningSummary>(),
            contentEntries: Array<OpenAiResponseReasoningContent>(),
            itemStatus: .inProgress);
    }

    /// Creates the empty assistant message opened by a streaming lifecycle event.
    public static func messageInProgress(id: String) -> OpenAiResponseOutputItem {
        return .message(
            identifier: id,
            role: "assistant",
            contentParts: Array<OpenAiResponseOutputContent>(),
            itemStatus: .inProgress);
    }

    /// Creates the empty function call opened by a streaming lifecycle event.
    public static func functionCallInProgress(
        id: String, callId: String, functionName: String) -> OpenAiResponseOutputItem {
        return .functionCall(
            identifier: id,
            callId: callId,
            functionName: functionName,
            argumentsJson: "",
            itemStatus: .inProgress);
    }

    /// Creates one completed reasoning item holding one summary text block.
    public static func reasoning(id: String, reasoningText: String) -> OpenAiResponseOutputItem {
        return .reasoning(
            identifier: id,
            summaryEntries: [OpenAiResponseReasoningSummary(
                summaryKind: "summary_text", text: reasoningText)],
            contentEntries: Array<OpenAiResponseReasoningContent>(),
            itemStatus: .completed);
    }

    /// Creates one completed assistant message holding one output text part.
    public static func message(id: String, outputText: String) -> OpenAiResponseOutputItem {
        return .message(
            identifier: id,
            role: "assistant",
            contentParts: [OpenAiResponseOutputContent.outputText(outputText: outputText)],
            itemStatus: .completed);
    }

    /// Creates one completed function call with its serialized arguments.
    public static func functionCall(
        id: String, callId: String, functionName: String, argumentsJson: String
    ) -> OpenAiResponseOutputItem {
        return .functionCall(
            identifier: id,
            callId: callId,
            functionName: functionName,
            argumentsJson: argumentsJson,
            itemStatus: .completed);
    }

    /// The first message content part's text, or `None` for other item kinds.
    internal func messageText() -> String? {
        switch self {
        case .message(_, _, let contentParts, _):
            guard let firstContentPart: OpenAiResponseOutputContent = contentParts.first else {
                return nil;
            }
            return firstContentPart.text;
        case .reasoning, .functionCall:
            return nil;
        }
    }

    internal mutating func markIncomplete() -> Void {
        switch self {
        case .reasoning(let identifier, let summaryEntries, let contentEntries, _):
            self = .reasoning(
                identifier: identifier, summaryEntries: summaryEntries,
                contentEntries: contentEntries, itemStatus: .incomplete);
        case .message(let identifier, let role, let contentParts, _):
            self = .message(
                identifier: identifier, role: role, contentParts: contentParts,
                itemStatus: .incomplete);
        case .functionCall(let identifier, let callId, let functionName, let argumentsJson, _):
            self = .functionCall(
                identifier: identifier, callId: callId, functionName: functionName,
                argumentsJson: argumentsJson, itemStatus: .incomplete);
        }
    }

    /// Serializes the internally tagged item exactly as the serde derive does:
    /// the `type` tag first, then the variant fields in declaration order.
    public func wireValue() -> JsonWireValue {
        var itemObject: JsonWireObject = JsonWireObject(entries: Array());
        switch self {
        case .reasoning(let identifier, let summaryEntries, let contentEntries, let itemStatus):
            itemObject.appendEntry(key: "type", value: .string("reasoning"));
            itemObject.appendEntry(key: "id", value: .string(identifier));
            itemObject.appendEntry(
                key: "summary",
                value: JsonWireValue.mappedArray(
                    summaryEntries,
                    mappedWireValue: { (summaryEntry: OpenAiResponseReasoningSummary) -> JsonWireValue in
                        return summaryEntry.wireValue();
                    }));
            itemObject.appendEntry(
                key: "content",
                value: JsonWireValue.mappedArray(
                    contentEntries,
                    mappedWireValue: { (contentEntry: OpenAiResponseReasoningContent) -> JsonWireValue in
                        return contentEntry.wireValue();
                    }));
            itemObject.appendEntry(key: "status", value: itemStatus.wireValue());
        case .message(let identifier, let role, let contentParts, let itemStatus):
            itemObject.appendEntry(key: "type", value: .string("message"));
            itemObject.appendEntry(key: "id", value: .string(identifier));
            itemObject.appendEntry(key: "role", value: .string(role));
            itemObject.appendEntry(
                key: "content",
                value: JsonWireValue.mappedArray(
                    contentParts,
                    mappedWireValue: { (contentPart: OpenAiResponseOutputContent) -> JsonWireValue in
                        return contentPart.wireValue();
                    }));
            itemObject.appendEntry(key: "status", value: itemStatus.wireValue());
        case .functionCall(
            let identifier, let callId, let functionName, let argumentsJson, let itemStatus):
            itemObject.appendEntry(key: "type", value: .string("function_call"));
            itemObject.appendEntry(key: "id", value: .string(identifier));
            itemObject.appendEntry(key: "call_id", value: .string(callId));
            itemObject.appendEntry(key: "name", value: .string(functionName));
            itemObject.appendEntry(key: "arguments", value: .string(argumentsJson));
            itemObject.appendEntry(key: "status", value: itemStatus.wireValue());
        }
        return .object(itemObject);
    }
}

/// One reasoning summary block of a reasoning output item.
public struct OpenAiResponseReasoningSummary: Equatable {
    private let summaryKind: String;
    private let summaryText: String;

    fileprivate init(summaryKind: String, text: String) {
        self.summaryKind = summaryKind;
        self.summaryText = text;
    }

    /// Serializes with the serde `rename = "type"` spelling.
    public func wireValue() -> JsonWireValue {
        var summaryObject: JsonWireObject = JsonWireObject(entries: Array());
        summaryObject.appendEntry(key: "type", value: .string(self.summaryKind));
        summaryObject.appendEntry(key: "text", value: .string(self.summaryText));
        return .object(summaryObject);
    }
}

/// One reasoning content block of a reasoning output item. The Rust contract
/// declares this name twice (a private input-side enum and this public
/// struct) with one wire shape; one Swift module holds one type, so the
/// input-side decode lives here too.
public struct OpenAiResponseReasoningContent: Equatable {
    private let contentKind: String;
    private let reasoningText: String;

    internal init(contentKind: String, text: String) {
        self.contentKind = contentKind;
        self.reasoningText = text;
    }

    /// The visible reasoning text carried by this block.
    internal var text: String {
        return self.reasoningText;
    }

    /// Mirrors the serde derive of the input-side shape: the `type` tag must
    /// spell `reasoning_text`, the payload carries exactly one `text` field,
    /// and any other key is rejected.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseReasoningContent {
        let contentObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        _ = try contentObject.decodeTaggedVariantName(
            tagFieldName: "type", expectedVariantNames: ["reasoning_text"]);
        try contentObject.rejectUnknownFieldsBesidesTag(
            tagFieldName: "type", allowedFieldNames: ["text"]);
        return OpenAiResponseReasoningContent(
            contentKind: "reasoning_text",
            text: try contentObject.decodeString(fieldName: "text"));
    }

    /// Serializes with the serde `rename = "type"` spelling.
    public func wireValue() -> JsonWireValue {
        var contentObject: JsonWireObject = JsonWireObject(entries: Array());
        contentObject.appendEntry(key: "type", value: .string(self.contentKind));
        contentObject.appendEntry(key: "text", value: .string(self.reasoningText));
        return .object(contentObject);
    }
}

/// One assistant message content part of a message output item.
public struct OpenAiResponseOutputContent: Equatable {
    private let contentKind: String;
    private let textContent: String;
    private let annotationValues: Array<JsonWireValue>;
    private let logprobValues: Array<JsonWireValue>;

    fileprivate init(
        contentKind: String, text: String, annotations: Array<JsonWireValue>,
        logprobs: Array<JsonWireValue>) {
        self.contentKind = contentKind;
        self.textContent = text;
        self.annotationValues = annotations;
        self.logprobValues = logprobs;
    }

    /// The visible text fragment carried by this content part.
    internal var text: String {
        return self.textContent;
    }

    /// Creates one `output_text` part with no annotations and no logprobs.
    public static func outputText(outputText: String) -> OpenAiResponseOutputContent {
        return OpenAiResponseOutputContent(
            contentKind: "output_text",
            text: outputText,
            annotations: Array<JsonWireValue>(),
            logprobs: Array<JsonWireValue>());
    }

    /// Serializes with the serde `rename = "type"` spelling.
    public func wireValue() -> JsonWireValue {
        var partObject: JsonWireObject = JsonWireObject(entries: Array());
        partObject.appendEntry(key: "type", value: .string(self.contentKind));
        partObject.appendEntry(key: "text", value: .string(self.textContent));
        partObject.appendEntry(key: "annotations", value: .array(self.annotationValues));
        partObject.appendEntry(key: "logprobs", value: .array(self.logprobValues));
        return .object(partObject);
    }
}

/// The function declaration echoed inside a response's `tools` list.
public struct OpenAiResponseFunctionTool: Equatable {
    private let toolKind: String;
    private let functionName: String;
    private let functionDescription: String?;
    private let schemaParameters: JsonWireValue;
    private let strictSchemaFlag: Bool;

    fileprivate init(
        toolKind: String, functionName: String, functionDescription: String?,
        schemaParameters: JsonWireValue, strictSchemaFlag: Bool) {
        self.toolKind = toolKind;
        self.functionName = functionName;
        self.functionDescription = functionDescription;
        self.schemaParameters = schemaParameters;
        self.strictSchemaFlag = strictSchemaFlag;
    }

    /// Creates one function tool echo with the `function` tool type.
    public static func new(
        name: String, description: String?, parameters: JsonWireValue, strict: Bool
    ) -> OpenAiResponseFunctionTool {
        return OpenAiResponseFunctionTool(
            toolKind: "function",
            functionName: name,
            functionDescription: description,
            schemaParameters: parameters,
            strictSchemaFlag: strict);
    }

    /// The declared function name.
    internal var name: String {
        return self.functionName;
    }

    /// The optional declared function description.
    internal var description: String? {
        return self.functionDescription;
    }

    /// The declared JSON Schema for the function parameters.
    internal var parameters: JsonWireValue {
        return self.schemaParameters;
    }

    /// Whether the schema was declared strict.
    internal var strict: Bool {
        return self.strictSchemaFlag;
    }

    /// Serializes with the serde `rename = "type"` spelling and `None`
    /// description emitted as JSON null.
    public func wireValue() -> JsonWireValue {
        var toolObject: JsonWireObject = JsonWireObject(entries: Array());
        toolObject.appendEntry(key: "type", value: .string(self.toolKind));
        toolObject.appendEntry(key: "name", value: .string(self.functionName));
        if let functionDescription: String = self.functionDescription {
            toolObject.appendEntry(key: "description", value: .string(functionDescription));
        } else {
            toolObject.appendEntry(key: "description", value: .null);
        }
        toolObject.appendEntry(key: "parameters", value: self.schemaParameters);
        toolObject.appendEntry(key: "strict", value: .boolean(self.strictSchemaFlag));
        return .object(toolObject);
    }
}
