// ResponsesInput.swift — RestContract
//
// Port of crates/rest-contract/src/openai_responses_input.rs (wire shapes and
// decoding; the validated parts live in ResponsesInput+Parts.swift).

import Foundation;
import IpcProtocol;

/// Stateless input accepted by the local Responses endpoint.
public enum OpenAiResponseInput: Equatable {
    /// One compact user input string.
    case text(String);
    /// Ordered Responses input and prior-output items.
    case items(Array<OpenAiResponseInputItem>);

    /// Mirrors the untagged serde derive: the `String` alternative is tried
    /// first, then the item list; anything else fails with serde's wording.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseInput {
        if case let .string(inputText) = wireValue {
            return .text(inputText);
        }
        if case let .array(itemWireValues) = wireValue {
            var inputItems: Array<OpenAiResponseInputItem> = Array<OpenAiResponseInputItem>();
            inputItems.reserveCapacity(itemWireValues.count);
            for itemWireValue: JsonWireValue in itemWireValues {
                inputItems.append(try OpenAiResponseInputItem.decoded(wireValue: itemWireValue));
            }
            return .items(inputItems);
        }
        throw JsonWireProblem.malformedDocument(
            problem: "data did not match any variant of untagged enum OpenAiResponseInput");
    }
}

/// One input item supported by the local Responses endpoint.
public enum OpenAiResponseInputItem: Equatable {
    case message(OpenAiResponseMessageInput);
    case reasoning(OpenAiResponseReasoningInput);
    case functionCall(OpenAiResponseFunctionCallInput);
    case functionCallOutput(OpenAiResponseFunctionCallOutputInput);
    /// Any other item shape, retained raw so validation can name its type.
    case unsupported(JsonWireValue);

    /// Mirrors the untagged serde derive: each concrete alternative is tried
    /// in declaration order and the trailing `Value` alternative accepts
    /// anything, so decoding always succeeds.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseInputItem {
        // Empty catch bodies are the untagged fall-through: a rejected
        // alternative hands the value to the next one, exactly as serde does.
        do {
            return .message(try OpenAiResponseMessageInput.decoded(wireValue: wireValue));
        } catch { }
        do {
            return .reasoning(try OpenAiResponseReasoningInput.decoded(wireValue: wireValue));
        } catch { }
        do {
            return .functionCall(try OpenAiResponseFunctionCallInput.decoded(wireValue: wireValue));
        } catch { }
        do {
            return .functionCallOutput(
                try OpenAiResponseFunctionCallOutputInput.decoded(wireValue: wireValue));
        } catch { }
        return .unsupported(wireValue);
    }
}

/// One input message replayed into the conversation.
public struct OpenAiResponseMessageInput: Equatable {
    private let messageKind: OpenAiResponseMessageType?;
    private let itemIdentifier: String?;
    private let messageRole: OpenAiResponseMessageRole;
    private let messageContent: OpenAiResponseMessageContent;
    private let itemStatus: OpenAiResponseItemStatus?;
    private let assistantPhase: OpenAiResponseAssistantPhase?;

    fileprivate init(
        messageKind: OpenAiResponseMessageType?, itemIdentifier: String?,
        messageRole: OpenAiResponseMessageRole, messageContent: OpenAiResponseMessageContent,
        itemStatus: OpenAiResponseItemStatus?, assistantPhase: OpenAiResponseAssistantPhase?) {
        self.messageKind = messageKind;
        self.itemIdentifier = itemIdentifier;
        self.messageRole = messageRole;
        self.messageContent = messageContent;
        self.itemStatus = itemStatus;
        self.assistantPhase = assistantPhase;
    }

    /// The role carried by this input message.
    internal var role: OpenAiResponseMessageRole {
        return self.messageRole;
    }

    /// The message content, either compact text or typed parts.
    internal var content: OpenAiResponseMessageContent {
        return self.messageContent;
    }

    /// Mirrors the serde derive with `deny_unknown_fields`: exact wire field
    /// names, duplicate rejection, and missing-or-null `Option` to `None`.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseMessageInput {
        let inputObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try inputObject.rejectUnknownFields(allowedFieldNames: ["type", "id", "role", "content", "status", "phase"]);
        var decodedMessageKind: OpenAiResponseMessageType? = nil;
        var decodedItemIdentifier: String? = nil;
        var decodedMessageRole: OpenAiResponseMessageRole? = nil;
        var decodedMessageContent: OpenAiResponseMessageContent? = nil;
        var decodedItemStatus: OpenAiResponseItemStatus? = nil;
        var decodedAssistantPhase: OpenAiResponseAssistantPhase? = nil;
        for entry in inputObject.entries {
            switch entry.key {
            case "type":
                if decodedMessageKind != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "type");
                }
                if entry.value.isNull == false {
                    decodedMessageKind = try OpenAiResponseMessageType.decoded(wireValue: entry.value);
                }
            case "id":
                if decodedItemIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "id");
                }
                if entry.value.isNull == false {
                    decodedItemIdentifier = try JsonWireValue.extractString(entry.value);
                }
            case "role":
                if decodedMessageRole != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "role");
                }
                decodedMessageRole = try OpenAiResponseMessageRole.decoded(wireValue: entry.value);
            case "content":
                if decodedMessageContent != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "content");
                }
                decodedMessageContent = try OpenAiResponseMessageContent.decoded(wireValue: entry.value);
            case "status":
                if decodedItemStatus != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "status");
                }
                if entry.value.isNull == false {
                    decodedItemStatus = try OpenAiResponseItemStatus.decoded(wireValue: entry.value);
                }
            case "phase":
                if decodedAssistantPhase != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "phase");
                }
                if entry.value.isNull == false {
                    decodedAssistantPhase = try OpenAiResponseAssistantPhase.decoded(wireValue: entry.value);
                }
            default:
                throw JsonWireProblem.unknownField(
                    fieldName: entry.key,
                    expectedFields: ["type", "id", "role", "content", "status", "phase"]);
            }
        }
        guard let resolvedMessageRole: OpenAiResponseMessageRole = decodedMessageRole else {
            throw JsonWireProblem.missingField(fieldName: "role");
        }
        guard let resolvedMessageContent: OpenAiResponseMessageContent = decodedMessageContent else {
            throw JsonWireProblem.missingField(fieldName: "content");
        }
        return OpenAiResponseMessageInput(
            messageKind: decodedMessageKind,
            itemIdentifier: decodedItemIdentifier,
            messageRole: resolvedMessageRole,
            messageContent: resolvedMessageContent,
            itemStatus: decodedItemStatus,
            assistantPhase: decodedAssistantPhase);
    }
}

/// Recognized `type` spelling of a Responses input message.
internal enum OpenAiResponseMessageType: Equatable {
    case message;

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseMessageType {
        _ = try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["message"]);
        return .message;
    }
}

/// The role of one Responses input message.
internal enum OpenAiResponseMessageRole: Equatable {
    case user;
    case system;
    case developer;
    case assistant;

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseMessageRole {
        switch try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["user", "system", "developer", "assistant"]) {
        case "user": return .user;
        case "system": return .system;
        case "developer": return .developer;
        default: return .assistant;
        }
    }
}

/// Message content accepted on a Responses input item. Untagged on the wire:
/// a bare string or an array of typed content parts.
internal enum OpenAiResponseMessageContent: Equatable {
    /// A compact string message.
    case text(String);
    /// A list of explicitly typed content parts.
    case parts(Array<OpenAiResponseContentPart>);

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseMessageContent {
        if case let .string(textValue) = wireValue {
            return .text(textValue);
        }
        if case let .array(partWireValues) = wireValue {
            return .parts(try partWireValues.map({ (partWireValue: JsonWireValue) throws -> OpenAiResponseContentPart in
                return try OpenAiResponseContentPart.decoded(wireValue: partWireValue);
            }));
        }
        throw JsonWireProblem.malformedDocument(
            problem: "data did not match any variant of untagged enum OpenAiResponseMessageContent");
    }
}

/// A typed content part inside Responses input, discriminated by `type`.
internal enum OpenAiResponseContentPart: Equatable {
    case inputText(text: String);
    case outputText(text: String, annotations: Array<JsonWireValue>, logprobs: Array<JsonWireValue>);
    case inputImage(imageUrl: String, detail: String?);

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseContentPart {
        let partObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        let partType: String = try partObject.decodeTaggedVariantName(
            tagFieldName: "type",
            expectedVariantNames: ["input_text", "output_text", "input_image"]);
        switch partType {
        case "input_text":
            try partObject.rejectUnknownFieldsBesidesTag(tagFieldName: "type", allowedFieldNames: ["text"]);
            return .inputText(text: try partObject.decodeString(fieldName: "text"));
        case "output_text":
            try partObject.rejectUnknownFieldsBesidesTag(
                tagFieldName: "type", allowedFieldNames: ["text", "annotations", "logprobs"]);
            return .outputText(
                text: try partObject.decodeString(fieldName: "text"),
                annotations: try partObject.decodeArrayAllowingAbsent(
                    fieldName: "annotations",
                    mappedElement: { (annotationWireValue: JsonWireValue) -> JsonWireValue in
                        return annotationWireValue;
                    }),
                logprobs: try partObject.decodeArrayAllowingAbsent(
                    fieldName: "logprobs",
                    mappedElement: { (logprobWireValue: JsonWireValue) -> JsonWireValue in
                        return logprobWireValue;
                    }));
        default:
            try partObject.rejectUnknownFieldsBesidesTag(
                tagFieldName: "type", allowedFieldNames: ["image_url", "detail"]);
            return .inputImage(
                imageUrl: try partObject.decodeString(fieldName: "image_url"),
                detail: try partObject.decodeOptionalStringAllowingAbsent(fieldName: "detail"));
        }
    }
}

/// One replayed reasoning item.
public struct OpenAiResponseReasoningInput: Equatable {
    private let encryptedContentText: String?;
    private let summaryEntries: Array<OpenAiResponseReasoningSummaryInput>;
    private let contentEntries: Array<OpenAiResponseReasoningContent>;

    fileprivate init(
        encryptedContentText: String?,
        summaryEntries: Array<OpenAiResponseReasoningSummaryInput>,
        contentEntries: Array<OpenAiResponseReasoningContent>) {
        self.encryptedContentText = encryptedContentText;
        self.summaryEntries = summaryEntries;
        self.contentEntries = contentEntries;
    }

    /// Whether foreign encrypted reasoning content is present, which cannot
    /// be replayed locally.
    internal var hasEncryptedContent: Bool {
        return self.encryptedContentText != nil;
    }

    /// The replayed reasoning summary entries in declaration order.
    internal var summaries: Array<OpenAiResponseReasoningSummaryInput> {
        return self.summaryEntries;
    }

    /// The replayed reasoning content entries in declaration order.
    internal var contents: Array<OpenAiResponseReasoningContent> {
        return self.contentEntries;
    }

    /// Mirrors the serde derive with `deny_unknown_fields`; the `type` key is
    /// required to spell `reasoning`.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseReasoningInput {
        let reasoningObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        try reasoningObject.rejectUnknownFields(
            allowedFieldNames: ["type", "id", "summary", "content", "encrypted_content", "status"]);
        var decodedReasoningType: Bool = false;
        var decodedItemIdentifier: String? = nil;
        var decodedSummaryEntries: Array<OpenAiResponseReasoningSummaryInput>? = nil;
        var decodedContentEntries: Array<OpenAiResponseReasoningContent>? = nil;
        var decodedEncryptedContentText: String? = nil;
        var decodedItemStatus: OpenAiResponseItemStatus? = nil;
        for entry in reasoningObject.entries {
            switch entry.key {
            case "type":
                if decodedReasoningType {
                    throw JsonWireProblem.duplicateField(fieldName: "type");
                }
                decodedReasoningType = true;
                _ = try OpenAiResponseReasoningType.decoded(wireValue: entry.value);
            case "id":
                if decodedItemIdentifier != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "id");
                }
                if entry.value.isNull == false {
                    decodedItemIdentifier = try JsonWireValue.extractString(entry.value);
                }
            case "summary":
                if decodedSummaryEntries != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "summary");
                }
                decodedSummaryEntries = try JsonWireValue.extractArray(
                    entry.value,
                    mappedElement: { (summaryWireValue: JsonWireValue) throws -> OpenAiResponseReasoningSummaryInput in
                        return try OpenAiResponseReasoningSummaryInput.decoded(wireValue: summaryWireValue);
                    });
            case "content":
                if decodedContentEntries != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "content");
                }
                decodedContentEntries = try JsonWireValue.extractArray(
                    entry.value,
                    mappedElement: { (contentWireValue: JsonWireValue) throws -> OpenAiResponseReasoningContent in
                        return try OpenAiResponseReasoningContent.decoded(wireValue: contentWireValue);
                    });
            case "encrypted_content":
                if decodedEncryptedContentText != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "encrypted_content");
                }
                if entry.value.isNull == false {
                    decodedEncryptedContentText = try JsonWireValue.extractString(entry.value);
                }
            case "status":
                if decodedItemStatus != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "status");
                }
                if entry.value.isNull == false {
                    decodedItemStatus = try OpenAiResponseItemStatus.decoded(wireValue: entry.value);
                }
            default:
                throw JsonWireProblem.unknownField(
                    fieldName: entry.key,
                    expectedFields: ["type", "id", "summary", "content", "encrypted_content", "status"]);
            }
        }
        if decodedReasoningType == false {
            throw JsonWireProblem.missingField(fieldName: "type");
        }
        return OpenAiResponseReasoningInput(
            encryptedContentText: decodedEncryptedContentText,
            summaryEntries: decodedSummaryEntries ?? Array<OpenAiResponseReasoningSummaryInput>(),
            contentEntries: decodedContentEntries ?? Array<OpenAiResponseReasoningContent>());
    }
}

/// Recognized `type` spelling of a Responses reasoning input item.
internal enum OpenAiResponseReasoningType: Equatable {
    case reasoning;

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseReasoningType {
        _ = try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["reasoning"]);
        return .reasoning;
    }
}

/// One reasoning content block inside replayed reasoning input; the shared
/// `OpenAiResponseReasoningContent` type in
/// ResponsesResponse+OutputItems.swift carries both wire directions.

/// One reasoning summary block inside replayed reasoning input.
internal enum OpenAiResponseReasoningSummaryInput: Equatable {
    case summaryText(text: String);

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseReasoningSummaryInput {
        let summaryObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        _ = try summaryObject.decodeTaggedVariantName(
            tagFieldName: "type", expectedVariantNames: ["summary_text"]);
        try summaryObject.rejectUnknownFieldsBesidesTag(tagFieldName: "type", allowedFieldNames: ["text"]);
        return .summaryText(text: try summaryObject.decodeString(fieldName: "text"));
    }
}

/// Assistant phase annotation accepted on replayed assistant input items.
internal enum OpenAiResponseAssistantPhase: Equatable {
    case commentary;
    case finalAnswer;

    fileprivate static func decoded(wireValue: JsonWireValue) throws -> OpenAiResponseAssistantPhase {
        switch try ResponsesInputWire.decodedVariantName(
            wireValue, expectedVariantNames: ["commentary", "final_answer"]) {
        case "commentary": return .commentary;
        default: return .finalAnswer;
        }
    }
}

/// Module-private decode helpers mirroring serde behavior for the Responses
/// input shapes. Internal so the split function-call input file shares them.
internal enum ResponsesInputWire {

    /// serde's plain (non-tagged) enum field decoding with its exact wording:
    /// a single expected variant names only itself, several name the full set.
    internal static func decodedVariantName(
        _ wireValue: JsonWireValue, expectedVariantNames: Array<String>) throws -> String {
        let variantName: String = try JsonWireValue.extractString(wireValue);
        if expectedVariantNames.contains(variantName) {
            return variantName;
        }
        if expectedVariantNames.count == 1 {
            throw JsonWireProblem.malformedDocument(
                problem: "unknown variant `\(variantName)`, expected `\(expectedVariantNames[0])`");
        }
        throw JsonWireProblem.malformedDocument(
            problem: "unknown variant `\(variantName)`, expected one of \(JsonWireProblem.formattedFieldList(expectedVariantNames))");
    }
}
