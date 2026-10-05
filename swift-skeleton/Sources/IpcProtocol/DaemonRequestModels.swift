import Foundation;

/// One request an ephemeral CLI process sends to the local daemon.
/// Wire shape is internally tagged with `kind` in snake_case.
public enum DaemonRequest: Equatable {
    case handshake;
    /// Reports the daemon's current availability and resident model.
    case status;
    /// Streams one chat completion from the resident model. The daemon
    /// fills tools, tool choice, and the Qwen thinking seed itself: the
    /// CLI surface never leases those capabilities. Structured generation
    /// is the exception the caller owns: the CLI sends raw schema JSON
    /// here and the daemon validates it into the enforced worker
    /// constraint before the worker ever sees it.
    case chatGenerate(model: String, messages: Array<ChatMessage>, settings: ChatGenerationSettings, schemaJson: String?);
    /// Computes one embedding batch. A `nil` model means "use whatever is
    /// resident"; the daemon swaps models itself when the requested model
    /// differs from the loaded one.
    case embedGenerate(model: String?, inputs: Array<String>, dimensions: UInt32?);
    /// Lists the models discovered on this Mac with their capability flags.
    case modelsList;
    /// Lists the release download catalog with ready/download state per entry.
    case catalog;
    /// Starts (or resumes a matching paused) download for one catalog entry.
    /// Accepts the requestable model id or the full huggingface id.
    case downloadStart(modelId: String);
    /// Reports the active library download job, if any.
    case downloadStatus;
    /// Persists a new default model id for one-shot CLI verbs.
    case defaultModelSet(modelId: String);

    private static let expectedVariantNames: Array<String> = [
        "handshake", "status", "chat_generate", "embed_generate", "models_list", "catalog",
        "download_start", "download_status", "default_model_set",
    ];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case .handshake:
            return .object(DaemonRequest.tagOnlyWireObject(variantName: "handshake"));
        case .status:
            return .object(DaemonRequest.tagOnlyWireObject(variantName: "status"));
        case let .chatGenerate(model, messages, settings, schemaJson):
            var wireObject = DaemonRequest.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("chat_generate"));
            wireObject.appendEntry(key: "model", value: .string(model));
            wireObject.appendEntry(key: "messages", value: JsonWireValue.mappedArray(messages, mappedWireValue: { (message: ChatMessage) -> JsonWireValue in message.wireValue() }));
            wireObject.appendEntry(key: "settings", value: settings.wireValue());
            wireObject.appendEntry(key: "schema_json", value: DaemonRequest.optionalStringWireValue(schemaJson));
            return .object(wireObject);
        case let .embedGenerate(model, inputs, dimensions):
            var wireObject = DaemonRequest.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("embed_generate"));
            wireObject.appendEntry(key: "model", value: DaemonRequest.optionalStringWireValue(model));
            wireObject.appendEntry(key: "inputs", value: DaemonRequest.stringArrayWireValue(inputs));
            wireObject.appendEntry(key: "dimensions", value: DaemonRequest.optionalUInt32WireValue(dimensions));
            return .object(wireObject);
        case .modelsList:
            return .object(DaemonRequest.tagOnlyWireObject(variantName: "models_list"));
        case .catalog:
            return .object(DaemonRequest.tagOnlyWireObject(variantName: "catalog"));
        case let .downloadStart(modelId):
            var wireObject = DaemonRequest.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("download_start"));
            wireObject.appendEntry(key: "model_id", value: .string(modelId));
            return .object(wireObject);
        case .downloadStatus:
            return .object(DaemonRequest.tagOnlyWireObject(variantName: "download_status"));
        case let .defaultModelSet(modelId):
            var wireObject = DaemonRequest.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("default_model_set"));
            wireObject.appendEntry(key: "model_id", value: .string(modelId));
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> DaemonRequest {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: DaemonRequest.expectedVariantNames) {
        case "handshake":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .handshake;
        case "status":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .status;
        case "chat_generate":
            let parsedRequest = DaemonRequest.chatGenerate(
                model: try wireObject.decodeString(fieldName: "model"),
                messages: try wireObject.decodeArray(fieldName: "messages", mappedElement: { (elementWireValue: JsonWireValue) throws -> ChatMessage in
                    try ChatMessage.fromWireValue(elementWireValue)
                }),
                settings: try ChatGenerationSettings.fromWireValue(try wireObject.requireObjectValue(fieldName: "settings")),
                schemaJson: try wireObject.decodeOptionalString(fieldName: "schema_json"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model", "messages", "settings", "schema_json"]);
            return parsedRequest;
        case "embed_generate":
            let parsedRequest = DaemonRequest.embedGenerate(
                model: try wireObject.decodeOptionalString(fieldName: "model"),
                inputs: try wireObject.decodeArray(fieldName: "inputs", mappedElement: { (elementWireValue: JsonWireValue) throws -> String in
                    try JsonWireValue.extractString(elementWireValue)
                }),
                dimensions: try wireObject.decodeOptionalUInt32(fieldName: "dimensions"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model", "inputs", "dimensions"]);
            return parsedRequest;
        case "models_list":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .modelsList;
        case "catalog":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .catalog;
        case "download_start":
            let parsedRequest = DaemonRequest.downloadStart(modelId: try wireObject.decodeString(fieldName: "model_id"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model_id"]);
            return parsedRequest;
        case "download_status":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .downloadStatus;
        default:
            let parsedRequest = DaemonRequest.defaultModelSet(modelId: try wireObject.decodeString(fieldName: "model_id"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model_id"]);
            return parsedRequest;
        }
    }

    private static func emptyWireObject() -> JsonWireObject {
        return JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
    }

    private static func tagOnlyWireObject(variantName: String) -> JsonWireObject {
        var wireObject = DaemonRequest.emptyWireObject();
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        return wireObject;
    }

    private static func optionalStringWireValue(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }

    private static func optionalUInt32WireValue(_ optionalValue: UInt32?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(UInt64(unwrappedValue));
    }

    private static func stringArrayWireValue(_ textValues: Array<String>) -> JsonWireValue {
        return JsonWireValue.mappedArray(textValues, mappedWireValue: { (textValue: String) -> JsonWireValue in .string(textValue) });
    }
}
